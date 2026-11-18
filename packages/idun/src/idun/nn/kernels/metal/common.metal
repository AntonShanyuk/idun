// idun_tabm and idun_realmlp: what the networks' kernels share (docs/architecture/networks.md). The host compiles
// "#define ACT_T ...", this file and one model's file (tabm.metal, rmlp.metal) as one library.
//
// Every reduction is a sequential loop in a fixed order (no atomics), so two runs give identical bits. Activations and
// their gradients are stored as act_t: float, or bfloat when the host compiles with ACT_T=bfloat (TabMParams.storage
// "bf16"); weights, optimizer moments, logits and losses stay float, and all arithmetic is float.

#include <metal_stdlib>
using namespace metal;

#define UNROLL _Pragma("clang loop unroll(full)")

#ifndef ACT_T
#define ACT_T float
#endif
typedef ACT_T act_t;

static uint lowbias32(uint x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM: C[M,N] = op(A)[M,K] · op(B)[K,N], fp32, adapted from MLX's Steel GEMM (MIT licence, © 2023 Apple Inc.; its
// notice is in NOTICE; docs/architecture/provenance.md): the tile loader's thread layout and the fragment's lane
// coordinates are Steel's. Tiles keep their stored layout in threadgroup memory, padded by 16 bytes per row; each
// lane loads its own two elements of an 8 x 8 fragment; fragments interleave across the SIMD-groups. "Scaled" operands
// are a BatchEnsemble product read on the fly: row q of the operand is x[q / div] ⊙ r[q % members].
// M, N must be multiples of 64 and K a multiple of 16.
// ---------------------------------------------------------------------------------------------------------------------

struct GemmParams {
  uint M, N, K;
  uint lda, ldb, ldc;
  uint members, div, ldr;  // the scaled operand: x[q / div] · r[q % members], r's row stride ldr
};

constant constexpr short BM = 64, BN = 64, BK = 16, WM = 2, WN = 2, THREADS = 128;
constant constexpr short PAD = 16 / sizeof(act_t);  // 16 bytes per tile row (both tiles of a GEMM use the same pad)

template <typename D, short ROWS, short COLS, short LD>
struct Loader {
  static constant constexpr const short VEC = ROWS * COLS / THREADS;
  static constant constexpr const short TCOLS = COLS / VEC;
  static constant constexpr const short TROWS = THREADS / TCOLS;
  short bi, bj;
  Loader(ushort tid) : bi(tid / TCOLS), bj(VEC * (tid % TCOLS)) {}
  template <typename S>
  void load(threadgroup D *dst, device const S *src, uint ld) const {
    UNROLL for (short i = 0; i < ROWS; i += TROWS) {
      device const vec<S, 4> *s = (device const vec<S, 4> *)(src + (bi + i) * ld + bj);
      threadgroup vec<D, 4> *d = (threadgroup vec<D, 4> *)(dst + (bi + i) * LD + bj);
      UNROLL for (short v = 0; v < VEC / 4; ++v) d[v] = vec<D, 4>(s[v]);
    }
  }
  // Rows row0 .., columns col0 .. of the scaled operand x[q / div] ⊙ r[q % members].
  template <typename S>
  void load_scaled(threadgroup D *dst, device const S *x, uint ldx, device const float *r, uint ldr, uint row0,
                   uint col0, uint members, uint div) const {
    UNROLL for (short i = 0; i < ROWS; i += TROWS) {
      const uint q = row0 + bi + i;
      device const vec<S, 4> *s = (device const vec<S, 4> *)(x + (q / div) * ldx + col0 + bj);
      device const float4 *f = (device const float4 *)(r + (q % members) * ldr + col0 + bj);
      threadgroup vec<D, 4> *d = (threadgroup vec<D, 4> *)(dst + (bi + i) * LD + bj);
      UNROLL for (short v = 0; v < VEC / 4; ++v) d[v] = vec<D, 4>(float4(s[v]) * f[v]);
    }
  }
};

// TA: A stored (K, M); TB: B stored (N, K); SA / SB: that operand is scaled (SA only without TA, SB only without TB).
// The output stage gets each lane's elements row by row: state = epi.begin_row(row), then
// epi.store(state, row, col, v0, v1) for columns col and col + 1.
template <typename AT, typename BT, bool TA, bool TB, bool SA, bool SB, typename Epi>
void gemm(device const AT *A, device const BT *B, device const float *R, constant GemmParams &p, uint2 tg, ushort sg,
          ushort lane, threadgroup AT *As, threadgroup BT *Bs, thread const Epi &epi) {
  constexpr short LDA = TA ? BM + PAD : BK + PAD;
  constexpr short LDB = TB ? BK + PAD : BN + PAD;
  constexpr short TM = BM / (8 * WM), TN = BN / (8 * WN);
  const uint m0 = tg.y * BM, n0 = tg.x * BN;
  Loader<AT, TA ? BK : BM, TA ? BM : BK, LDA> load_a(sg * 32 + lane);
  Loader<BT, TB ? BN : BK, TB ? BK : BN, LDB> load_b(sg * 32 + lane);
  const short qid = lane / 4;                       // Steel's get_coord (mma.h)
  const short fm = (qid & 4) + ((lane / 2) % 4);    // the lane's row in a fragment
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;  // its first column
  const short tm = 8 * (sg / WN), tn = 8 * (sg % WN);

  simdgroup_matrix<float, 8, 8> acc[TM][TN];
  UNROLL for (short i = 0; i < TM; ++i) UNROLL for (short j = 0; j < TN; ++j) acc[i][j] =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k0 = 0; k0 < p.K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (TA) {
      load_a.load(As, A + k0 * p.lda + m0, p.lda);
    } else if (SA) {
      load_a.load_scaled(As, A, p.lda, R, p.ldr, m0, k0, p.members, p.div);
    } else {
      load_a.load(As, A + m0 * p.lda + k0, p.lda);
    }
    if (TB) {
      load_b.load(Bs, B + n0 * p.ldb + k0, p.ldb);
    } else if (SB) {
      load_b.load_scaled(Bs, B, p.ldb, R, p.ldr, k0, n0, p.members, p.div);
    } else {
      load_b.load(Bs, B + k0 * p.ldb + n0, p.ldb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<float, 8, 8> a[TM], b[TN];
      UNROLL for (short i = 0; i < TM; ++i) {
        const short row = tm + i * 8 * WM + fm;
        thread auto &e = a[i].thread_elements();
        UNROLL for (short c = 0; c < 2; ++c) {
          const short k = kk + fn + c;
          e[c] = float(TA ? As[k * LDA + row] : As[row * LDA + k]);
        }
      }
      UNROLL for (short j = 0; j < TN; ++j) {
        const short col = tn + j * 8 * WN + fn;
        thread auto &e = b[j].thread_elements();
        UNROLL for (short c = 0; c < 2; ++c) {
          const short k = kk + fm;
          e[c] = float(TB ? Bs[(col + c) * LDB + k] : Bs[k * LDB + col + c]);
        }
      }
      UNROLL for (short i = 0; i < TM; ++i) UNROLL for (short j = 0; j < TN; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  UNROLL for (short i = 0; i < TM; ++i) {
    const uint row = m0 + tm + i * 8 * WM + fm;
    const uint state = epi.begin_row(row);
    UNROLL for (short j = 0; j < TN; ++j) {
      thread auto &e = acc[i][j].thread_elements();
      epi.store(state, row, n0 + tn + j * 8 * WN + fn, e[0], e[1]);
    }
  }
}

// Output stage: store the product (as T).
template <typename T>
struct Plain {
  device T *C;
  uint ldc;
  uint begin_row(uint) const { return 0; }
  void store(uint, uint row, uint col, float v0, float v1) const {
    C[row * ldc + col] = T(v0);
    C[row * ldc + col + 1] = T(v1);
  }
};

struct AdamParams {
  uint n;
  float decay;  // 1 - lr · weight_decay
  float beta1, beta2;
  float step_size;  // lr / (1 - beta1^t)
  float root_c2;    // sqrt(1 - beta2^t)
  float eps;
  float grad_scale;  // the gradients' factor: clip_grad_norm_'s clip coefficient, or 1 (exact: the same bits)
};

// torch.optim.AdamW: decoupled decay, then the bias-corrected update.
kernel void adamw(device float *param [[buffer(0)]], device const float *grad [[buffer(1)]],
                  device float *m1 [[buffer(2)]], device float *m2 [[buffer(3)]], constant AdamParams &p [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
  if (i >= p.n) return;
  const float g = grad[i] * p.grad_scale;
  const float m = p.beta1 * m1[i] + (1.0f - p.beta1) * g;
  const float v = p.beta2 * m2[i] + (1.0f - p.beta2) * g * g;
  m1[i] = m;
  m2[i] = v;
  param[i] = param[i] * p.decay - p.step_size * m / (sqrt(v) / p.root_c2 + p.eps);
}
