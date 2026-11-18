// idun_tabm and idun_realmlp on CUDA: what the networks' kernels share (docs/architecture/networks.md). The host
// compiles "#define ACT_BF16 1" (or nothing; TabM only), this file and one model's file (tabm.cu, rmlp.cu) as one
// program.
//
// The kernels compute what the Metal kernels of the same names compute (kernels/metal), within float32 error of the
// float64 reference: every reduction is a sequential loop in a fixed order (no atomics), so two runs give identical
// bits. The GEMM reads float operands, or with ACT_BF16 bfloat16 bits (16-bit words) as well; all arithmetic is
// float. NVRTC compiles without contraction (--fmad=false): a * b + c is a multiply and an add.

typedef unsigned int uint;
typedef unsigned short ushort;

static __device__ __forceinline__ float to_float(float v) { return v; }
#ifdef ACT_BF16
static __device__ __forceinline__ float to_float(ushort v) { return __uint_as_float(uint(v) << 16); }
#endif

static __device__ __forceinline__ uint lowbias32(uint x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// ---------------------------------------------------------------------------------------------------------------------
// GEMM: C[M,N] = op(A)[M,K] · op(B)[K,N], fp32. A block of 256 threads computes a 128 x 128 tile of C, each thread an
// 8 x 8 block of it: rows 4 · ty .. and 64 + 4 · ty .., columns 4 · tx .. and 64 + 4 · tx .. (tx, ty < 16), so its
// reads of the tiles in shared memory are whole float4s without bank conflicts. Each element is one sequential sum
// over k = 0, 1, ..., accumulated with a fused multiply-add (`fmaf`, deterministic and more exact than a multiply and
// an add; idun_gbm's kernels have none). The tiles go through shared memory k-major, 8 values of k at a time and
// double-buffered: the next tile is read from global memory into registers while this one is multiplied. "Scaled"
// operands are a BatchEnsemble product read on the fly: row q of the operand is x[q / div] ⊙ r[q % members]. M and N
// are multiples of 64 (tiles past them are read as zeros and not stored), K a multiple of 8. The host launches
// (ceil(N / 128), ceil(M / 128), z) blocks of 256 threads (idun.nn.cuda.GEMM).
// ---------------------------------------------------------------------------------------------------------------------

struct GemmParams {
  uint M, N, K;
  uint lda, ldb, ldc;
  uint members, div, ldr;  // the scaled operand: x[q / div] · r[q % members], r's row stride ldr
};

#define GM 128
#define GN 128
#define GK 8

// TA: A stored (K, M); TB: B stored (N, K); SA / SB: that operand is scaled (SA only without TA, SB only without TB).
// The output stage gets each thread's elements row by row: state = epi.begin_row(row), then
// epi.store(state, row, col, v0, v1) for columns col and col + 1.
template <typename AT, typename BT, bool TA, bool TB, bool SA, bool SB, typename Epi>
static __device__ __forceinline__ void gemm(const AT *A, const BT *B, const float *R, const GemmParams &p, uint bx,
                                            uint by, const Epi &epi) {
  __shared__ __align__(16) float As[2][GK][GM];
  __shared__ __align__(16) float Bs[2][GK][GN];
  const uint tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const uint m0 = by * GM, n0 = bx * GN;
  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
  }
  // Each thread reads four values of each operand a tile: (TA) row k = tid / 32, columns 4 · (tid % 32) ..; (not TA)
  // row tid / 2 of the tile, four values of k from 4 · (tid % 2).
  const uint wide_k = tid / 32, wide = 4 * (tid % 32), long_row = tid / 2, long_k = 4 * (tid % 2);
  float ra[4], rb[4];
  auto load = [&](uint k0) {
    if (TA) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint m = m0 + wide + v;
        ra[v] = m < p.M ? to_float(A[(k0 + wide_k) * p.lda + m]) : 0.0f;
      }
    } else {
      const uint q = m0 + long_row;
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint k = k0 + long_k + v;
        ra[v] = q >= p.M ? 0.0f
                : SA     ? to_float(A[(q / p.div) * p.lda + k]) * R[(q % p.members) * p.ldr + k]
                         : to_float(A[q * p.lda + k]);
      }
    }
    if (TB) {
      const uint n = n0 + long_row;
#pragma unroll
      for (int v = 0; v < 4; ++v) rb[v] = n < p.N ? to_float(B[n * p.ldb + k0 + long_k + v]) : 0.0f;
    } else {
      const uint q = k0 + wide_k;
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint n = n0 + wide + v;
        rb[v] = n >= p.N ? 0.0f
                : SB     ? to_float(B[(q / p.div) * p.ldb + n]) * R[(q % p.members) * p.ldr + n]
                         : to_float(B[q * p.ldb + n]);
      }
    }
  };
  auto keep = [&](int buf) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      if (TA) {
        As[buf][wide_k][wide + v] = ra[v];
      } else {
        As[buf][long_k + v][long_row] = ra[v];
      }
      if (TB) {
        Bs[buf][long_k + v][long_row] = rb[v];
      } else {
        Bs[buf][wide_k][wide + v] = rb[v];
      }
    }
  };
  load(0);
  keep(0);
  __syncthreads();
  int buf = 0;
  for (uint k0 = 0; k0 < p.K; k0 += GK) {
    const bool more = k0 + GK < p.K;
    if (more) load(k0 + GK);  // the next tile's values, read while this one is multiplied
#pragma unroll
    for (int kk = 0; kk < GK; ++kk) {
      const float4 a0 = *reinterpret_cast<const float4 *>(&As[buf][kk][4 * ty]);
      const float4 a1 = *reinterpret_cast<const float4 *>(&As[buf][kk][64 + 4 * ty]);
      const float4 b0 = *reinterpret_cast<const float4 *>(&Bs[buf][kk][4 * tx]);
      const float4 b1 = *reinterpret_cast<const float4 *>(&Bs[buf][kk][64 + 4 * tx]);
      const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      }
    }
    if (more) keep(buf ^ 1);  // the other buffer: every thread finished reading it before the last barrier
    __syncthreads();
    buf ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint row = m0 + (i < 4 ? 4 * ty + i : 64 + 4 * ty + i - 4);
    if (row >= p.M) continue;
    const uint state = epi.begin_row(row);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint col = n0 + 64 * half + 4 * tx;
      if (col >= p.N) continue;
      epi.store(state, row, col, acc[i][4 * half], acc[i][4 * half + 1]);
      epi.store(state, row, col + 2, acc[i][4 * half + 2], acc[i][4 * half + 3]);
    }
  }
}

// Output stage: store the product (as T).
template <typename T>
struct Plain {
  T *C;
  uint ldc;
  __device__ uint begin_row(uint) const { return 0; }
  __device__ void store(uint, uint row, uint col, float v0, float v1) const {
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
extern "C" __global__ void adamw(float *param, const float *grad, float *m1, float *m2, AdamParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) return;
  const float g = grad[i] * p.grad_scale;
  const float m = p.beta1 * m1[i] + (1.0f - p.beta1) * g;
  const float v = p.beta2 * m2[i] + (1.0f - p.beta2) * g * g;
  m1[i] = m;
  m2[i] = v;
  param[i] = param[i] * p.decay - p.step_size * m / (sqrtf(v) / p.root_c2 + p.eps);
}
