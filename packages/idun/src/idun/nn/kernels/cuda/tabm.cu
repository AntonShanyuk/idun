// idun_tabm's kernels on CUDA (the math is idun.tabm.reference; kernels/metal/tabm.metal holds the Metal kernels of the
// same names, buffers and launch shapes). Compiled after common.cu.
//
// Rows of the member-expanded matrices are m = b · k + j (batch row b, member j); M = rows_pad · k. Widths are padded:
// ldw = width rounded up to 64, ldd = inputs rounded up to 64; padded rows, columns and parameters are zero and stay
// zero. The element-wise work of a block runs in the output stages of its GEMMs. A block's weight and adapter gradients
// come from one product per member, G_j = dT_jᵀ · h_j over that member's rows (gemm_tn_members). Micro-batches: the
// kernels see one micro-batch's rows, the logit gradient keeps the whole batch's scale, and every parameter gradient is
// stored by the first micro-batch and added to by each later one (`accumulate`). Every output is one sequential sum in
// the order the Metal kernel sums it.
//
// Activations and their gradients are stored as act_t: float, or bfloat16 bits when the host compiles with ACT_BF16
// (TabMParams.storage "bf16"), rounded to nearest with ties to even as the host's to_bf16; weights, optimizer moments,
// logits and losses stay float.

#ifdef ACT_BF16
typedef ushort act_t;
static __device__ __forceinline__ float act_load(act_t v) { return to_float(v); }
static __device__ __forceinline__ act_t act_store(float v) {
  uint u = __float_as_uint(v);
  return act_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
#else
typedef float act_t;
static __device__ __forceinline__ float act_load(act_t v) { return v; }
static __device__ __forceinline__ act_t act_store(float v) { return v; }
#endif

struct NetParams {
  uint batch;       // real rows b < batch of this (micro-)batch; rows batch .. rows_pad are padding
  uint rows_pad;    // rows of the batch buffers
  uint members;     // k
  uint width;       // w (unpadded)
  uint ldw;         // padded width
  uint classes;     // c
  uint key;         // dropout key of this step and block (idun.rng.derive_key)
  uint threshold;   // keep when the 24-bit draw >= threshold; 0 keeps everything
  float scale;      // 1 / (1 - p) as float32, or 1 without dropout
  float inv_count;  // 1 / (B · k), B the whole batch's rows (the sum of its micro-batches')
  uint din;         // member_weight_grad, member_adapter_grad: the block's input width (padded)
  uint chunk;       // the two-stage reductions: rows b per chunk
  uint chunks;      // and the number of chunks (partials to sum)
  uint accumulate;  // 0: the gradient reductions store their sums; 1: they add them (micro-batches after the first)
  uint targets;     // K: the head has targets · classes outputs per member
  float low;        // HL-Gauss: the first bin's lower edge
  float bin_width;  // and the bins' width (predict_centre)
};

// idun.rng.element_bits_one is lowbias32(lowbias32(key ^ row) ^ element) >> 8; the row's half is hashed once.
static __device__ __forceinline__ uint row_hash(const NetParams &p, const uint *rows, uint m) {
  return p.threshold == 0 ? 0u : lowbias32(p.key ^ rows[m / p.members]);
}

static __device__ __forceinline__ bool kept(const NetParams &p, uint hash, uint j, uint u) {
  return u < p.width && (p.threshold == 0 || (lowbias32(hash ^ (j * p.width + u)) >> 8) >= p.threshold);
}

#define MAX_CLASSES 8

struct FusedParams {  // the GEMMs with an element-wise output stage take both
  GemmParams g;
  NetParams n;
};

// Output stage of a block's forward: keep T (the backward pass reads it) and write relu(T ⊙ s + b) ⊙ mask.
struct Forward {
  act_t *T, *out;
  const float *s, *bias;
  const uint *rows;
  const NetParams *p;
  __device__ uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  __device__ void store(uint hash, uint row, uint col, float v0, float v1) const {
    const uint j = row % p->members;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->ldw + u;
      const float t = c == 0 ? v0 : v1;
      const float z = t * s[j * p->ldw + u] + bias[j * p->ldw + u];
      T[at] = act_store(t);
      out[at] = act_store(kept(*p, hash, j, u) ? fmaxf(z, 0.0f) * p->scale : 0.0f);
    }
  }
};

// Output stage of block i's input gradient du = dT · W, carried through block i - 1's dropout and ReLU:
// da = du ⊙ r_i, dz = da ⊙ mask ⊙ (z > 0) with z = T ⊙ s + b of block i - 1, dt = dz ⊙ s (du itself is not kept).
struct Backward {
  const float *r;
  const act_t *T;
  const float *s, *bias;
  const uint *rows;
  act_t *dz, *dt;
  const NetParams *p;
  __device__ uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  __device__ void store(uint hash, uint row, uint col, float v0, float v1) const {
    const uint j = row % p->members;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->ldw + u;
      const float v = c == 0 ? v0 : v1;
      const float da = v * r[j * p->ldw + u];
      const float sv = s[j * p->ldw + u];
      const float z = act_load(T[at]) * sv + bias[j * p->ldw + u];
      const float g = (kept(*p, hash, j, u) && z > 0.0f) ? da * p->scale : 0.0f;
      dz[at] = act_store(g);
      dt[at] = act_store(g * sv);
    }
  }
};

// Forward of a block: T = (h ⊙ r) · Wᵀ with h scaled on load (div = k for block 0's batch inputs, 1 afterwards); the
// output stage writes the block's activation.
extern "C" __global__ void gemm_forward(const act_t *A, const float *B, act_t *T, const float *R, const float *s,
                                        const float *bias, const uint *rows, act_t *out, FusedParams fp) {
  const Forward epi{T, out, s, bias, rows, &fp.n};
  gemm<act_t, float, false, true, true, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

// Input gradient of block i > 0 (dU = dT · W; R = r_i) and block i - 1's backward through dropout and ReLU.
extern "C" __global__ void gemm_backward(const act_t *A, const float *B, const float *R, const act_t *T, const float *s,
                                         const float *bias, const uint *rows, act_t *dz, act_t *dt, FusedParams fp) {
  const Backward epi{R, T, s, bias, rows, dz, dt, &fp.n};
  gemm<act_t, float, false, false, false, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

struct MemberGemmParams {
  GemmParams g;  // M = ldw, N = din, K = rows_pad; lda, ldb: the row strides of member j's rows
  uint a_step;   // member j's dT starts at A + j · a_step (its rows b · k + j)
  uint b_step;   // its h at B + j · b_step (0: block 0's batch inputs, every member's)
  uint c_step;   // its product at C + j · c_step
};

// The members' products G_j = dT_jᵀ · h_j (block z = member j), stored as they are.
extern "C" __global__ void gemm_tn_members(const act_t *A, const act_t *B, float *C, MemberGemmParams mp) {
  const Plain<float> epi{C + blockIdx.z * mp.c_step, mp.g.ldc};
  gemm<act_t, act_t, true, false, false, false>(A + blockIdx.z * mp.a_step, B + blockIdx.z * mp.b_step, nullptr, mp.g,
                                                blockIdx.x, blockIdx.y, epi);
}

// dW[u, c] = Σ_j r[j, c] · G_j[u, c], over the members j = 0, 1, ... in order; stored, or added with `accumulate`.
extern "C" __global__ void member_weight_grad(const float *G, const float *r, float *dW, NetParams p) {
  const uint c = blockIdx.x * blockDim.x + threadIdx.x, u = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= p.din || u >= p.ldw) return;
  float acc = 0.0f;
  for (uint j = 0; j < p.members; ++j) acc += r[j * p.din + c] * G[(j * p.ldw + u) * p.din + c];
  float *out = dW + u * p.din + c;
  *out = p.accumulate ? *out + acc : acc;
}

// d_r[j, c] = Σ_u W[u, c] · G_j[u, c], over the units u = 0, 1, ... in order; stored, or added with `accumulate`.
extern "C" __global__ void member_adapter_grad(const float *G, const float *W, float *dr, NetParams p) {
  const uint c = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= p.din || j >= p.members) return;
  float acc = 0.0f;
  for (uint u = 0; u < p.ldw; ++u) acc += W[u * p.din + c] * G[(j * p.ldw + u) * p.din + c];
  float *out = dr + j * p.din + c;
  *out = p.accumulate ? *out + acc : acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// The head, the loss and the reductions.
// ---------------------------------------------------------------------------------------------------------------------

// logits[m, t·c + i] = Σ_u a[m, u] · head[j, t·c + i, u] + head_bias[j, t·c + i], summed over u in order (four units
// at a time, each added in turn, as the Metal kernel); one thread per (batch row, target, member).
extern "C" __global__ void head_forward(const act_t *a, const float *head, const float *head_bias, float *logits,
                                        NetParams p) {
  const uint b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y * blockDim.y + threadIdx.y,
             j = blockIdx.z * blockDim.z + threadIdx.z;
  if (b >= p.rows_pad || t >= p.targets || j >= p.members) return;
  const uint m = b * p.members + j, outs = p.targets * p.classes, o = t * p.classes;
  const act_t *row = a + m * p.ldw;
  const float *h = head + (j * outs + o) * p.ldw;
  float acc[MAX_CLASSES] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint u = 0;
  for (; u + 4 <= p.width; u += 4) {
    const float v0 = act_load(row[u]), v1 = act_load(row[u + 1]), v2 = act_load(row[u + 2]), v3 = act_load(row[u + 3]);
#pragma unroll
    for (uint c = 0; c < MAX_CLASSES; ++c) {
      if (c < p.classes) {
        const float *w = h + c * p.ldw + u;
        acc[c] += v0 * w[0];
        acc[c] += v1 * w[1];
        acc[c] += v2 * w[2];
        acc[c] += v3 * w[3];
      }
    }
  }
  for (; u < p.width; ++u) {
    const float v = act_load(row[u]);
#pragma unroll
    for (uint c = 0; c < MAX_CLASSES; ++c) {
      if (c < p.classes) acc[c] += v * h[c * p.ldw + u];
    }
  }
#pragma unroll
  for (uint c = 0; c < MAX_CLASSES; ++c) {
    if (c < p.classes) logits[m * outs + o + c] = acc[c] + head_bias[j * outs + o + c];
  }
}

// Cross-entropy of each member output and target, loss[m · K + t], and its logit gradient (softmax - one-hot) ·
// inv_count · w (inv_count = 1 / (Σw · k · K), the row's weight w: 1 / (B · k · K) for unit weights); one thread per
// (row, target); padding rows give 0.
extern "C" __global__ void softmax_ce(const float *logits, const uint *labels, float *d_logits, float *loss,
                                      const float *weights, NetParams p) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint m = id / p.targets, t = id % p.targets, at = m * p.targets * p.classes + t * p.classes;
  if (m >= p.rows_pad * p.members) return;
  const float *l = logits + at;
  float *d = d_logits + at;
  const uint b = m / p.members;
  if (b >= p.batch) {
    for (uint c = 0; c < p.classes; ++c) d[c] = 0.0f;
    loss[id] = 0.0f;
    return;
  }
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = fmaxf(top, l[c]);
  float total = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += expf(l[c] - top);
  const float log_total = logf(total);
  const uint y = labels[b * p.targets + t];
  const float s = p.inv_count * weights[b];  // inv_count itself at weight 1
  for (uint c = 0; c < p.classes; ++c) {
    const float log_p = (l[c] - top) - log_total;
    d[c] = (expf(log_p) - (c == y ? 1.0f : 0.0f)) * s;
    if (c == y) loss[id] = -log_p;
  }
}

// HL-Gauss's soft-label cross-entropy of each member output, loss[m] = −Σ_c p[b, c] · log softmax(l)_c, and its
// gradient (softmax − p) · inv_count; one thread per member row; padding rows give 0. The bins are summed in order.
extern "C" __global__ void hl_gauss_ce(const float *logits, const float *labels, float *d_logits, float *loss,
                                       const float *weights, NetParams p) {
  const uint m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= p.rows_pad * p.members) return;
  const float *l = logits + m * p.classes;
  float *d = d_logits + m * p.classes;
  const uint b = m / p.members;
  if (b >= p.batch) {
    for (uint c = 0; c < p.classes; ++c) d[c] = 0.0f;
    loss[m] = 0.0f;
    return;
  }
  const float *y = labels + b * p.classes;
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = fmaxf(top, l[c]);
  float total = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += expf(l[c] - top);
  const float log_total = logf(total);
  float sum = 0.0f;
  const float s = p.inv_count * weights[b];  // inv_count itself at weight 1
  for (uint c = 0; c < p.classes; ++c) {
    const float log_p = (l[c] - top) - log_total;
    d[c] = (expf(log_p) - y[c]) * s;
    sum -= y[c] * log_p;
  }
  loss[m] = sum;
}

// Squared error of each member output and target (objective "mse"), loss[m · K + t] = (o − y)², and its gradient
// 2 (o − y) · inv_count; one thread per (row, target); padding rows give 0.
extern "C" __global__ void mse_loss(const float *logits, const float *targets, float *d_logits, float *loss,
                                    const float *weights, NetParams p) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint m = id / p.targets, t = id % p.targets, at = m * p.targets + t;
  if (m >= p.rows_pad * p.members) return;
  const uint b = m / p.members;
  if (b >= p.batch) {
    d_logits[at] = 0.0f;
    loss[id] = 0.0f;
    return;
  }
  const float diff = logits[at] - targets[b * p.targets + t];
  d_logits[at] = 2.0f * diff * (p.inv_count * weights[b]);  // inv_count itself at weight 1
  loss[id] = diff * diff;
}

// d_head[j, o, u] = Σ_b a[b·k + j, u] · d_logits[b·k + j, o], summed over b in order. Thread (q, t, j) computes the
// classes of target t at units u = 4q .. 4q + 3; the thread q == ldw / 4 computes d_head_bias[j, o] (unit value 1).
// Stored, or added with `accumulate`.
extern "C" __global__ void head_backward(const act_t *a, const float *d_logits, float *d_head, float *d_head_bias,
                                         NetParams p) {
  const uint q = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y * blockDim.y + threadIdx.y,
             j = blockIdx.z * blockDim.z + threadIdx.z;
  const uint outs = p.targets * p.classes, o = t * p.classes;
  if (q > p.ldw / 4 || t >= p.targets || j >= p.members) return;
  const bool bias = q == p.ldw / 4;
  float acc[MAX_CLASSES][4];
#pragma unroll
  for (uint c = 0; c < MAX_CLASSES; ++c) {
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[c][e] = 0.0f;
  }
  for (uint b = 0; b < p.batch; ++b) {
    const uint m = b * p.members + j;
    float v[4];
    if (bias) {
      v[0] = 1.0f;
      v[1] = v[2] = v[3] = 0.0f;
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] = act_load(a[m * p.ldw + 4 * q + e]);
    }
    const float *g = d_logits + m * outs + o;
#pragma unroll
    for (uint c = 0; c < MAX_CLASSES; ++c) {
      if (c < p.classes) {
        const float gc = g[c];
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[c][e] += v[e] * gc;
      }
    }
  }
#pragma unroll
  for (uint c = 0; c < MAX_CLASSES; ++c) {
    if (c >= p.classes) continue;
    if (bias) {
      float *out = d_head_bias + j * outs + o + c;
      *out = p.accumulate ? *out + acc[c][0] : acc[c][0];
    } else {
      float *out = d_head + (j * outs + o + c) * p.ldw + 4 * q;
#pragma unroll
      for (int e = 0; e < 4; ++e) out[e] = p.accumulate ? out[e] + acc[c][e] : acc[c][e];
    }
  }
}

// The top block's backward: da = Σ_o d_logits[m, o] · head[j, o, u] over the K · c outputs in order, then
// dz = da ⊙ mask ⊙ (z > 0) with z = T ⊙ s + b recomputed, and dt = dz ⊙ s. Thread (u, y) computes HBB_ROWS batch rows
// b = g · HBB_ROWS + r of member j, y = j · groups + g.
#define HBB_ROWS 4

extern "C" __global__ void head_block_backward(const float *d_logits, const float *head, const act_t *T, const float *s,
                                               const float *bias, const uint *rows, act_t *dz, act_t *dt, NetParams p) {
  const uint groups = (p.rows_pad + HBB_ROWS - 1) / HBB_ROWS;
  const uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint u = x, j = y / groups, b0 = (y % groups) * HBB_ROWS;
  if (u >= p.ldw || j >= p.members) return;
  const uint outs = p.targets * p.classes;
  const float *h = head + j * outs * p.ldw + u;
  const float *g[HBB_ROWS];
#pragma unroll
  for (uint r = 0; r < HBB_ROWS; ++r) g[r] = d_logits + (min(b0 + r, p.rows_pad - 1) * p.members + j) * outs;
  float da[HBB_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint o = 0; o < outs; ++o) {
    const float w = h[o * p.ldw];
#pragma unroll
    for (uint r = 0; r < HBB_ROWS; ++r) da[r] += g[r][o] * w;
  }
  const float sv = s[j * p.ldw + u], bv = bias[j * p.ldw + u];
#pragma unroll
  for (uint r = 0; r < HBB_ROWS; ++r) {
    if (b0 + r >= p.rows_pad) break;
    const uint m = (b0 + r) * p.members + j, at = m * p.ldw + u;
    const float z = act_load(T[at]) * sv + bv;
    const float grad = (kept(p, row_hash(p, rows, m), j, u) && z > 0.0f) ? da[r] * p.scale : 0.0f;
    dz[at] = act_store(grad);
    dt[at] = act_store(grad * sv);
  }
}

// The batch sums run in two stages, each in a fixed order: chunk q of rows b in [q · chunk, (q + 1) · chunk) writes
// partial sums, then sum_partials adds the chunks in order q = 0, 1, ....

// First stage of d_bias[j, u] = Σ_b dz[b·k + j, u] and d_s[j, u] = Σ_b dz · T: partials (chunks, k, ldw) each.
extern "C" __global__ void member_sums(const act_t *dz, const act_t *T, float *part_bias, float *part_s, NetParams p) {
  const uint u = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y,
             q = blockIdx.z * blockDim.z + threadIdx.z;
  if (u >= p.ldw || j >= p.members || q >= p.chunks) return;
  float sum_b = 0.0f, sum_s = 0.0f;
  for (uint b = q * p.chunk; b < min((q + 1) * p.chunk, p.batch); ++b) {
    const uint at = (b * p.members + j) * p.ldw + u;
    const float g = act_load(dz[at]);
    sum_b += g;
    sum_s += g * act_load(T[at]);
  }
  part_bias[(q * p.members + j) * p.ldw + u] = sum_b;
  part_s[(q * p.members + j) * p.ldw + u] = sum_s;
}

// Second stage: out[i] = Σ_q part[q · n + i] for i < n = k · din, in chunk order; added to out[i] with `accumulate`.
extern "C" __global__ void sum_partials(const float *part, float *out, NetParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x, n = p.members * p.din;
  if (i >= n) return;
  float acc = 0.0f;
  for (uint q = 0; q < p.chunks; ++q) acc += part[q * n + i];
  out[i] = p.accumulate ? out[i] + acc : acc;
}

// Expected class of the member-averaged softmax, out[b · K + t], one thread per (batch row, target) (classes <= 8).
extern "C" __global__ void predict_class(const float *logits, float *out, NetParams p) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint b = id / p.targets, outs = p.targets * p.classes, o = (id % p.targets) * p.classes;
  if (b >= p.batch) return;
  float mean[MAX_CLASSES] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint j = 0; j < p.members; ++j) {
    const float *l = logits + (b * p.members + j) * outs + o;
    float top = l[0];
    for (uint c = 1; c < p.classes; ++c) top = fmaxf(top, l[c]);
    float total = 0.0f;
    for (uint c = 0; c < p.classes; ++c) total += expf(l[c] - top);
    for (uint c = 0; c < p.classes; ++c) mean[c] += expf(l[c] - top) / total;
  }
  float expected = 0.0f;
  for (uint c = 0; c < p.classes; ++c) expected += float(c) * (mean[c] / float(p.members));
  out[id] = expected;
}

// Mean of the members' outputs (objective "mse"), out[b · K + t], one thread per (batch row, target); in member order.
extern "C" __global__ void predict_mean(const float *logits, float *out, NetParams p) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint b = id / p.targets, t = id % p.targets;
  if (b >= p.batch) return;
  float sum = 0.0f;
  for (uint j = 0; j < p.members; ++j) sum += logits[(b * p.members + j) * p.targets + t];
  out[id] = sum / float(p.members);
}

// HL-Gauss's prediction, out[b]: the expected bin centre low + (c + ½) · bin_width under the members' mean softmax,
// one thread per batch row (n_bins <= MAX_BINS); members, then bins, summed in order.
#define MAX_BINS 512

extern "C" __global__ void predict_centre(const float *logits, float *out, NetParams p) {
  const uint b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= p.batch) return;
  float mean[MAX_BINS];
  for (uint c = 0; c < p.classes; ++c) mean[c] = 0.0f;
  for (uint j = 0; j < p.members; ++j) {
    const float *l = logits + (b * p.members + j) * p.classes;
    float top = l[0];
    for (uint c = 1; c < p.classes; ++c) top = fmaxf(top, l[c]);
    float total = 0.0f;
    for (uint c = 0; c < p.classes; ++c) total += expf(l[c] - top);
    for (uint c = 0; c < p.classes; ++c) mean[c] += expf(l[c] - top) / total;
  }
  float expected = 0.0f;
  for (uint c = 0; c < p.classes; ++c)
    expected += (p.low + (float(c) + 0.5f) * p.bin_width) * (mean[c] / float(p.members));
  out[b] = expected;
}

// ---------------------------------------------------------------------------------------------------------------------
// The temporal encoder (idun.tabm.encoder): per step a shared Linear(C → E) and GELU, then a GRU(E → H) over the steps,
// oldest first; its last state takes block 0's first H input columns. Always float. Pair i = b · steps + s is row b's
// step s. seq_forward and seq_backward run one thread per (hidden unit u, row), SEQ_ROWS rows per block, a row's state
// in shared memory; every sum runs over its index in order.
// ---------------------------------------------------------------------------------------------------------------------

struct SeqParams {
  uint batch;     // real rows b < batch
  uint rows_pad;  // rows of the batch buffers (a multiple of SEQ_ROWS)
  uint steps;     // S
  uint inputs;    // C: a step's inputs, the channels and the present flag
  uint ld_in;     // the sequence buffer's row stride (C padded to 64)
  uint embed;     // E
  uint ld_e;      // E padded to 64
  uint hidden;    // H
  uint ld_h;      // H padded to 64
  uint ld_g;      // 3H padded to 64: the gate gradients' row stride (gates r, z, n at rows 0, H, 2H)
  uint members;   // k
  uint ldd;       // the batch inputs' row stride
};

#define SEQ_ROWS 4   // rows per block (host: SEQ_ROWS)
#define MAX_ENC 128  // the most units of E and of H (params.MAX_ENCODER_WIDTH)

static __device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Φ(x) through erf as Abramowitz & Stegun 7.1.26 give it (|error| < 1.5e-7), as the Metal kernels compute it.
static __device__ __forceinline__ float normal_cdf(float x) {
  const float a = fabsf(x) * 0.70710678118654752f, t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float tail = 0.5f * poly * expf(-a * a);  // ½ erfc(a)
  return x >= 0.0f ? 1.0f - tail : tail;
}

static __device__ __forceinline__ float gelu(float x) { return x * normal_cdf(x); }

static __device__ __forceinline__ float gelu_grad(float x) {
  return normal_cdf(x) + x * expf(-0.5f * x * x) * 0.39894228040143268f;
}

// Forward. Thread (u, rr) of block g runs unit u of row b = g · SEQ_ROWS + rr through every step and stores, per pair
// i: the embedding's pre-activation and output (pre, emb: i · ld_e + c), the state before the step (hprev: i · ld_h +
// u) and r, z, n and W_hn h + b_hn (gates: (4 i + gate) · ld_h + u). The last state goes to the row's first H inputs
// (0 on a padding row).
extern "C" __global__ void seq_forward(const float *x, const float *we, const float *be, const float *wih,
                                       const float *whh, const float *bih, const float *bhh, float *pre, float *emb,
                                       float *hprev, float *gates, act_t *inputs, SeqParams p) {
  __shared__ float e_tg[SEQ_ROWS * MAX_ENC], h_tg[SEQ_ROWS * MAX_ENC];
  const uint u = threadIdx.x, rr = threadIdx.y, b = blockIdx.y * SEQ_ROWS + rr, H = p.hidden;
  float *e_row = e_tg + rr * MAX_ENC, *h_row = h_tg + rr * MAX_ENC;
  h_row[u] = 0.0f;
  for (uint s = 0; s < p.steps; ++s) {
    const uint i = b * p.steps + s;
    const float *xs = x + i * p.ld_in;
    for (uint c = u; c < p.embed; c += H) {
      float acc = 0.0f;
      for (uint k = 0; k < p.inputs; ++k) acc += we[c * p.ld_in + k] * xs[k];
      const float v = acc + be[c];
      pre[i * p.ld_e + c] = v;
      e_row[c] = emb[i * p.ld_e + c] = gelu(v);
    }
    __syncthreads();
    float gi[3], gh[3];
#pragma unroll
    for (uint g = 0; g < 3; ++g) {
      const uint row = g * H + u;
      float ai = 0.0f, ah = 0.0f;
      for (uint c = 0; c < p.embed; ++c) ai += wih[row * p.ld_e + c] * e_row[c];
      for (uint c = 0; c < H; ++c) ah += whh[row * p.ld_h + c] * h_row[c];
      gi[g] = ai + bih[row];
      gh[g] = ah + bhh[row];
    }
    const float r = sigmoid(gi[0] + gh[0]), z = sigmoid(gi[1] + gh[1]), n = tanhf(gi[2] + r * gh[2]);
    const float h = h_row[u];
    float *gt = gates + 4 * i * p.ld_h + u;
    hprev[i * p.ld_h + u] = h;
    gt[0] = r;
    gt[p.ld_h] = z;
    gt[2 * p.ld_h] = n;
    gt[3 * p.ld_h] = gh[2];
    __syncthreads();
    h_row[u] = (1.0f - z) * n + z * h;
  }
  inputs[b * p.ldd + u] = act_store(b < p.batch ? h_row[u] : 0.0f);
}

// The gradient at block 0's encoder inputs, dU = dT_0 · W_0[:, :ld_h] (W_0 stored (ldw, ldd)): one GEMM per batch.
extern "C" __global__ void seq_input_grad(const act_t *A, const float *B, float *C, GemmParams g) {
  const Plain<float> epi{C, g.ldc};
  gemm<act_t, float, false, false, false, false>(A, B, nullptr, g, blockIdx.x, blockIdx.y, epi);
}

// The encoder's output gradient dh[b, c] = Σ_j r_0[j, c] · dU[b · k + j, c] over the members j = 0, 1, ... in order;
// 0 on padding rows and columns. One thread per (c, b).
extern "C" __global__ void seq_member_sum(const float *dU, const float *r, float *dh, SeqParams p) {
  const uint c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= p.ld_h || b >= p.rows_pad) return;
  float acc = 0.0f;
  if (b < p.batch && c < p.hidden) {
    for (uint j = 0; j < p.members; ++j) acc += r[j * p.ldd + c] * dU[(b * p.members + j) * p.ld_h + c];
  }
  dh[b * p.ld_h + c] = acc;
}

// Backward through time, laid out as seq_forward: from the output gradient dh of a row's last state, every step's gate
// gradients d_gi (the input part: dr, dz, dn) and d_gh (the hidden part: dr, dz, dn · r) at i · ld_g + gate · H + u,
// and the embedding's pre-activation gradient d_pre (i · ld_e + c). Padding rows give 0.
extern "C" __global__ void seq_backward(const float *dh_in, const float *wih, const float *whh, const float *pre,
                                        const float *hprev, const float *gates, float *d_gi, float *d_gh, float *d_pre,
                                        SeqParams p) {
  __shared__ float gi_tg[SEQ_ROWS * 3 * MAX_ENC], gh_tg[SEQ_ROWS * 3 * MAX_ENC];
  const uint u = threadIdx.x, rr = threadIdx.y, b = blockIdx.y * SEQ_ROWS + rr, H = p.hidden;
  float *gi_row = gi_tg + rr * 3 * MAX_ENC, *gh_row = gh_tg + rr * 3 * MAX_ENC;
  float dh = b < p.batch ? dh_in[b * p.ld_h + u] : 0.0f;
  for (uint s = p.steps; s-- > 0;) {
    const uint i = b * p.steps + s;
    const float *gt = gates + 4 * i * p.ld_h + u;
    const float r = gt[0], z = gt[p.ld_h], n = gt[2 * p.ld_h], hn = gt[3 * p.ld_h];
    const float dn = dh * (1.0f - z) * (1.0f - n * n);
    const float dz = dh * (hprev[i * p.ld_h + u] - n) * z * (1.0f - z);
    const float dr = dn * hn * r * (1.0f - r);
    gi_row[u] = d_gi[i * p.ld_g + u] = dr;
    gi_row[H + u] = d_gi[i * p.ld_g + H + u] = dz;
    gi_row[2 * H + u] = d_gi[i * p.ld_g + 2 * H + u] = dn;
    gh_row[u] = d_gh[i * p.ld_g + u] = dr;
    gh_row[H + u] = d_gh[i * p.ld_g + H + u] = dz;
    gh_row[2 * H + u] = d_gh[i * p.ld_g + 2 * H + u] = dn * r;
    __syncthreads();
    float acc = 0.0f;
    for (uint g = 0; g < 3 * H; ++g) acc += whh[g * p.ld_h + u] * gh_row[g];
    for (uint c = u; c < p.embed; c += H) {
      float de = 0.0f;
      for (uint g = 0; g < 3 * H; ++g) de += wih[g * p.ld_e + c] * gi_row[g];
      d_pre[i * p.ld_e + c] = de * gelu_grad(pre[i * p.ld_e + c]);
    }
    __syncthreads();
    dh = dh * z + acc;
  }
}

// A weight gradient's partial over chunk q (block z) of pairs: C_q = A_qᵀ · B_q, A stored (pairs, M), B (pairs, N);
// sum_partials adds the chunks in order.
extern "C" __global__ void seq_weight_grad(const float *A, const float *B, float *C, MemberGemmParams mp) {
  const Plain<float> epi{C + blockIdx.z * mp.c_step, mp.g.ldc};
  gemm<float, float, true, false, false, false>(A + blockIdx.z * mp.a_step, B + blockIdx.z * mp.b_step, nullptr, mp.g,
                                                blockIdx.x, blockIdx.y, epi);
}

// A bias gradient's partial over chunk q of pairs: part[q · din + c] = Σ_i A[i · din + c] over the chunk's pairs in
// order (NetParams: din the columns, chunk the pairs per chunk, chunks). One thread per (c, q).
extern "C" __global__ void seq_column_partials(const float *A, float *part, NetParams p) {
  const uint c = blockIdx.x * blockDim.x + threadIdx.x, q = blockIdx.y * blockDim.y + threadIdx.y;
  if (c >= p.din || q >= p.chunks) return;
  float acc = 0.0f;
  for (uint i = q * p.chunk; i < (q + 1) * p.chunk; ++i) acc += A[i * p.din + c];
  part[q * p.din + c] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// Piecewise-linear numeric embeddings (TabMParams.num_emb_type "pwl"; idun.tabm.embeddings), as tabm.metal's: the first
// F inputs of a row (raw, (rows_pad, ld_raw)) become F · d embedded values at block 0's input columns 0 .. F · d; the
// other inputs follow them, written there by the host. Block 0's input gradient for the embedded columns comes from
// seq_input_grad and emb_member_sum; the weight gradients are sums over the batch rows in order, one thread per weight.

struct EmbParams {
  uint batch;       // real rows b < batch
  uint rows_pad;    // rows of the batch buffers
  uint features;    // F: the embedded inputs, the first F of a row's
  uint components;  // P: the encoding's components per feature
  uint dim;         // d: a feature's embedded values
  uint ld_raw;      // the embedded inputs' row stride
  uint ldd;         // block 0's input row stride
  uint ld_e;        // F · d padded to 64: the embedded values' gradient row stride
  uint members;     // k
  uint accumulate;  // emb_weight_grad: 0 stores the sums, 1 adds them (micro-batches after the first)
};

// Component c of the encoding of input v of feature f: v · scale + shift, the first component clamped above at 1, the
// middle ones to [0, 1] and the last below at 0 (unclamped for a feature of one bin); nothing clamped when P = 1.
static __device__ float ple_component(float v, const float *scale, const float *shift, const uint *single, uint f,
                                      uint c, uint P) {
  const float q = v * scale[f * P + c] + shift[f * P + c];
  if (P == 1) return q;
  if (c == 0) return fminf(q, 1.0f);
  if (c + 1 < P) return fminf(fmaxf(q, 0.0f), 1.0f);
  return single[f] != 0u ? q : fmaxf(q, 0.0f);
}

// x[b, i] = v · linear[i] + linear_bias[i] + Σ_c q_c(v) · ple[f, c, j] (c = 0, 1, ... in order) for i = f · d + j and
// v = raw[b, f]; 0 on padding rows. One thread per (i, b).
extern "C" __global__ void emb_forward(const float *raw, const float *scale, const float *shift, const uint *single,
                                       const float *linear, const float *linear_bias, const float *ple, act_t *x,
                                       EmbParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  const uint d = p.dim, P = p.components;
  if (i >= p.features * d || b >= p.rows_pad) return;
  const uint f = i / d, j = i % d;
  float acc = 0.0f;
  if (b < p.batch) {
    const float v = raw[b * p.ld_raw + f];
    acc = v * linear[i] + linear_bias[i];
    for (uint c = 0; c < P; ++c) acc += ple_component(v, scale, shift, single, f, c, P) * ple[(f * P + c) * d + j];
  }
  x[b * p.ldd + i] = act_store(acc);
}

// The embedded values' gradient d_emb[b, i] = Σ_j r_0[j, i] · dU[b · k + j, i] over the members j = 0, 1, ... in order;
// 0 on padding rows and columns. One thread per (i, b).
extern "C" __global__ void emb_member_sum(const float *dU, const float *r, float *d_emb, EmbParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= p.ld_e || b >= p.rows_pad) return;
  float acc = 0.0f;
  if (b < p.batch && i < p.features * p.dim) {
    for (uint j = 0; j < p.members; ++j) acc += r[j * p.ldd + i] * dU[(b * p.members + j) * p.ld_e + i];
  }
  d_emb[b * p.ld_e + i] = acc;
}

// The weights' gradients, each a sum over the batch rows b = 0, 1, ... in order: d_linear[i] = Σ v · g,
// d_linear_bias[i] = Σ g and d_ple[f, c, j] = Σ q_c(v) · g, for g = d_emb[b, i], i = f · d + j, v = raw[b, f]; stored,
// or added to when accumulate. One thread per weight: t = part · F · d + i, part 0 linear, 1 linear_bias, 2 + c ple's
// c.
extern "C" __global__ void emb_weight_grad(const float *raw, const float *scale, const float *shift, const uint *single,
                                           const float *d_emb, float *d_linear, float *d_linear_bias, float *d_ple,
                                           EmbParams p) {
  const uint t = blockIdx.x * blockDim.x + threadIdx.x;
  const uint width = p.features * p.dim, P = p.components;
  if (t >= (P + 2) * width) return;
  const uint part = t / width, i = t % width, f = i / p.dim, j = i % p.dim;
  float acc = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    const float g = d_emb[b * p.ld_e + i], v = raw[b * p.ld_raw + f];
    acc += (part == 0 ? v : part == 1 ? 1.0f : ple_component(v, scale, shift, single, f, part - 2, P)) * g;
  }
  float *out = part == 0 ? d_linear + i : part == 1 ? d_linear_bias + i : d_ple + (f * P + part - 2) * p.dim + j;
  *out = (p.accumulate != 0u ? *out : 0.0f) + acc;
}
