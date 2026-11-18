// idun_realmlp's kernels on CUDA (docs/architecture/networks.md §8; the math is idun.realmlp.reference;
// kernels/metal/rmlp.metal holds the Metal kernels of the same names, buffers and launch shapes). Compiled after
// common.cu (fp32 only), with RMLP_FREQS (Q) and RMLP_EMBED (E) defined by the host: the embedding kernels
// unroll over them and keep their sums in registers. The cosines are CUDA's cosf and sincosf (not the fast intrinsics).
//
// Rows b < rows_pad (padded to the GEMM tile); rows b >= batch are padding, their logit gradients are 0, so they add
// nothing to any gradient. The embedded inputs z have ldz columns: E values per input (feature-major), then the F
// clipped inputs, then zero padding. Hidden layers are H wide (a multiple of 64). K targets: the output layer has
// O = K · C outputs, target k's classes at k · C ..; own last block (own = 1): the last hidden layer has K · H units,
// target k's at k · H, and target k's logits read only its own H units. Every sum runs in the Metal kernel's order.

struct RmlpParams {
  uint batch;       // real rows
  uint rows_pad;    // rows of the batch buffers
  uint inputs;      // F
  uint freqs;       // Q
  uint embed;       // E
  uint ldz;         // padded (E + 1) · F
  uint hidden;      // H
  uint classes;     // C, per target
  uint targets;     // K
  uint units;       // this layer's units: H, or K · H for an own last block
  uint own;         // 1: the last hidden layer is each target's own (K · H units)
  uint key;         // dropout key of this step and block (idun.rng.derive_key)
  uint threshold;   // keep when the 24-bit draw >= threshold; 0 keeps everything
  float scale;      // 1 / (1 - p), or 1 without dropout
  float gain;       // the layer's NTK gain 1 / sqrt(fan_in)
  float smoothing;  // label smoothing
  float inv_batch;  // 1 / (batch · targets): the loss is the mean over the rows and the targets
  uint activation;  // the hidden layers' activation f: 0 SELU, 1 Mish, 2 ReLU (RealMLPParams.act)
};

struct RmlpFused {  // the GEMMs with an element-wise output stage take both
  GemmParams g;
  RmlpParams n;
};

#define TWO_PI 6.28318530717958647692f
#define SELU_ALPHA 1.6732632423543772848f
#define SELU_SCALE 1.0507009873554804934f

static __device__ __forceinline__ float selu(float v) {
  return SELU_SCALE * (v > 0.0f ? v : SELU_ALPHA * (expf(v) - 1.0f));
}

static __device__ __forceinline__ float selu_grad(float v) {
  return SELU_SCALE * (v > 0.0f ? 1.0f : SELU_ALPHA * expf(v));
}

// Mish, v · tanh(softplus(v)), with tanh(log(1 + e^v)) = n / (n + 2) for n = e^v (e^v + 2): no overflow below 20, and
// v itself above (tanh(softplus(20)) is 1 in float32).
static __device__ __forceinline__ float mish(float v) {
  if (v > 20.0f) return v;
  const float e = expf(v), n = e * (e + 2.0f);
  return v * (n / (n + 2.0f));
}

static __device__ __forceinline__ float mish_grad(float v) {
  if (v > 20.0f) return 1.0f;
  const float e = expf(v), n = e * (e + 2.0f), t = n / (n + 2.0f);
  return t + v * (1.0f - t * t) * (e / (1.0f + e));
}

// The hidden layers' activation f and its derivative (RmlpParams::activation); SELU's arithmetic is its own.
static __device__ __forceinline__ float activate(float v, uint kind) {
  if (kind == 1u) return mish(v);
  if (kind == 2u) return fmaxf(v, 0.0f);
  return selu(v);
}

static __device__ __forceinline__ float activate_grad(float v, uint kind) {
  if (kind == 1u) return mish_grad(v);
  if (kind == 2u) return v > 0.0f ? 1.0f : 0.0f;
  return selu_grad(v);
}

// idun.rng.element_bits_one: lowbias32(lowbias32(key ^ row) ^ unit) >> 8, kept when >= threshold.
static __device__ __forceinline__ bool kept(const RmlpParams &p, uint hash, uint u) {
  return u < p.hidden && (p.threshold == 0 || (lowbias32(hash ^ u) >> 8) >= p.threshold);
}

static __device__ __forceinline__ uint row_hash(const RmlpParams &p, const uint *rows, uint b) {
  return p.threshold == 0 ? 0u : lowbias32(p.key ^ rows[b]);
}

#define QF RMLP_FREQS
#define EM RMLP_EMBED

// z[b] = [cos(2π c w1 + b1) · w2 + b2 for each input (E values), the clipped inputs c].
extern "C" __global__ void rmlp_embed(const float *c, const float *w1, const float *b1, const float *w2,
                                      const float *b2, float *z, RmlpParams p) {
  const uint f = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  if (f >= p.inputs || b >= p.rows_pad) return;
  const float cv = c[b * p.inputs + f];
  float acc[EM];
#pragma unroll
  for (uint o = 0; o < EM; ++o) acc[o] = b2[f * EM + o];
  const float *wq = w1 + f * QF, *bq = b1 + f * QF, *w2f = w2 + f * QF * EM;
#pragma unroll
  for (uint q = 0; q < QF; ++q) {
    const float cq = cosf(TWO_PI * cv * wq[q] + bq[q]);
#pragma unroll
    for (uint o = 0; o < EM; ++o) acc[o] += cq * w2f[q * EM + o];
  }
#pragma unroll
  for (uint o = 0; o < EM; ++o) z[b * p.ldz + f * EM + o] = acc[o];
  z[b * p.ldz + EM * p.inputs + f] = cv;
}

// Output stage of a hidden layer: pre = acc · gain + bias, a = pre + (f(pre) - pre) · act, h = a ⊙ mask (unit u's
// mask is that of u mod H: an own last block repeats the block's mask for every target).
struct LayerForward {
  float *pre, *out;
  const float *bias, *act;
  const uint *rows;
  const RmlpParams *p;
  __device__ uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  __device__ void store(uint hash, uint row, uint col, float v0, float v1) const {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->units + u;
      const float v = (c == 0 ? v0 : v1) * p->gain + bias[u];
      pre[at] = v;
      const float a = v + (activate(v, p->activation) - v) * act[u];
      out[at] = kept(*p, hash, u % p->hidden) ? a * p->scale : 0.0f;
    }
  }
};

// Layer 0: h = (z ⊙ scale) · W0 · gain + ..., the front scale applied to the tiles as they load (R = scale).
extern "C" __global__ void rmlp_forward_scaled(const float *A, const float *B, float *pre, const float *R,
                                               const float *bias, const float *act, const uint *rows, float *out,
                                               RmlpFused fp) {
  const LayerForward epi{pre, out, bias, act, rows, &fp.n};
  gemm<float, float, false, false, true, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

// Layers 1 ..: h = h_in · W · gain + ....
extern "C" __global__ void rmlp_forward(const float *A, const float *B, float *pre, const float *R, const float *bias,
                                        const float *act, const uint *rows, float *out, RmlpFused fp) {
  const LayerForward epi{pre, out, bias, act, rows, &fp.n};
  gemm<float, float, false, false, false, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

// The H units output o reads: the last hidden layer's, or with an own last block those of o's target.
static __device__ __forceinline__ uint head_units(const RmlpParams &p, uint o) {
  return p.own != 0 ? (o / p.classes) * p.hidden : 0u;
}

// logits[b, o] = Σ_u h[b, u] · W[u, o] · gain + bias[o]: one thread per (row b, output o), i = b · O + o.
extern "C" __global__ void rmlp_out_forward(const float *h, const float *w, const float *bias, float *logits,
                                            RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x, outputs = p.targets * p.classes;
  if (i >= p.rows_pad * outputs) return;
  const uint b = i / outputs, o = i % outputs;
  const float *hb = h + b * p.units + head_units(p, o);
  float acc = 0.0f;
  for (uint u = 0; u < p.hidden; ++u) acc += hb[u] * w[u * outputs + o];
  logits[i] = acc * p.gain + bias[o];
}

// Cross-entropy against labels smoothed to (1 - s) one-hot + s / C, and its logit gradient (p - target) / (batch · K):
// one thread per (row b, target k), i = b · K + k, whose C logits start at i · C.
extern "C" __global__ void rmlp_loss(const float *logits, const uint *labels, float *d_logits, float *loss,
                                     const float *weights, RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.rows_pad * p.targets) return;
  const float *l = logits + i * p.classes;
  float *d = d_logits + i * p.classes;
  if (i / p.targets >= p.batch) {
    for (uint c = 0; c < p.classes; ++c) d[c] = 0.0f;
    loss[i] = 0.0f;
    return;
  }
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = fmaxf(top, l[c]);
  float total = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += expf(l[c] - top);
  const float log_total = logf(total);
  const float floor_p = p.smoothing / float(p.classes);
  const float s = p.inv_batch * weights[i / p.targets];  // inv_batch itself at weight 1
  float value = 0.0f;
  for (uint c = 0; c < p.classes; ++c) {
    const float target = floor_p + (c == labels[i] ? 1.0f - p.smoothing : 0.0f);
    const float log_p = (l[c] - top) - log_total;
    value -= log_p * target;
    d[c] = (expf(log_p) - target) * s;
  }
  loss[i] = value;
}

// Squared error (objective "mse": one output per target), loss[i] = (o − y)², and its gradient 2 (o − y) · inv_batch:
// one thread per (row b, target k), i = b · K + k; padding rows give 0.
extern "C" __global__ void rmlp_mse(const float *logits, const float *targets, float *d_logits, float *loss,
                                    const float *weights, RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.rows_pad * p.targets) return;
  if (i / p.targets >= p.batch) {
    d_logits[i] = 0.0f;
    loss[i] = 0.0f;
    return;
  }
  const float diff = logits[i] - targets[i];
  d_logits[i] = 2.0f * diff * (p.inv_batch * weights[i / p.targets]);  // inv_batch itself at weight 1
  loss[i] = diff * diff;
}

// dW[u, o] = gain · Σ_b h[b, u] · d_logits[b, o]; the threads u == hidden compute d_bias[o] = Σ_b d_logits[b, o]. One
// thread per (unit u <= H, output o), i = u · O + o.
extern "C" __global__ void rmlp_out_backward(const float *h, const float *d_logits, float *dw, float *d_bias,
                                             RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x, outputs = p.targets * p.classes;
  if (i >= (p.hidden + 1) * outputs) return;
  const uint u = i / outputs, o = i % outputs, first = head_units(p, o);
  float acc = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    acc += (u == p.hidden ? 1.0f : h[b * p.units + first + u]) * d_logits[b * outputs + o];
  }
  if (u == p.hidden) {
    d_bias[o] = acc;
  } else {
    dw[i] = acc * p.gain;
  }
}

// The last hidden layer's backward from the logits: dh = gain · Σ_o d_logits[b, o] · W[u, o], dy = dh ⊙ mask,
// dpre = dy ⊙ (1 + (f'(pre) - 1) · act). Blocks of OB_U units × OB_B rows; W and d_logits pass through shared
// memory OB_O outputs at a time; each thread still sums over o in order.
#define OB_U 32
#define OB_B 8
#define OB_O 32

extern "C" __global__ void rmlp_out_block_backward(const float *d_logits, const float *w, const float *pre,
                                                   const float *act, const uint *rows, float *dy, float *dpre,
                                                   RmlpParams p) {
  __shared__ float ws[OB_U * (OB_O + 1)];  // W[u0 + r, o0 + j] at r · (OB_O + 1) + j
  __shared__ float ds[OB_B * OB_O];        // d_logits[b0 + r, o0 + j] at r · OB_O + j
  const uint tx = threadIdx.x, ty = threadIdx.y, outputs = p.targets * p.classes;
  const uint u = blockIdx.x * OB_U + tx, b = blockIdx.y * OB_B + ty;
  const uint flat = ty * OB_U + tx, u0 = blockIdx.x * OB_U, b0 = blockIdx.y * OB_B;
  float dh = 0.0f;
  for (uint o0 = 0; o0 < outputs; o0 += OB_O) {
    __syncthreads();
    for (uint e = flat; e < OB_U * OB_O; e += OB_U * OB_B) {
      const uint r = e / OB_O, o = o0 + e % OB_O;
      ws[r * (OB_O + 1) + e % OB_O] = u0 + r < p.hidden && o < outputs ? w[(u0 + r) * outputs + o] : 0.0f;
    }
    const uint r = flat / OB_O, o = o0 + flat % OB_O;
    ds[flat] = b0 + r < p.rows_pad && o < outputs ? d_logits[(b0 + r) * outputs + o] : 0.0f;
    __syncthreads();
    const uint n = min((uint)OB_O, outputs - o0);
    for (uint j = 0; j < n; ++j) dh += ds[ty * OB_O + j] * ws[tx * (OB_O + 1) + j];
  }
  if (u >= p.hidden || b >= p.rows_pad) return;
  dh *= p.gain;
  const uint at = b * p.hidden + u;
  const float g = kept(p, row_hash(p, rows, b), u) ? dh * p.scale : 0.0f;
  dy[at] = g;
  dpre[at] = g * (1.0f + (activate_grad(pre[at], p.activation) - 1.0f) * act[u]);
}

// The own last block's backward from the logits: unit U = k · H + u reads only target k's C logits,
// dh = gain · Σ_c d_logits[b, k · C + c] · W[u, k · C + c], then the block's mask (unit u) and f. One thread per
// (unit U < K · H, row b); the sum runs over c in order.
extern "C" __global__ void rmlp_out_block_backward_own(const float *d_logits, const float *w, const float *pre,
                                                       const float *act, const uint *rows, float *dy, float *dpre,
                                                       RmlpParams p) {
  const uint unit = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  const uint outputs = p.targets * p.classes;
  if (unit >= p.units || b >= p.rows_pad) return;
  const uint k = unit / p.hidden, u = unit % p.hidden;
  const float *dl = d_logits + b * outputs + k * p.classes;
  const float *wu = w + u * outputs + k * p.classes;
  float dh = 0.0f;
  for (uint c = 0; c < p.classes; ++c) dh += dl[c] * wu[c];
  dh *= p.gain;
  const uint at = b * p.units + unit;
  const float g = kept(p, row_hash(p, rows, b), u) ? dh * p.scale : 0.0f;
  dy[at] = g;
  dpre[at] = g * (1.0f + (activate_grad(pre[at], p.activation) - 1.0f) * act[unit]);
}

// Output stage of layer l's input gradient (dpre_l · W_lᵀ · gain), carried through layer l - 1's dropout and f.
struct LayerBackward {
  const float *pre, *act;
  const uint *rows;
  float *dy, *dpre;
  const RmlpParams *p;
  __device__ uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  __device__ void store(uint hash, uint row, uint col, float v0, float v1) const {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->units + u;
      const float dh = (c == 0 ? v0 : v1) * p->gain;
      const float g = kept(*p, hash, u % p->hidden) ? dh * p->scale : 0.0f;
      dy[at] = g;
      dpre[at] = g * (1.0f + (activate_grad(pre[at], p->activation) - 1.0f) * act[u]);
    }
  }
};

extern "C" __global__ void rmlp_input_backward(const float *A, const float *B, const float *pre, const float *R,
                                               const float *act, const uint *rows, float *dy, float *dpre,
                                               RmlpFused fp) {
  const LayerBackward epi{pre, act, rows, dy, dpre, &fp.n};
  gemm<float, float, false, true, false, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

// Output stage scaling the product by gain (and per row by R, when given).
struct Scaled {
  float *C;
  uint ldc;
  float gain;
  const float *rowscale;  // nullptr: none
  __device__ uint begin_row(uint) const { return 0; }
  __device__ void store(uint, uint row, uint col, float v0, float v1) const {
    const float f = rowscale == nullptr ? gain : gain * rowscale[row];
    C[row * ldc + col] = v0 * f;
    C[row * ldc + col + 1] = v1 * f;
  }
};

// Layer 0's input gradient dz_s = dpre_0 · W0ᵀ · gain (the gradient with respect to the scaled embedding).
extern "C" __global__ void rmlp_embed_input_grad(const float *A, const float *B, float *C, const float *R,
                                                 RmlpFused fp) {
  const Scaled epi{C, fp.g.ldc, fp.n.gain, nullptr};
  gemm<float, float, false, true, false, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

// dW = gain · inᵀ · dpre, summed over the rows; layer 0's input is z ⊙ scale, so its rows are scaled by R = scale
// (the GemmParams' members field set to 1 selects it).
extern "C" __global__ void rmlp_weight_grad(const float *A, const float *B, float *C, const float *R, RmlpFused fp) {
  const Scaled epi{C, fp.g.ldc, fp.n.gain, fp.g.members == 1 ? R : nullptr};
  gemm<float, float, true, false, false, false>(A, B, R, fp.g, blockIdx.x, blockIdx.y, epi);
}

// d_bias[u] = Σ_b dpre[b, u]; d_act[u] = Σ_b dy[b, u] · (f(pre) - pre), for the layer's units u.
extern "C" __global__ void rmlp_block_sums(const float *dy, const float *dpre, const float *pre, float *d_bias,
                                           float *d_act, RmlpParams p) {
  const uint u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= p.units) return;
  float sum_b = 0.0f, sum_a = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    const uint at = b * p.units + u;
    sum_b += dpre[at];
    sum_a += dy[at] * (activate(pre[at], p.activation) - pre[at]);
  }
  d_bias[u] = sum_b;
  d_act[u] = sum_a;
}

// d_scale[i] = Σ_b dz_s[b, i] · z[b, i].
extern "C" __global__ void rmlp_scale_grad(const float *dzs, const float *z, float *d_scale, RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.ldz) return;
  float acc = 0.0f;
  for (uint b = 0; b < p.batch; ++b) acc += dzs[b * p.ldz + i] * z[b * p.ldz + i];
  d_scale[i] = acc;
}

// The embedding's gradients for input f and frequency q, from de[b, o] = dz_s[b, f·E + o] · scale[f·E + o]:
// dw2[f, q, o] = Σ_b cos(arg) de, darg = -sin(arg) Σ_o de · w2[f, q, o], dw1 = Σ_b darg · 2π c, db1 = Σ_b darg; the
// thread q == 0 also computes db2[f, o] = Σ_b de.
extern "C" __global__ void rmlp_embed_backward(const float *c, const float *w1, const float *b1, const float *w2,
                                               const float *dzs, const float *scale, float *dw1, float *db1, float *dw2,
                                               float *db2, RmlpParams p) {
  const uint q = blockIdx.x * blockDim.x + threadIdx.x, f = blockIdx.y * blockDim.y + threadIdx.y;
  if (q >= p.freqs || f >= p.inputs) return;
  const uint at = f * QF + q;
  float acc_w2[EM], acc_b2[EM], w2q[EM], sc[EM];
#pragma unroll
  for (uint o = 0; o < EM; ++o) {
    acc_w2[o] = 0.0f;
    acc_b2[o] = 0.0f;
    w2q[o] = w2[at * EM + o];
    sc[o] = scale[f * EM + o];
  }
  const float w1q = w1[at], b1q = b1[at];
  float acc_w1 = 0.0f, acc_b1 = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    const float cv = c[b * p.inputs + f];
    float sn, cs;
    sincosf(TWO_PI * cv * w1q + b1q, &sn, &cs);
    float dcos = 0.0f;
#pragma unroll
    for (uint o = 0; o < EM; ++o) {
      const float de = dzs[b * p.ldz + f * EM + o] * sc[o];
      acc_w2[o] += cs * de;
      acc_b2[o] += de;
      dcos += de * w2q[o];
    }
    const float darg = -sn * dcos;
    acc_w1 += darg * TWO_PI * cv;
    acc_b1 += darg;
  }
#pragma unroll
  for (uint o = 0; o < EM; ++o) dw2[at * EM + o] = acc_w2[o];
  dw1[at] = acc_w1;
  db1[at] = acc_b1;
  if (q == 0) {
#pragma unroll
    for (uint o = 0; o < EM; ++o) db2[f * EM + o] = acc_b2[o];
  }
}

// The expected class Σ_c c · softmax_c of each real row and target: one thread per (row b, target k), i = b · K + k.
extern "C" __global__ void rmlp_predict(const float *logits, float *out, RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.batch * p.targets) return;
  const float *l = logits + i * p.classes;
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = fmaxf(top, l[c]);
  float total = 0.0f, expected = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += expf(l[c] - top);
  for (uint c = 0; c < p.classes; ++c) expected += float(c) * (expf(l[c] - top) / total);
  out[i] = expected;
}

// The output itself (objective "mse"): one thread per (row b, target k), i = b · K + k.
extern "C" __global__ void rmlp_predict_value(const float *logits, float *out, RmlpParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.batch * p.targets) return;
  out[i] = logits[i];
}

struct GatherParams {
  uint batch;  // real rows; rows batch .. rows_pad are padding (zero inputs)
  uint rows_pad;
  uint n_named;  // the cache's columns: named factors, obfuscated codes, then the country one-hot
  uint n_obf;
  uint n_countries;
  uint named_ld;      // the row stride of the placed named values, in elements
  uint codes_ld;      // the row stride of the row-major codes (idun.nn.inputs.codes_row_major), in bytes
  uint targets;       // K
  uint target_ld;     // the row stride of the placed (n_rows, K) classes or targets, in elements
  float inv_clip_sq;  // 1 / clip²
};

// The preprocessed inputs of the batch rows, from the row-major codes and the placed cache: named factor (NaN → 0),
// code 0..4 or country one-hot; v = (x - median) · factor; c = v / sqrt(1 + v² / clip²). One thread per (input, row).
extern "C" __global__ void rmlp_gather(const unsigned char *codes, const float *named, const unsigned char *country,
                                       const uint *rows, const float *median, const float *factor, float *c,
                                       GatherParams p) {
  const uint f = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  const uint inputs = p.n_named + p.n_obf + p.n_countries;
  if (f >= inputs || b >= p.rows_pad) return;
  if (b >= p.batch) {
    c[b * inputs + f] = 0.0f;
    return;
  }
  const uint row = rows[b];
  float x;
  if (f < p.n_named) {
    x = named[(unsigned long long)f * p.named_ld + row];
    x = isnan(x) ? 0.0f : x;
  } else if (f < p.n_named + p.n_obf) {
    const uint j = f - p.n_named;
    const unsigned char packed = codes[(unsigned long long)row * p.codes_ld + j / 2];
    x = float((j & 1u) ? (packed >> 4) : (packed & 0x0Fu));
  } else {
    x = country[row] == f - p.n_named - p.n_obf ? 1.0f : 0.0f;
  }
  const float v = (x - median[f]) * factor[f];
  c[b * inputs + f] = v / sqrtf(1.0f + v * v * p.inv_clip_sq);
}

// The class of each batch row and target read from a compact (n_rows, K) uint8 class array: one thread per (row b,
// target k), i = b · K + k, reading byte row · target_ld + k.
extern "C" __global__ void rmlp_class_labels(const unsigned char *classes, const uint *rows, uint *labels,
                                             GatherParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.rows_pad * p.targets) return;
  const uint b = i / p.targets, k = i % p.targets;
  labels[i] = b < p.batch ? uint(classes[(unsigned long long)rows[b] * p.target_ld + k]) : 0u;
}

// The regression target of each batch row and target read from a placed (n_rows, K) float32 target array: as
// rmlp_class_labels, the value copied as is.
extern "C" __global__ void rmlp_target_values(const float *targets, const uint *rows, float *labels, GatherParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.rows_pad * p.targets) return;
  const uint b = i / p.targets, k = i % p.targets;
  labels[i] = b < p.batch ? targets[(unsigned long long)rows[b] * p.target_ld + k] : 0.0f;
}

// Categorical embeddings (train_realmlp's categories; idun.realmlp.reference), as rmlp.metal's: G categorical inputs,
// each row's level codes (rows_pad, G) uint32; input g's levels have the table rows starts[g] .. (D values each). Their
// embedded values follow the clipped inputs in z, at columns offset .. offset + G · D, under the front scale.
struct CatParams {
  uint batch;     // real rows
  uint rows_pad;  // rows of the batch buffers
  uint features;  // G
  uint dim;       // D: values per level (RealMLPParams.embedding_size)
  uint ldz;       // z's row stride
  uint offset;    // the embedded values' first column of z: (E + 1) · F
  uint rows;      // the table's rows: the levels of every categorical input
};

// z[b, offset + g · D + j] = table[starts[g] + code[b, g], j]. One thread per (g · D + j, b).
extern "C" __global__ void rmlp_cat_embed(const uint *codes, const uint *starts, const float *table, float *z,
                                          CatParams p) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= p.features * p.dim || b >= p.rows_pad) return;
  const uint g = i / p.dim, j = i % p.dim;
  z[b * p.ldz + p.offset + i] = table[(starts[g] + codes[b * p.features + g]) * p.dim + j];
}

// The table's gradient d_table[r, j] = Σ dzs[b, col] · scale[col] over the rows b = 0, 1, ... in order whose level of
// input g is r's (col = offset + g · D + j, g the input whose rows hold r); stored. One thread per (j, r).
extern "C" __global__ void rmlp_cat_backward(const uint *codes, const uint *starts, const float *dzs,
                                             const float *scale, float *d_table, CatParams p) {
  const uint j = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= p.dim || r >= p.rows) return;
  uint g = p.features - 1;
  while (starts[g] > r) --g;
  const uint level = r - starts[g], col = p.offset + g * p.dim + j;
  float acc = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    if (codes[b * p.features + g] == level) acc += dzs[b * p.ldz + col] * scale[col];
  }
  d_table[r * p.dim + j] = acc;
}
