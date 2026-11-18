// idun_realmlp's kernels (docs/architecture/networks.md §8; the math is idun.realmlp.reference). Compiled
// after common.metal with ACT_T = float, and with RMLP_FREQS (Q) and RMLP_EMBED (E) defined by the host: the embedding
// kernels unroll over them and keep their sums in registers (on an Apple M1, with fast::cos, the embedding forward went
// from 2.13 to 0.29 ms; a runtime-sized loop kept the arrays in memory). fast::cos and fast::sincos: within 3e-7 of
// float64 on the init's arguments (|2π c w1 + b1| of a few tens at most), deterministic run to run.
//
// Rows b < rows_pad (padded to the GEMM tile); rows b >= batch are padding, their logit gradients are 0, so they add
// nothing to any gradient. The embedded inputs z have ldz columns: E values per input (feature-major), then the F
// clipped inputs, then zero padding. Hidden layers are H wide (a multiple of 64).
//
// K targets: the output layer has O = K · C outputs, target k's classes at k · C .. k · C + C - 1 of each row; a
// row's labels, losses and predictions are K values. The output layer's kernels run one thread per (row, output) or
// (unit, output), the loss and the prediction one per (row, target), so nothing is held in a per-thread array of the
// outputs and any K works; each sum keeps the order of the single-target kernels (K = 1 gives their bits).
//
// Own last block (own = 1): the last hidden layer has K · H units, target k's at k · H; it is one layer of K · H units
// to the GEMMs, its dropout mask is the H-unit mask of that block repeated for each target, and target k's logits read
// only its own H units (the output layer is block-diagonal, stored as H × O).

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

constant constexpr float TWO_PI = 6.28318530717958647692f;
constant constexpr float SELU_ALPHA = 1.6732632423543772848f;
constant constexpr float SELU_SCALE = 1.0507009873554804934f;

static float selu(float v) { return SELU_SCALE * (v > 0.0f ? v : SELU_ALPHA * (exp(v) - 1.0f)); }

static float selu_grad(float v) { return SELU_SCALE * (v > 0.0f ? 1.0f : SELU_ALPHA * exp(v)); }

// Mish, v · tanh(softplus(v)), with tanh(log(1 + e^v)) = n / (n + 2) for n = e^v (e^v + 2): no overflow below 20, and
// v itself above (tanh(softplus(20)) is 1 in float32).
static float mish(float v) {
  if (v > 20.0f) return v;
  const float e = exp(v), n = e * (e + 2.0f);
  return v * (n / (n + 2.0f));
}

static float mish_grad(float v) {
  if (v > 20.0f) return 1.0f;
  const float e = exp(v), n = e * (e + 2.0f), t = n / (n + 2.0f);
  return t + v * (1.0f - t * t) * (e / (1.0f + e));
}

// The hidden layers' activation f and its derivative (RmlpParams::activation); SELU's arithmetic is its own.
static float activate(float v, uint kind) {
  if (kind == 1u) return mish(v);
  if (kind == 2u) return max(v, 0.0f);
  return selu(v);
}

static float activate_grad(float v, uint kind) {
  if (kind == 1u) return mish_grad(v);
  if (kind == 2u) return v > 0.0f ? 1.0f : 0.0f;
  return selu_grad(v);
}

// idun.rng.element_bits_one: lowbias32(lowbias32(key ^ row) ^ unit) >> 8, kept when >= threshold.
static bool kept(constant RmlpParams &p, uint hash, uint u) {
  return u < p.hidden && (p.threshold == 0 || (lowbias32(hash ^ u) >> 8) >= p.threshold);
}

static uint row_hash(constant RmlpParams &p, device const uint *rows, uint b) {
  return p.threshold == 0 ? 0u : lowbias32(p.key ^ rows[b]);
}

constant constexpr uint QF = RMLP_FREQS, EM = RMLP_EMBED;

// z[b] = [cos(2π c w1 + b1) · w2 + b2 for each input (E values), the clipped inputs c].
kernel void rmlp_embed(device const float *c [[buffer(0)]], device const float *w1 [[buffer(1)]],
                       device const float *b1 [[buffer(2)]], device const float *w2 [[buffer(3)]],
                       device const float *b2 [[buffer(4)]], device float *z [[buffer(5)]],
                       constant RmlpParams &p [[buffer(6)]], uint2 id [[thread_position_in_grid]]) {
  const uint f = id.x, b = id.y;
  if (f >= p.inputs || b >= p.rows_pad) return;
  const float cv = c[b * p.inputs + f];
  float acc[EM];
  UNROLL for (uint o = 0; o < EM; ++o) acc[o] = b2[f * EM + o];
  device const float *wq = w1 + f * QF, *bq = b1 + f * QF, *w2f = w2 + f * QF * EM;
  UNROLL for (uint q = 0; q < QF; ++q) {
    const float cq = fast::cos(TWO_PI * cv * wq[q] + bq[q]);
    UNROLL for (uint o = 0; o < EM; ++o) acc[o] += cq * w2f[q * EM + o];
  }
  UNROLL for (uint o = 0; o < EM; ++o) z[b * p.ldz + f * EM + o] = acc[o];
  z[b * p.ldz + EM * p.inputs + f] = cv;
}

// Output stage of a hidden layer: pre = acc · gain + bias, a = pre + (f(pre) - pre) · act, h = a ⊙ mask (unit u's
// mask is that of u mod H: an own last block repeats the block's mask for every target).
struct LayerForward {
  device float *pre, *out;
  device const float *bias, *act;
  device const uint *rows;
  constant RmlpParams *p;
  uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  void store(uint hash, uint row, uint col, float v0, float v1) const {
    UNROLL for (short c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->units + u;
      const float v = (c == 0 ? v0 : v1) * p->gain + bias[u];
      pre[at] = v;
      const float a = v + (activate(v, p->activation) - v) * act[u];
      out[at] = kept(*p, hash, u % p->hidden) ? a * p->scale : 0.0f;
    }
  }
};

// Layer 0: h = (z ⊙ scale) · W0 · gain + ..., the front scale applied to the tiles as they load (R = scale).
kernel void rmlp_forward_scaled(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                                device float *pre [[buffer(2)]], device const float *R [[buffer(3)]],
                                device const float *bias [[buffer(4)]], device const float *act [[buffer(5)]],
                                device const uint *rows [[buffer(6)]], device float *out [[buffer(7)]],
                                constant RmlpFused &fp [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[BM * (BK + PAD)];
  threadgroup float Bs[BK * (BN + PAD)];
  const LayerForward epi{pre, out, bias, act, rows, &fp.n};
  gemm<float, float, false, false, true, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

// Layers 1 ..: h = h_in · W · gain + ....
kernel void rmlp_forward(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                         device float *pre [[buffer(2)]], device const float *R [[buffer(3)]],
                         device const float *bias [[buffer(4)]], device const float *act [[buffer(5)]],
                         device const uint *rows [[buffer(6)]], device float *out [[buffer(7)]],
                         constant RmlpFused &fp [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[BM * (BK + PAD)];
  threadgroup float Bs[BK * (BN + PAD)];
  const LayerForward epi{pre, out, bias, act, rows, &fp.n};
  gemm<float, float, false, false, false, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

// The H units output o reads: the last hidden layer's, or with an own last block those of o's target.
static uint head_units(constant RmlpParams &p, uint o) { return p.own != 0 ? (o / p.classes) * p.hidden : 0u; }

// logits[b, o] = Σ_u h[b, u] · W[u, o] · gain + bias[o]: one thread per (row b, output o), i = b · O + o.
kernel void rmlp_out_forward(device const float *h [[buffer(0)]], device const float *w [[buffer(1)]],
                             device const float *bias [[buffer(2)]], device float *logits [[buffer(3)]],
                             constant RmlpParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  const uint outputs = p.targets * p.classes;
  if (i >= p.rows_pad * outputs) return;
  const uint b = i / outputs, o = i % outputs;
  device const float *hb = h + b * p.units + head_units(p, o);
  float acc = 0.0f;
  for (uint u = 0; u < p.hidden; ++u) acc += hb[u] * w[u * outputs + o];
  logits[i] = acc * p.gain + bias[o];
}

// Cross-entropy against labels smoothed to (1 - s) one-hot + s / C, and its logit gradient (p - target) / (batch · K):
// one thread per (row b, target k), i = b · K + k, whose C logits start at i · C.
kernel void rmlp_loss(device const float *logits [[buffer(0)]], device const uint *labels [[buffer(1)]],
                      device float *d_logits [[buffer(2)]], device float *loss [[buffer(3)]],
                      device const float *weights [[buffer(4)]], constant RmlpParams &p [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p.rows_pad * p.targets) return;
  device const float *l = logits + i * p.classes;
  device float *d = d_logits + i * p.classes;
  if (i / p.targets >= p.batch) {
    for (uint c = 0; c < p.classes; ++c) d[c] = 0.0f;
    loss[i] = 0.0f;
    return;
  }
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = max(top, l[c]);
  float total = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += exp(l[c] - top);
  const float log_total = log(total);
  const float floor_p = p.smoothing / float(p.classes);
  const float s = p.inv_batch * weights[i / p.targets];  // inv_batch itself at weight 1
  float value = 0.0f;
  for (uint c = 0; c < p.classes; ++c) {
    const float target = floor_p + (c == labels[i] ? 1.0f - p.smoothing : 0.0f);
    const float log_p = (l[c] - top) - log_total;
    value -= log_p * target;
    d[c] = (exp(log_p) - target) * s;
  }
  loss[i] = value;
}

// Squared error (objective "mse": one output per target), loss[i] = (o − y)², and its gradient 2 (o − y) · inv_batch
// (inv_batch = 1 / (batch · K)): one thread per (row b, target k), i = b · K + k; padding rows give 0.
kernel void rmlp_mse(device const float *logits [[buffer(0)]], device const float *targets [[buffer(1)]],
                     device float *d_logits [[buffer(2)]], device float *loss [[buffer(3)]],
                     device const float *weights [[buffer(4)]], constant RmlpParams &p [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
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
kernel void rmlp_out_backward(device const float *h [[buffer(0)]], device const float *d_logits [[buffer(1)]],
                              device float *dw [[buffer(2)]], device float *d_bias [[buffer(3)]],
                              constant RmlpParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  const uint outputs = p.targets * p.classes;
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
// dpre = dy ⊙ (1 + (f'(pre) - 1) · act). Threadgroups of OB_U units × OB_B rows; W and d_logits pass through
// threadgroup memory OB_O outputs at a time, so their reads are coalesced for any K (a unit's row of W is O floats);
// each thread still sums over o in order.
constant constexpr uint OB_U = 32, OB_B = 8, OB_O = 32;

kernel void rmlp_out_block_backward(device const float *d_logits [[buffer(0)]], device const float *w [[buffer(1)]],
                                    device const float *pre [[buffer(2)]], device const float *act [[buffer(3)]],
                                    device const uint *rows [[buffer(4)]], device float *dy [[buffer(5)]],
                                    device float *dpre [[buffer(6)]], constant RmlpParams &p [[buffer(7)]],
                                    uint2 id [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]],
                                    uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup float ws[OB_U * (OB_O + 1)];  // W[u0 + r, o0 + j] at r · (OB_O + 1) + j
  threadgroup float ds[OB_B * OB_O];        // d_logits[b0 + r, o0 + j] at r · OB_O + j
  const uint u = id.x, b = id.y, outputs = p.targets * p.classes;
  const uint flat = tid.y * OB_U + tid.x, u0 = tg.x * OB_U, b0 = tg.y * OB_B;
  float dh = 0.0f;
  for (uint o0 = 0; o0 < outputs; o0 += OB_O) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = flat; e < OB_U * OB_O; e += OB_U * OB_B) {
      const uint r = e / OB_O, o = o0 + e % OB_O;
      ws[r * (OB_O + 1) + e % OB_O] = u0 + r < p.hidden && o < outputs ? w[(u0 + r) * outputs + o] : 0.0f;
    }
    const uint r = flat / OB_O, o = o0 + flat % OB_O;
    ds[flat] = b0 + r < p.rows_pad && o < outputs ? d_logits[(b0 + r) * outputs + o] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = min(OB_O, outputs - o0);
    for (uint j = 0; j < n; ++j) dh += ds[tid.y * OB_O + j] * ws[tid.x * (OB_O + 1) + j];
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
// (unit U < K · H, row b); the sum runs over c in order, as rmlp_out_block_backward's over o (K = 1 gives its bits).
kernel void rmlp_out_block_backward_own(device const float *d_logits [[buffer(0)]], device const float *w [[buffer(1)]],
                                        device const float *pre [[buffer(2)]], device const float *act [[buffer(3)]],
                                        device const uint *rows [[buffer(4)]], device float *dy [[buffer(5)]],
                                        device float *dpre [[buffer(6)]], constant RmlpParams &p [[buffer(7)]],
                                        uint2 id [[thread_position_in_grid]]) {
  const uint unit = id.x, b = id.y, outputs = p.targets * p.classes;
  if (unit >= p.units || b >= p.rows_pad) return;
  const uint k = unit / p.hidden, u = unit % p.hidden;
  device const float *dl = d_logits + b * outputs + k * p.classes;
  device const float *wu = w + u * outputs + k * p.classes;
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
  device const float *pre, *act;
  device const uint *rows;
  device float *dy, *dpre;
  constant RmlpParams *p;
  uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  void store(uint hash, uint row, uint col, float v0, float v1) const {
    UNROLL for (short c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->units + u;
      const float dh = (c == 0 ? v0 : v1) * p->gain;
      const float g = kept(*p, hash, u % p->hidden) ? dh * p->scale : 0.0f;
      dy[at] = g;
      dpre[at] = g * (1.0f + (activate_grad(pre[at], p->activation) - 1.0f) * act[u]);
    }
  }
};

kernel void rmlp_input_backward(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                                device const float *pre [[buffer(2)]], device const float *R [[buffer(3)]],
                                device const float *act [[buffer(4)]], device const uint *rows [[buffer(5)]],
                                device float *dy [[buffer(6)]], device float *dpre [[buffer(7)]],
                                constant RmlpFused &fp [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[BM * (BK + PAD)];
  threadgroup float Bs[BN * (BK + PAD)];
  const LayerBackward epi{pre, act, rows, dy, dpre, &fp.n};
  gemm<float, float, false, true, false, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

// Output stage scaling the product by gain (and per row by R, when given).
struct Scaled {
  device float *C;
  uint ldc;
  float gain;
  device const float *rowscale;  // nullptr: none
  uint begin_row(uint) const { return 0; }
  void store(uint, uint row, uint col, float v0, float v1) const {
    const float f = rowscale == nullptr ? gain : gain * rowscale[row];
    C[row * ldc + col] = v0 * f;
    C[row * ldc + col + 1] = v1 * f;
  }
};

// Layer 0's input gradient dz_s = dpre_0 · W0ᵀ · gain (the gradient with respect to the scaled embedding).
kernel void rmlp_embed_input_grad(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                                  device float *C [[buffer(2)]], device const float *R [[buffer(3)]],
                                  constant RmlpFused &fp [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[BM * (BK + PAD)];
  threadgroup float Bs[BN * (BK + PAD)];
  const Scaled epi{C, fp.g.ldc, fp.n.gain, nullptr};
  gemm<float, float, false, true, false, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

// dW = gain · inᵀ · dpre, summed over the rows; layer 0's input is z ⊙ scale, so its rows are scaled by R = scale
// (the GemmParams' members field set to 1 selects it).
kernel void rmlp_weight_grad(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                             device float *C [[buffer(2)]], device const float *R [[buffer(3)]],
                             constant RmlpFused &fp [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[BK * (BM + PAD)];
  threadgroup float Bs[BK * (BN + PAD)];
  const Scaled epi{C, fp.g.ldc, fp.n.gain, fp.g.members == 1 ? R : nullptr};
  gemm<float, float, true, false, false, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

// d_bias[u] = Σ_b dpre[b, u]; d_act[u] = Σ_b dy[b, u] · (f(pre) - pre), for the layer's units u.
kernel void rmlp_block_sums(device const float *dy [[buffer(0)]], device const float *dpre [[buffer(1)]],
                            device const float *pre [[buffer(2)]], device float *d_bias [[buffer(3)]],
                            device float *d_act [[buffer(4)]], constant RmlpParams &p [[buffer(5)]],
                            uint u [[thread_position_in_grid]]) {
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
kernel void rmlp_scale_grad(device const float *dzs [[buffer(0)]], device const float *z [[buffer(1)]],
                            device float *d_scale [[buffer(2)]], constant RmlpParams &p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= p.ldz) return;
  float acc = 0.0f;
  for (uint b = 0; b < p.batch; ++b) acc += dzs[b * p.ldz + i] * z[b * p.ldz + i];
  d_scale[i] = acc;
}

// The embedding's gradients for input f and frequency q, from de[b, o] = dz_s[b, f·E + o] · scale[f·E + o]:
// dw2[f, q, o] = Σ_b cos(arg) de, darg = -sin(arg) Σ_o de · w2[f, q, o], dw1 = Σ_b darg · 2π c, db1 = Σ_b darg; the
// thread q == 0 also computes db2[f, o] = Σ_b de.
kernel void rmlp_embed_backward(device const float *c [[buffer(0)]], device const float *w1 [[buffer(1)]],
                                device const float *b1 [[buffer(2)]], device const float *w2 [[buffer(3)]],
                                device const float *dzs [[buffer(4)]], device const float *scale [[buffer(5)]],
                                device float *dw1 [[buffer(6)]], device float *db1 [[buffer(7)]],
                                device float *dw2 [[buffer(8)]], device float *db2 [[buffer(9)]],
                                constant RmlpParams &p [[buffer(10)]], uint2 id [[thread_position_in_grid]]) {
  const uint q = id.x, f = id.y;
  if (q >= p.freqs || f >= p.inputs) return;
  const uint at = f * QF + q;
  float acc_w2[EM], acc_b2[EM], w2q[EM], sc[EM];
  UNROLL for (uint o = 0; o < EM; ++o) {
    acc_w2[o] = 0.0f;
    acc_b2[o] = 0.0f;
    w2q[o] = w2[at * EM + o];
    sc[o] = scale[f * EM + o];
  }
  const float w1q = w1[at], b1q = b1[at];
  float acc_w1 = 0.0f, acc_b1 = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    const float cv = c[b * p.inputs + f];
    float cs;
    const float sn = fast::sincos(TWO_PI * cv * w1q + b1q, cs);
    float dcos = 0.0f;
    UNROLL for (uint o = 0; o < EM; ++o) {
      const float de = dzs[b * p.ldz + f * EM + o] * sc[o];
      acc_w2[o] += cs * de;
      acc_b2[o] += de;
      dcos += de * w2q[o];
    }
    const float darg = -sn * dcos;
    acc_w1 += darg * TWO_PI * cv;
    acc_b1 += darg;
  }
  UNROLL for (uint o = 0; o < EM; ++o) dw2[at * EM + o] = acc_w2[o];
  dw1[at] = acc_w1;
  db1[at] = acc_b1;
  if (q == 0) {
    UNROLL for (uint o = 0; o < EM; ++o) db2[f * EM + o] = acc_b2[o];
  }
}

// The expected class Σ_c c · softmax_c of each real row and target: one thread per (row b, target k), i = b · K + k.
kernel void rmlp_predict(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]],
                         constant RmlpParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.batch * p.targets) return;
  device const float *l = logits + i * p.classes;
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = max(top, l[c]);
  float total = 0.0f, expected = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += exp(l[c] - top);
  for (uint c = 0; c < p.classes; ++c) expected += float(c) * (exp(l[c] - top) / total);
  out[i] = expected;
}

// The output itself (objective "mse"): one thread per (row b, target k), i = b · K + k.
kernel void rmlp_predict_value(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]],
                               constant RmlpParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.batch * p.targets) return;
  out[i] = logits[i];
}

struct GatherParams {
  uint batch;  // real rows; rows batch .. rows_pad are padding (zero inputs)
  uint rows_pad;
  uint n_named;  // the cache's columns: named factors, obfuscated codes, then the country one-hot
  uint n_obf;
  uint n_countries;
  uint named_ld;      // the row stride of the wrapped named values, in elements
  uint codes_ld;      // the row stride of the row-major codes (idun.nn.inputs.codes_row_major), in bytes
  uint targets;       // K
  uint target_ld;     // the row stride of the wrapped (n_rows, K) classes or targets, in elements
  float inv_clip_sq;  // 1 / clip²
};

// The preprocessed inputs of the batch rows, from the row-major codes and the wrapped cache (idun.nn.inputs with raw
// codes, then idun.realmlp.reference.preprocess): named factor (NaN → 0), code 0..4 or country one-hot;
// v = (x - median) · factor; c = v / sqrt(1 + v² / clip²). One thread per (input, row).
kernel void rmlp_gather(device const uchar *codes [[buffer(0)]], device const float *named [[buffer(1)]],
                        device const uchar *country [[buffer(2)]], device const uint *rows [[buffer(3)]],
                        device const float *median [[buffer(4)]], device const float *factor [[buffer(5)]],
                        device float *c [[buffer(6)]], constant GatherParams &p [[buffer(7)]],
                        uint2 id [[thread_position_in_grid]]) {
  const uint f = id.x, b = id.y;
  const uint inputs = p.n_named + p.n_obf + p.n_countries;
  if (f >= inputs || b >= p.rows_pad) return;
  if (b >= p.batch) {
    c[b * inputs + f] = 0.0f;
    return;
  }
  const uint row = rows[b];
  float x;
  if (f < p.n_named) {
    x = named[ulong(f) * p.named_ld + row];
    x = isnan(x) ? 0.0f : x;
  } else if (f < p.n_named + p.n_obf) {
    const uint j = f - p.n_named;
    const uchar packed = codes[ulong(row) * p.codes_ld + j / 2];
    x = float((j & 1u) ? (packed >> 4) : (packed & 0x0Fu));
  } else {
    x = country[row] == f - p.n_named - p.n_obf ? 1.0f : 0.0f;
  }
  const float v = (x - median[f]) * factor[f];
  c[b * inputs + f] = v / sqrt(1.0f + v * v * p.inv_clip_sq);
}

// The class of each batch row and target read from a compact (n_rows, K) uint8 class array (the training rows' classes
// are checked on the host): one thread per (row b, target k), i = b · K + k, reading byte row · target_ld + k.
kernel void rmlp_class_labels(device const uchar *classes [[buffer(0)]], device const uint *rows [[buffer(1)]],
                              device uint *labels [[buffer(2)]], constant GatherParams &p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= p.rows_pad * p.targets) return;
  const uint b = i / p.targets, k = i % p.targets;
  labels[i] = b < p.batch ? uint(classes[ulong(rows[b]) * p.target_ld + k]) : 0u;
}

// The regression target of each batch row and target read from a wrapped (n_rows, K) float32 target array (the
// training rows' targets are checked finite on the host): as rmlp_class_labels, the value copied as is.
kernel void rmlp_target_values(device const float *targets [[buffer(0)]], device const uint *rows [[buffer(1)]],
                               device float *labels [[buffer(2)]], constant GatherParams &p [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.rows_pad * p.targets) return;
  const uint b = i / p.targets, k = i % p.targets;
  labels[i] = b < p.batch ? targets[ulong(rows[b]) * p.target_ld + k] : 0.0f;
}

// Categorical embeddings (train_realmlp's categories; idun.realmlp.reference): G categorical inputs, each row's level
// codes (rows_pad, G) uint32; input g's levels have the table rows starts[g] .. (D values each). Their embedded values
// follow the clipped inputs in z, at columns offset .. offset + G · D, under the front scale as every input.
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
kernel void rmlp_cat_embed(device const uint *codes [[buffer(0)]], device const uint *starts [[buffer(1)]],
                           device const float *table [[buffer(2)]], device float *z [[buffer(3)]],
                           constant CatParams &p [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
  const uint i = id.x, b = id.y;
  if (i >= p.features * p.dim || b >= p.rows_pad) return;
  const uint g = i / p.dim, j = i % p.dim;
  z[b * p.ldz + p.offset + i] = table[(starts[g] + codes[b * p.features + g]) * p.dim + j];
}

// The table's gradient d_table[r, j] = Σ dzs[b, col] · scale[col] over the rows b = 0, 1, ... in order whose level of
// input g is r's (col = offset + g · D + j, g the input whose rows hold r); stored. One thread per (j, r).
kernel void rmlp_cat_backward(device const uint *codes [[buffer(0)]], device const uint *starts [[buffer(1)]],
                              device const float *dzs [[buffer(2)]], device const float *scale [[buffer(3)]],
                              device float *d_table [[buffer(4)]], constant CatParams &p [[buffer(5)]],
                              uint2 id [[thread_position_in_grid]]) {
  const uint j = id.x, r = id.y;
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
