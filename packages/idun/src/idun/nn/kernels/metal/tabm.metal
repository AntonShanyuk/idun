// idun_tabm's kernels (the math is idun.tabm.reference). Compiled after common.metal.
//
// Rows of the member-expanded matrices are m = b · k + j (batch row b, member j); M = rows_pad · k. Widths are padded:
// ldw = width rounded up to 64, ldd = inputs rounded up to 64; padded rows, columns and parameters are zero and stay
// zero. The element-wise work of a block runs in the output stages of its GEMMs (one pass over the activations fewer
// each). A block's weight and adapter gradients come from one product per member, G_j = dT_jᵀ · h_j over that
// member's rows (gemm_tn_members): dW = Σ_j G_j ⊙ r_j and d_r_j = Σ_u W ⊙ G_j, so block 0 needs no input gradient
// at all. A batch may run as several micro-batches (TabMParams.micro_batch): the kernels then see one micro-batch's
// rows, the logit gradient keeps the whole batch's scale, and every parameter gradient is stored by the first
// micro-batch and added to by each later one, in micro-batch order (`accumulate`).
//
// HL-Gauss (one target, n_bins outputs): the head kernels run the bins as n_bins / 8 groups of 8 (targets = groups,
// classes = 8 in their NetParams), and hl_gauss_ce and predict_centre see one target of n_bins classes. Its labels are
// soft, (rows_pad, n_bins) float32 from the host (idun.tabm.hl_gauss).
//
// Multi-target heads: every member's head has outs = K · classes outputs, target t's classes at t · classes ..;
// logits and their gradients are (M, outs), labels (rows_pad, K), losses (M, K), predictions (rows_pad, K). The head
// and its gradient are stored transposed, (k, outs, ldw): row (j, o) is output o's weights over the units of member j.
// Threads that keep per-class arrays (head_forward, softmax_ce, head_backward, predict_class) handle one target's
// classes; every output is still one sequential sum in the single-target kernels' order (over u in head_forward, over
// the batch rows in head_backward, over the outputs o = 0, 1, ... in head_block_backward), so K = 1 gives their bits.

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
static uint row_hash(constant NetParams &p, device const uint *rows, uint m) {
  return p.threshold == 0 ? 0u : lowbias32(p.key ^ rows[m / p.members]);
}

static bool kept(constant NetParams &p, uint hash, uint j, uint u) {
  return u < p.width && (p.threshold == 0 || (lowbias32(hash ^ (j * p.width + u)) >> 8) >= p.threshold);
}

// Per-class register arrays have MAX_CLASSES entries; loops over them are unrolled to MAX_CLASSES and predicated on
// c < classes, so the arrays stay in registers (a loop to the runtime p.classes would index them dynamically).
constant constexpr uint MAX_CLASSES = 8;

struct FusedParams {  // the GEMMs with an element-wise output stage take both
  GemmParams g;
  NetParams n;
};

// Output stage of a block's forward: keep T (the backward pass reads it) and write relu(T ⊙ s + b) ⊙ mask.
struct Forward {
  device act_t *T, *out;
  device const float *s, *bias;
  device const uint *rows;
  constant NetParams *p;
  uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  void store(uint hash, uint row, uint col, float v0, float v1) const {
    const uint j = row % p->members;
    UNROLL for (short c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->ldw + u;
      const float t = c == 0 ? v0 : v1;
      const float z = t * s[j * p->ldw + u] + bias[j * p->ldw + u];
      T[at] = act_t(t);
      out[at] = act_t(kept(*p, hash, j, u) ? max(z, 0.0f) * p->scale : 0.0f);
    }
  }
};

// Output stage of block i's input gradient du = dT · W, carried through block i - 1's dropout and ReLU:
// da = du ⊙ r_i, dz = da ⊙ mask ⊙ (z > 0) with z = T ⊙ s + b of block i - 1, dt = dz ⊙ s (du itself is not kept).
struct Backward {
  device const float *r;
  device const act_t *T;
  device const float *s, *bias;
  device const uint *rows;
  device act_t *dz, *dt;
  constant NetParams *p;
  uint begin_row(uint row) const { return row_hash(*p, rows, row); }
  void store(uint hash, uint row, uint col, float v0, float v1) const {
    const uint j = row % p->members;
    UNROLL for (short c = 0; c < 2; ++c) {
      const uint u = col + c, at = row * p->ldw + u;
      const float v = c == 0 ? v0 : v1;
      const float da = v * r[j * p->ldw + u];
      const float sv = s[j * p->ldw + u];
      const float z = float(T[at]) * sv + bias[j * p->ldw + u];
      const float g = (kept(*p, hash, j, u) && z > 0.0f) ? da * p->scale : 0.0f;
      dz[at] = act_t(g);
      dt[at] = act_t(g * sv);
    }
  }
};

// Forward of a block: T = (h ⊙ r) · Wᵀ with h scaled on load (div = k for block 0's batch inputs, 1 afterwards); the
// output stage writes the block's activation.
kernel void gemm_forward(device const act_t *A [[buffer(0)]], device const float *B [[buffer(1)]],
                         device act_t *T [[buffer(2)]], device const float *R [[buffer(3)]],
                         device const float *s [[buffer(4)]], device const float *bias [[buffer(5)]],
                         device const uint *rows [[buffer(6)]], device act_t *out [[buffer(7)]],
                         constant FusedParams &fp [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup act_t As[BM * (BK + PAD)];
  threadgroup float Bs[BN * (BK + PAD)];
  const Forward epi{T, out, s, bias, rows, &fp.n};
  gemm<act_t, float, false, true, true, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

// Input gradient of block i > 0 (dU = dT · W; R = r_i) and block i - 1's backward through dropout and ReLU.
kernel void gemm_backward(device const act_t *A [[buffer(0)]], device const float *B [[buffer(1)]],
                          device const float *R [[buffer(2)]], device const act_t *T [[buffer(3)]],
                          device const float *s [[buffer(4)]], device const float *bias [[buffer(5)]],
                          device const uint *rows [[buffer(6)]], device act_t *dz [[buffer(7)]],
                          device act_t *dt [[buffer(8)]], constant FusedParams &fp [[buffer(9)]],
                          uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup act_t As[BM * (BK + PAD)];
  threadgroup float Bs[BK * (BN + PAD)];
  const Backward epi{R, T, s, bias, rows, dz, dt, &fp.n};
  gemm<act_t, float, false, false, false, false>(A, B, R, fp.g, tg, sg, lane, As, Bs, epi);
}

struct MemberGemmParams {
  GemmParams g;  // M = ldw, N = din, K = rows_pad; lda, ldb: the row strides of member j's rows
  uint a_step;   // member j's dT starts at A + j · a_step (its rows b · k + j)
  uint b_step;   // its h at B + j · b_step (0: block 0's batch inputs, every member's)
  uint c_step;   // its product at C + j · c_step
};

// The members' products G_j = dT_jᵀ · h_j (threadgroup z = member j): dT_j and h_j are member j's rows, so the sum
// runs over the batch rows b alone. Stored as they are; member_weight_grad and member_adapter_grad read them.
kernel void gemm_tn_members(device const act_t *A [[buffer(0)]], device const act_t *B [[buffer(1)]],
                            device float *C [[buffer(2)]], constant MemberGemmParams &mp [[buffer(3)]],
                            uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup act_t As[BK * (BM + PAD)];
  threadgroup act_t Bs[BK * (BN + PAD)];
  const Plain<float> epi{C + tg.z * mp.c_step, mp.g.ldc};
  gemm<act_t, act_t, true, false, false, false>(A + tg.z * mp.a_step, B + tg.z * mp.b_step, nullptr, mp.g, tg.xy, sg,
                                                lane, As, Bs, epi);
}

// dW[u, c] = Σ_j r[j, c] · G_j[u, c], over the members j = 0, 1, ... in order; stored, or added with `accumulate`.
kernel void member_weight_grad(device const float *G [[buffer(0)]], device const float *r [[buffer(1)]],
                               device float *dW [[buffer(2)]], constant NetParams &p [[buffer(3)]],
                               uint2 id [[thread_position_in_grid]]) {
  const uint c = id.x, u = id.y;
  if (c >= p.din || u >= p.ldw) return;
  float acc = 0.0f;
  for (uint j = 0; j < p.members; ++j) acc += r[j * p.din + c] * G[(j * p.ldw + u) * p.din + c];
  device float *out = dW + u * p.din + c;
  *out = p.accumulate ? *out + acc : acc;
}

// d_r[j, c] = Σ_u W[u, c] · G_j[u, c] (= Σ_b du ⊙ h with du = dT · W), over the units u = 0, 1, ... in order; stored,
// or added with `accumulate`.
kernel void member_adapter_grad(device const float *G [[buffer(0)]], device const float *W [[buffer(1)]],
                                device float *dr [[buffer(2)]], constant NetParams &p [[buffer(3)]],
                                uint2 id [[thread_position_in_grid]]) {
  const uint c = id.x, j = id.y;
  if (c >= p.din || j >= p.members) return;
  float acc = 0.0f;
  for (uint u = 0; u < p.ldw; ++u) acc += W[u * p.din + c] * G[(j * p.ldw + u) * p.din + c];
  device float *out = dr + j * p.din + c;
  *out = p.accumulate ? *out + acc : acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// The head, the loss and the reductions.
// ---------------------------------------------------------------------------------------------------------------------

// logits[m, t·c + i] = Σ_u a[m, u] · head[j, t·c + i, u] + head_bias[j, t·c + i], summed over u in order; one thread
// per (batch row, target) (classes <= 8), thread (b, t, j). A SIMD-group is 32 batch rows of one member and target, so
// its head reads are the same address in every lane; a threadgroup holds up to TARGETS_PER_GROUP (host) targets of the
// same rows, which read them at about the same time.
kernel void head_forward(device const act_t *a [[buffer(0)]], device const float *head [[buffer(1)]],
                         device const float *head_bias [[buffer(2)]], device float *logits [[buffer(3)]],
                         constant NetParams &p [[buffer(4)]], uint3 id [[thread_position_in_grid]]) {
  const uint b = id.x, t = id.y, j = id.z;
  if (b >= p.rows_pad || t >= p.targets || j >= p.members) return;
  const uint m = b * p.members + j, outs = p.targets * p.classes, o = t * p.classes;
  device const act_t *row = a + m * p.ldw;
  device const float *h = head + (j * outs + o) * p.ldw;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint u = 0;
  for (; u + 4 <= p.width; u += 4) {
    const float4 v = float4(*(device const vec<act_t, 4> *)(row + u));
    UNROLL for (uint c = 0; c < MAX_CLASSES; ++c) {
      if (c < p.classes) {
        const float4 w = *(device const float4 *)(h + c * p.ldw + u);
        acc[c] += v.x * w.x;
        acc[c] += v.y * w.y;
        acc[c] += v.z * w.z;
        acc[c] += v.w * w.w;
      }
    }
  }
  for (; u < p.width; ++u) {
    const float v = float(row[u]);
    UNROLL for (uint c = 0; c < MAX_CLASSES; ++c) if (c < p.classes) acc[c] += v * h[c * p.ldw + u];
  }
  UNROLL for (uint c = 0; c < MAX_CLASSES; ++c) {
    if (c < p.classes) logits[m * outs + o + c] = acc[c] + head_bias[j * outs + o + c];
  }
}

// Cross-entropy of each member output and target, loss[m · K + t], and its logit gradient (softmax - one-hot) ·
// inv_count · w (inv_count = 1 / (Σw · k · K), the row's weight w: 1 / (B · k · K) for unit weights); one thread per
// (row, target); padding rows give 0.
kernel void softmax_ce(device const float *logits [[buffer(0)]], device const uint *labels [[buffer(1)]],
                       device float *d_logits [[buffer(2)]], device float *loss [[buffer(3)]],
                       device const float *weights [[buffer(4)]], constant NetParams &p [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
  const uint m = id / p.targets, t = id % p.targets, at = m * p.targets * p.classes + t * p.classes;
  if (m >= p.rows_pad * p.members) return;
  device const float *l = logits + at;
  device float *d = d_logits + at;
  const uint b = m / p.members;
  if (b >= p.batch) {
    for (uint c = 0; c < p.classes; ++c) d[c] = 0.0f;
    loss[id] = 0.0f;
    return;
  }
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = max(top, l[c]);
  float total = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += exp(l[c] - top);
  const float log_total = log(total);
  const uint y = labels[b * p.targets + t];
  const float s = p.inv_count * weights[b];  // inv_count itself at weight 1
  for (uint c = 0; c < p.classes; ++c) {
    const float log_p = (l[c] - top) - log_total;
    d[c] = (exp(log_p) - (c == y ? 1.0f : 0.0f)) * s;
    if (c == y) loss[id] = -log_p;
  }
}

// HL-Gauss's soft-label cross-entropy of each member output, loss[m] = −Σ_c p[b, c] · log softmax(l)_c, and its
// gradient (softmax − p) · inv_count (inv_count = 1 / (B · k)); one thread per member row; padding rows give 0. The
// classes (bins) are summed in order.
kernel void hl_gauss_ce(device const float *logits [[buffer(0)]], device const float *labels [[buffer(1)]],
                        device float *d_logits [[buffer(2)]], device float *loss [[buffer(3)]],
                        device const float *weights [[buffer(4)]], constant NetParams &p [[buffer(5)]],
                        uint m [[thread_position_in_grid]]) {
  if (m >= p.rows_pad * p.members) return;
  device const float *l = logits + m * p.classes;
  device float *d = d_logits + m * p.classes;
  const uint b = m / p.members;
  if (b >= p.batch) {
    for (uint c = 0; c < p.classes; ++c) d[c] = 0.0f;
    loss[m] = 0.0f;
    return;
  }
  device const float *y = labels + b * p.classes;
  float top = l[0];
  for (uint c = 1; c < p.classes; ++c) top = max(top, l[c]);
  float total = 0.0f;
  for (uint c = 0; c < p.classes; ++c) total += exp(l[c] - top);
  const float log_total = log(total);
  float sum = 0.0f;
  const float s = p.inv_count * weights[b];  // inv_count itself at weight 1
  for (uint c = 0; c < p.classes; ++c) {
    const float log_p = (l[c] - top) - log_total;
    d[c] = (exp(log_p) - y[c]) * s;
    sum -= y[c] * log_p;
  }
  loss[m] = sum;
}

// Squared error of each member output and target (objective "mse": one output per target), loss[m · K + t] =
// (o − y)², and its gradient 2 (o − y) · inv_count (inv_count = 1 / (B · k · K)); one thread per (row, target);
// padding rows give 0.
kernel void mse_loss(device const float *logits [[buffer(0)]], device const float *targets [[buffer(1)]],
                     device float *d_logits [[buffer(2)]], device float *loss [[buffer(3)]],
                     device const float *weights [[buffer(4)]], constant NetParams &p [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
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
// classes of target t at units u = 4q .. 4q + 3 (the row's four values read once for all of their sums); the thread
// q == ldw / 4 computes d_head_bias[j, o] (unit value 1). A threadgroup holds up to TARGETS_PER_GROUP (host) targets
// of the same units. Stored, or added with `accumulate`.
kernel void head_backward(device const act_t *a [[buffer(0)]], device const float *d_logits [[buffer(1)]],
                          device float *d_head [[buffer(2)]], device float *d_head_bias [[buffer(3)]],
                          constant NetParams &p [[buffer(4)]], uint3 id [[thread_position_in_grid]]) {
  const uint q = id.x, t = id.y, j = id.z, outs = p.targets * p.classes, o = t * p.classes;
  if (q > p.ldw / 4 || t >= p.targets || j >= p.members) return;
  const bool bias = q == p.ldw / 4;
  float4 acc[8] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f),
                   float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
  for (uint b = 0; b < p.batch; ++b) {
    const uint m = b * p.members + j;
    const float4 v =
        bias ? float4(1.0f, 0.0f, 0.0f, 0.0f) : float4(*(device const vec<act_t, 4> *)(a + m * p.ldw + 4 * q));
    device const float *g = d_logits + m * outs + o;
    UNROLL for (uint c = 0; c < MAX_CLASSES; ++c) {
      if (c < p.classes) {
        const float gc = g[c];
        acc[c].x += v.x * gc;
        acc[c].y += v.y * gc;
        acc[c].z += v.z * gc;
        acc[c].w += v.w * gc;
      }
    }
  }
  UNROLL for (uint c = 0; c < MAX_CLASSES; ++c) {
    if (c >= p.classes) continue;
    if (bias) {
      device float *out = d_head_bias + j * outs + o + c;
      *out = p.accumulate ? *out + acc[c].x : acc[c].x;
    } else {
      device float4 *out = (device float4 *)(d_head + (j * outs + o + c) * p.ldw + 4 * q);
      *out = p.accumulate ? *out + acc[c] : acc[c];
    }
  }
}

// The top block's backward: da = Σ_o d_logits[m, o] · head[j, o, u] over the K · c outputs in order, then
// dz = da ⊙ mask ⊙ (z > 0) with z = T ⊙ s + b recomputed, and dt = dz ⊙ s. Thread (u, y) computes HBB_ROWS batch rows
// b = g · HBB_ROWS + r of member j, y = j · groups + g, so each head value is read once for all of them (and the
// SIMD-groups of a threadgroup share the member's head).
constant constexpr uint HBB_ROWS = 4;

kernel void head_block_backward(device const float *d_logits [[buffer(0)]], device const float *head [[buffer(1)]],
                                device const act_t *T [[buffer(2)]], device const float *s [[buffer(3)]],
                                device const float *bias [[buffer(4)]], device const uint *rows [[buffer(5)]],
                                device act_t *dz [[buffer(6)]], device act_t *dt [[buffer(7)]],
                                constant NetParams &p [[buffer(8)]], uint2 id [[thread_position_in_grid]]) {
  const uint groups = (p.rows_pad + HBB_ROWS - 1) / HBB_ROWS;
  const uint u = id.x, j = id.y / groups, b0 = (id.y % groups) * HBB_ROWS;
  if (u >= p.ldw || j >= p.members) return;
  const uint outs = p.targets * p.classes;
  device const float *h = head + j * outs * p.ldw + u;
  device const float *g[HBB_ROWS];
  UNROLL for (uint r = 0; r < HBB_ROWS; ++r) g[r] = d_logits + (min(b0 + r, p.rows_pad - 1) * p.members + j) * outs;
  float da[HBB_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint o = 0; o < outs; ++o) {
    const float w = h[o * p.ldw];
    UNROLL for (uint r = 0; r < HBB_ROWS; ++r) da[r] += g[r][o] * w;
  }
  const float sv = s[j * p.ldw + u], bv = bias[j * p.ldw + u];
  UNROLL for (uint r = 0; r < HBB_ROWS; ++r) {
    if (b0 + r >= p.rows_pad) break;
    const uint m = (b0 + r) * p.members + j, at = m * p.ldw + u;
    const float z = float(T[at]) * sv + bv;
    const float grad = (kept(p, row_hash(p, rows, m), j, u) && z > 0.0f) ? da[r] * p.scale : 0.0f;
    dz[at] = act_t(grad);
    dt[at] = act_t(grad * sv);
  }
}

// The batch sums run in two stages, each in a fixed order: chunk q of rows b in [q · chunk, (q + 1) · chunk) writes
// partial sums, then sum_partials adds the chunks in order q = 0, 1, ....

// First stage of d_bias[j, u] = Σ_b dz[b·k + j, u] and d_s[j, u] = Σ_b dz · T: partials (chunks, k, ldw) each.
kernel void member_sums(device const act_t *dz [[buffer(0)]], device const act_t *T [[buffer(1)]],
                        device float *part_bias [[buffer(2)]], device float *part_s [[buffer(3)]],
                        constant NetParams &p [[buffer(4)]], uint3 id [[thread_position_in_grid]]) {
  const uint u = id.x, j = id.y, q = id.z;
  if (u >= p.ldw || j >= p.members || q >= p.chunks) return;
  float sum_b = 0.0f, sum_s = 0.0f;
  for (uint b = q * p.chunk; b < min((q + 1) * p.chunk, p.batch); ++b) {
    const uint at = (b * p.members + j) * p.ldw + u;
    const float g = float(dz[at]);
    sum_b += g;
    sum_s += g * float(T[at]);
  }
  part_bias[(q * p.members + j) * p.ldw + u] = sum_b;
  part_s[(q * p.members + j) * p.ldw + u] = sum_s;
}

// Second stage: out[i] = Σ_q part[q · n + i] for i < n = k · din, in chunk order; added to out[i] with `accumulate`.
kernel void sum_partials(device const float *part [[buffer(0)]], device float *out [[buffer(1)]],
                         constant NetParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const uint n = p.members * p.din;
  if (i >= n) return;
  float acc = 0.0f;
  for (uint q = 0; q < p.chunks; ++q) acc += part[q * n + i];
  out[i] = p.accumulate ? out[i] + acc : acc;
}

// Expected class of the member-averaged softmax, out[b · K + t], one thread per (batch row, target) (classes <= 8).
kernel void predict_class(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]],
                          constant NetParams &p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  const uint b = id / p.targets, outs = p.targets * p.classes, o = (id % p.targets) * p.classes;
  if (b >= p.batch) return;
  float mean[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint j = 0; j < p.members; ++j) {
    device const float *l = logits + (b * p.members + j) * outs + o;
    float top = l[0];
    for (uint c = 1; c < p.classes; ++c) top = max(top, l[c]);
    float total = 0.0f;
    for (uint c = 0; c < p.classes; ++c) total += exp(l[c] - top);
    for (uint c = 0; c < p.classes; ++c) mean[c] += exp(l[c] - top) / total;
  }
  float expected = 0.0f;
  for (uint c = 0; c < p.classes; ++c) expected += float(c) * (mean[c] / float(p.members));
  out[id] = expected;
}

// Mean of the members' outputs (objective "mse"), out[b · K + t], one thread per (batch row, target); summed in member
// order.
kernel void predict_mean(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]],
                         constant NetParams &p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  const uint b = id / p.targets, t = id % p.targets;
  if (b >= p.batch) return;
  float sum = 0.0f;
  for (uint j = 0; j < p.members; ++j) sum += logits[(b * p.members + j) * p.targets + t];
  out[id] = sum / float(p.members);
}

// HL-Gauss's prediction, out[b]: the expected bin centre low + (c + ½) · bin_width under the members' mean softmax,
// one thread per batch row (n_bins <= MAX_BINS); members, then bins, summed in order.
constant constexpr uint MAX_BINS = 512;

kernel void predict_centre(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]],
                           constant NetParams &p [[buffer(2)]], uint b [[thread_position_in_grid]]) {
  if (b >= p.batch) return;
  float mean[MAX_BINS];
  for (uint c = 0; c < p.classes; ++c) mean[c] = 0.0f;
  for (uint j = 0; j < p.members; ++j) {
    device const float *l = logits + (b * p.members + j) * p.classes;
    float top = l[0];
    for (uint c = 1; c < p.classes; ++c) top = max(top, l[c]);
    float total = 0.0f;
    for (uint c = 0; c < p.classes; ++c) total += exp(l[c] - top);
    for (uint c = 0; c < p.classes; ++c) mean[c] += exp(l[c] - top) / total;
  }
  float expected = 0.0f;
  for (uint c = 0; c < p.classes; ++c) {
    expected += (p.low + (float(c) + 0.5f) * p.bin_width) * (mean[c] / float(p.members));
  }
  out[b] = expected;
}

// ---------------------------------------------------------------------------------------------------------------------
// The temporal encoder (idun.tabm.encoder): per step a shared Linear(C → E) and GELU, then a GRU(E → H) over the steps,
// oldest first; its last state takes block 0's first H input columns. Always float. Pair i = b · steps + s is row b's
// step s. seq_forward and seq_backward run one thread per (hidden unit u, row), SEQ_ROWS rows per threadgroup, a row's
// state in threadgroup memory; every sum runs over its index in order. The weight gradients are transposed GEMMs over
// chunks of pairs (seq_weight_grad) and the bias gradients column sums over the same chunks (seq_column_partials), each
// added up in chunk order by sum_partials.
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

constant constexpr uint SEQ_ROWS = 4;   // rows per threadgroup (host: SEQ_ROWS)
constant constexpr uint MAX_ENC = 128;  // the most units of E and of H (params.MAX_ENCODER_WIDTH)

static float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Φ(x) through erf as Abramowitz & Stegun 7.1.26 give it (|error| < 1.5e-7; a US government publication): Metal has no
// erf. The float64 reference computes Φ exactly.
static float normal_cdf(float x) {
  const float a = fabs(x) * 0.70710678118654752f, t = 1.0f / (1.0f + 0.3275911f * a);
  const float poly =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float tail = 0.5f * poly * exp(-a * a);  // ½ erfc(a)
  return x >= 0.0f ? 1.0f - tail : tail;
}

static float gelu(float x) { return x * normal_cdf(x); }

static float gelu_grad(float x) { return normal_cdf(x) + x * exp(-0.5f * x * x) * 0.39894228040143268f; }

// Forward. Thread (u, rr) of threadgroup g runs unit u of row b = g · SEQ_ROWS + rr through every step and stores, per
// pair i: the embedding's pre-activation and output (pre, emb: i · ld_e + c), the state before the step (hprev:
// i · ld_h + u) and r, z, n and W_hn h + b_hn (gates: (4 i + gate) · ld_h + u). The last state goes to the row's first
// H inputs (0 on a padding row).
kernel void seq_forward(device const float *x [[buffer(0)]], device const float *we [[buffer(1)]],
                        device const float *be [[buffer(2)]], device const float *wih [[buffer(3)]],
                        device const float *whh [[buffer(4)]], device const float *bih [[buffer(5)]],
                        device const float *bhh [[buffer(6)]], device float *pre [[buffer(7)]],
                        device float *emb [[buffer(8)]], device float *hprev [[buffer(9)]],
                        device float *gates [[buffer(10)]], device act_t *inputs [[buffer(11)]],
                        constant SeqParams &p [[buffer(12)]], uint2 tid [[thread_position_in_threadgroup]],
                        uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup float e_tg[SEQ_ROWS * MAX_ENC], h_tg[SEQ_ROWS * MAX_ENC];
  const uint u = tid.x, rr = tid.y, b = tg.y * SEQ_ROWS + rr, H = p.hidden;
  threadgroup float *e_row = e_tg + rr * MAX_ENC, *h_row = h_tg + rr * MAX_ENC;
  h_row[u] = 0.0f;
  for (uint s = 0; s < p.steps; ++s) {
    const uint i = b * p.steps + s;
    device const float *xs = x + i * p.ld_in;
    for (uint c = u; c < p.embed; c += H) {
      float acc = 0.0f;
      for (uint k = 0; k < p.inputs; ++k) acc += we[c * p.ld_in + k] * xs[k];
      const float v = acc + be[c];
      pre[i * p.ld_e + c] = v;
      e_row[c] = emb[i * p.ld_e + c] = gelu(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gi[3], gh[3];
    UNROLL for (uint g = 0; g < 3; ++g) {
      const uint row = g * H + u;
      float ai = 0.0f, ah = 0.0f;
      for (uint c = 0; c < p.embed; ++c) ai += wih[row * p.ld_e + c] * e_row[c];
      for (uint c = 0; c < H; ++c) ah += whh[row * p.ld_h + c] * h_row[c];
      gi[g] = ai + bih[row];
      gh[g] = ah + bhh[row];
    }
    const float r = sigmoid(gi[0] + gh[0]), z = sigmoid(gi[1] + gh[1]), n = tanh(gi[2] + r * gh[2]);
    const float h = h_row[u];
    device float *gt = gates + 4 * i * p.ld_h + u;
    hprev[i * p.ld_h + u] = h;
    gt[0] = r;
    gt[p.ld_h] = z;
    gt[2 * p.ld_h] = n;
    gt[3 * p.ld_h] = gh[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    h_row[u] = (1.0f - z) * n + z * h;
  }
  inputs[b * p.ldd + u] = act_t(b < p.batch ? h_row[u] : 0.0f);
}

// The gradient at block 0's encoder inputs, dU = dT_0 · W_0[:, :ld_h] (W_0 stored (ldw, ldd)): one GEMM per batch.
kernel void seq_input_grad(device const act_t *A [[buffer(0)]], device const float *B [[buffer(1)]],
                           device float *C [[buffer(2)]], constant GemmParams &g [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup act_t As[BM * (BK + PAD)];
  threadgroup float Bs[BK * (BN + PAD)];
  const Plain<float> epi{C, g.ldc};
  gemm<act_t, float, false, false, false, false>(A, B, nullptr, g, tg, sg, lane, As, Bs, epi);
}

// The encoder's output gradient dh[b, c] = Σ_j r_0[j, c] · dU[b · k + j, c] over the members j = 0, 1, ... in order;
// 0 on padding rows and columns. One thread per (c, b).
kernel void seq_member_sum(device const float *dU [[buffer(0)]], device const float *r [[buffer(1)]],
                           device float *dh [[buffer(2)]], constant SeqParams &p [[buffer(3)]],
                           uint2 id [[thread_position_in_grid]]) {
  const uint c = id.x, b = id.y;
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
kernel void seq_backward(device const float *dh_in [[buffer(0)]], device const float *wih [[buffer(1)]],
                         device const float *whh [[buffer(2)]], device const float *pre [[buffer(3)]],
                         device const float *hprev [[buffer(4)]], device const float *gates [[buffer(5)]],
                         device float *d_gi [[buffer(6)]], device float *d_gh [[buffer(7)]],
                         device float *d_pre [[buffer(8)]], constant SeqParams &p [[buffer(9)]],
                         uint2 tid [[thread_position_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup float gi_tg[SEQ_ROWS * 3 * MAX_ENC], gh_tg[SEQ_ROWS * 3 * MAX_ENC];
  const uint u = tid.x, rr = tid.y, b = tg.y * SEQ_ROWS + rr, H = p.hidden;
  threadgroup float *gi_row = gi_tg + rr * 3 * MAX_ENC, *gh_row = gh_tg + rr * 3 * MAX_ENC;
  float dh = b < p.batch ? dh_in[b * p.ld_h + u] : 0.0f;
  for (uint s = p.steps; s-- > 0;) {
    const uint i = b * p.steps + s;
    device const float *gt = gates + 4 * i * p.ld_h + u;
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint g = 0; g < 3 * H; ++g) acc += whh[g * p.ld_h + u] * gh_row[g];
    for (uint c = u; c < p.embed; c += H) {
      float de = 0.0f;
      for (uint g = 0; g < 3 * H; ++g) de += wih[g * p.ld_e + c] * gi_row[g];
      d_pre[i * p.ld_e + c] = de * gelu_grad(pre[i * p.ld_e + c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dh = dh * z + acc;
  }
}

// A weight gradient's partial over chunk q (threadgroup z) of pairs: C_q = A_qᵀ · B_q, A stored (pairs, M), B (pairs,
// N); sum_partials adds the chunks in order.
kernel void seq_weight_grad(device const float *A [[buffer(0)]], device const float *B [[buffer(1)]],
                            device float *C [[buffer(2)]], constant MemberGemmParams &mp [[buffer(3)]],
                            uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[BK * (BM + PAD)];
  threadgroup float Bs[BK * (BN + PAD)];
  const Plain<float> epi{C + tg.z * mp.c_step, mp.g.ldc};
  gemm<float, float, true, false, false, false>(A + tg.z * mp.a_step, B + tg.z * mp.b_step, nullptr, mp.g, tg.xy, sg,
                                                lane, As, Bs, epi);
}

// A bias gradient's partial over chunk q of pairs: part[q · din + c] = Σ_i A[i · din + c] over the chunk's pairs in
// order (NetParams: din the columns, chunk the pairs per chunk, chunks). One thread per (c, q).
kernel void seq_column_partials(device const float *A [[buffer(0)]], device float *part [[buffer(1)]],
                                constant NetParams &p [[buffer(2)]], uint2 id [[thread_position_in_grid]]) {
  const uint c = id.x, q = id.y;
  if (c >= p.din || q >= p.chunks) return;
  float acc = 0.0f;
  for (uint i = q * p.chunk; i < (q + 1) * p.chunk; ++i) acc += A[i * p.din + c];
  part[q * p.din + c] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// Piecewise-linear numeric embeddings (TabMParams.num_emb_type "pwl"; idun.tabm.embeddings): the first F inputs of a
// row (raw, (rows_pad, ld_raw)) become F · d embedded values at block 0's input columns 0 .. F · d; the other inputs
// follow them, written there by the host. The encoding (scale, shift (F, P), single (F,)) is fixed; the weights are
// linear and linear_bias (F, d) and ple (F, P, d). Block 0's input gradient for the embedded columns comes from
// seq_input_grad (dU = dT_0 · W_0[:, :ld_e]) and emb_member_sum; the weight gradients are sums over the batch rows in
// order, one thread per weight.

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
static float ple_component(float v, device const float *scale, device const float *shift, device const uint *single,
                           uint f, uint c, uint P) {
  const float q = v * scale[f * P + c] + shift[f * P + c];
  if (P == 1) return q;
  if (c == 0) return min(q, 1.0f);
  if (c + 1 < P) return clamp(q, 0.0f, 1.0f);
  return single[f] != 0u ? q : max(q, 0.0f);
}

// x[b, i] = v · linear[i] + linear_bias[i] + Σ_c q_c(v) · ple[f, c, j] (c = 0, 1, ... in order) for i = f · d + j and
// v = raw[b, f]; 0 on padding rows. One thread per (i, b).
kernel void emb_forward(device const float *raw [[buffer(0)]], device const float *scale [[buffer(1)]],
                        device const float *shift [[buffer(2)]], device const uint *single [[buffer(3)]],
                        device const float *linear [[buffer(4)]], device const float *linear_bias [[buffer(5)]],
                        device const float *ple [[buffer(6)]], device act_t *x [[buffer(7)]],
                        constant EmbParams &p [[buffer(8)]], uint2 id [[thread_position_in_grid]]) {
  const uint i = id.x, b = id.y, d = p.dim, P = p.components;
  if (i >= p.features * d || b >= p.rows_pad) return;
  const uint f = i / d, j = i % d;
  float acc = 0.0f;
  if (b < p.batch) {
    const float v = raw[b * p.ld_raw + f];
    acc = v * linear[i] + linear_bias[i];
    for (uint c = 0; c < P; ++c) acc += ple_component(v, scale, shift, single, f, c, P) * ple[(f * P + c) * d + j];
  }
  x[b * p.ldd + i] = act_t(acc);
}

// The embedded values' gradient d_emb[b, i] = Σ_j r_0[j, i] · dU[b · k + j, i] over the members j = 0, 1, ... in order;
// 0 on padding rows and columns. One thread per (i, b).
kernel void emb_member_sum(device const float *dU [[buffer(0)]], device const float *r [[buffer(1)]],
                           device float *d_emb [[buffer(2)]], constant EmbParams &p [[buffer(3)]],
                           uint2 id [[thread_position_in_grid]]) {
  const uint i = id.x, b = id.y;
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
kernel void emb_weight_grad(device const float *raw [[buffer(0)]], device const float *scale [[buffer(1)]],
                            device const float *shift [[buffer(2)]], device const uint *single [[buffer(3)]],
                            device const float *d_emb [[buffer(4)]], device float *d_linear [[buffer(5)]],
                            device float *d_linear_bias [[buffer(6)]], device float *d_ple [[buffer(7)]],
                            constant EmbParams &p [[buffer(8)]], uint t [[thread_position_in_grid]]) {
  const uint width = p.features * p.dim, P = p.components;
  if (t >= (P + 2) * width) return;
  const uint part = t / width, i = t % width, f = i / p.dim, j = i % p.dim;
  float acc = 0.0f;
  for (uint b = 0; b < p.batch; ++b) {
    const float g = d_emb[b * p.ld_e + i], v = raw[b * p.ld_raw + f];
    acc += (part == 0 ? v : part == 1 ? 1.0f : ple_component(v, scale, shift, single, f, part - 2, P)) * g;
  }
  device float *out = part == 0 ? d_linear + i : part == 1 ? d_linear_bias + i : d_ple + (f * P + part - 2) * p.dim + j;
  *out = (p.accumulate != 0u ? *out : 0.0f) + acc;
}
