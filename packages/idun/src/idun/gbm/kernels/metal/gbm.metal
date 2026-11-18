// gbm.metal - idun_gbm kernels for the Metal backend (docs/architecture/metal-kernels.md).
//
// Each kernel mirrors a numba kernel of idun/gbm/backends/cpu.py operation for operation, so the two backends
// produce identical bits (docs/architecture/determinism.md). Rules: integer atomics only; floats only in per-row
// `+`, `*` and `rint` (the runtime prepends `#pragma METAL fp contract(off)`); draws from the row's absolute index;
// one params struct per kernel, mirrored by a ctypes.Structure in idun/gbm/backends/device.py.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint NAMED = 0;
constant constexpr uint CATEGORICAL = 2;
constant constexpr uint MISSING_BIN = 255;
constant constexpr uint MAX_LEAVES = 256;
constant constexpr uint LANES = 32;
constant constexpr uint OBFUSCATED_BINS = 5;
constant constexpr uint MAX_ENTRIES = 16;  // entries per chunk: at most one per (model, slot) pair

// lowbias32 (Chris Wellons, public domain): idun.rng.
inline uint lowbias32(uint x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

inline uint row_bits(uint key, uint row) { return lowbias32(key ^ row) >> 8; }

// Bin of panel row `row` in one feature: named bins and categorical codes are bytes, obfuscated codes are nibbles.
// Categorical column c starts at c * cat_stride (0 for a single column).
inline uint bin_of(int kind, int column, uint row, device const uchar *named, uint named_stride,
                   device const uchar *packed, uint packed_stride, device const uchar *country, uint cat_stride) {
  if (uint(kind) == NAMED) {
    return named[ulong(column) * named_stride + row];
  }
  if (uint(kind) == CATEGORICAL) {
    return country[ulong(column) * cat_stride + row];
  }
  uchar byte = packed[ulong(column) * packed_stride + (row >> 1)];
  return (byte >> ((row & 1u) << 2)) & 15u;
}

// Child taken at an internal node: 0 = left, 1 = right. Pass-through nodes (kind 0) send every row left.
inline uint goes_right(uint kind, uint bin, uint threshold, uint missing_left, ulong cat_left) {
  if (kind == 0) {
    return 0;
  }
  if (kind == 2) {
    return ((cat_left >> ulong(bin)) & 1ul) != 0 ? 0 : 1;
  }
  if (bin == MISSING_BIN) {
    return missing_left != 0 ? 0 : 1;
  }
  return bin > threshold ? 1 : 0;
}

struct CountParams {
  uint n;
};

struct RngParams {
  uint key;
  uint first_row;
  uint n;
};

// Test kernel: the draws of idun.rng.uniform_bits.
kernel void rng_uniform_bits(device uint *out [[buffer(0)]], constant RngParams &p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < p.n) {
    out[i] = row_bits(p.key, p.first_row + i);
  }
}

// Test kernel: `a * b + c` must round twice, as a separate multiply and add.
kernel void fma_probe(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      device const float *c [[buffer(2)]], device float *out [[buffer(3)]],
                      constant CountParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) {
    out[i] = a[i] * b[i] + c[i];
  }
}

// Every per-row array of a fit holds K models back to back: element (model, i) is at model * n + i, and kernels
// that touch them run on a (rows, models) grid. The bag and the feature columns are shared by the K models
// (the models of one seed share the bag and the feature sample). K = 1 is a single fit.

struct ModelParams {
  uint n;
  uint k;
};

// g = pred - y per model; max |g| per model into max_bits[model] (the bits of a non-negative float order like it).
kernel void compute_gradients(device const float *pred [[buffer(0)]], device const float *y [[buffer(1)]],
                              device float *g [[buffer(2)]], device atomic_uint *max_bits [[buffer(3)]],
                              constant ModelParams &p [[buffer(4)]], uint2 at [[thread_position_in_grid]],
                              uint2 lane_xy [[thread_position_in_threadgroup]]) {
  float value = 0.0f;
  ulong index = ulong(at.y) * p.n + at.x;
  if (at.x < p.n) {
    value = pred[index] - y[index];
    g[index] = value;
  }
  float largest = simd_max(fabs(value));
  if (lane_xy.x % 32 == 0) {
    atomic_fetch_max_explicit(max_bits + at.y, as_type<uint>(largest), memory_order_relaxed);
  }
}

struct QuantizeParams {
  uint n;
  uint first_row;
  uint key;
  int half_bins;
};

// Stochastic rounding toward zero, clamped (determinism.md §2), of one in-bag row's gradient.
inline char round_gradient(float g, float inv_scale, constant QuantizeParams &p, uint i) {
  float u = float(row_bits(p.key, p.first_row + i)) * 0x1p-24f;
  float x = g * inv_scale;
  float v = g >= 0.0f ? x + u : x - u;
  return char(clamp(int(v), -p.half_bins, p.half_bins));
}

// q = the rounded g; 0 for rows out of the bag. One key for every model.
kernel void quantize_gradients(device const float *g [[buffer(0)]], device const uchar *bag [[buffer(1)]],
                               device char *q [[buffer(2)]], device const float *inv_scale [[buffer(3)]],
                               constant QuantizeParams &p [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) {
    return;
  }
  ulong index = ulong(at.y) * p.n + at.x;
  q[index] = bag[at.x] == 0 ? char(0) : round_gradient(g[index], inv_scale[at.y], p, at.x);
}

// Multi-target (no g buffer): max |pred - y| per target, as compute_gradients without storing g.
kernel void max_residuals(device const float *pred [[buffer(0)]], device const float *y [[buffer(1)]],
                          device atomic_uint *max_bits [[buffer(2)]], constant ModelParams &p [[buffer(3)]],
                          uint2 at [[thread_position_in_grid]], uint2 lane_xy [[thread_position_in_threadgroup]]) {
  float value = 0.0f;
  ulong index = ulong(at.y) * p.n + at.x;
  if (at.x < p.n) {
    value = pred[index] - y[index];
  }
  float largest = simd_max(fabs(value));
  if (lane_xy.x % 32 == 0) {
    atomic_fetch_max_explicit(max_bits + at.y, as_type<uint>(largest), memory_order_relaxed);
  }
}

// Multi-target (no g buffer): quantize_gradients of g = pred - y, the same float32 subtraction as compute_gradients.
kernel void quantize_residuals(device const float *pred [[buffer(0)]], device const float *y [[buffer(1)]],
                               device const uchar *bag [[buffer(2)]], device char *q [[buffer(3)]],
                               device const float *inv_scale [[buffer(4)]], constant QuantizeParams &p [[buffer(5)]],
                               uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) {
    return;
  }
  ulong index = ulong(at.y) * p.n + at.x;
  q[index] = bag[at.x] == 0 ? char(0) : round_gradient(pred[index] - y[index], inv_scale[at.y], p, at.x);
}

struct HistogramParams {
  uint n;
  uint first_row;
  uint n_features;
  uint cells;
  uint named_stride;
  uint packed_stride;
  uint width;  // nodes per model in `slots`
  uint cat_stride;
};

struct SharedHistogramParams {
  uint n;
  uint first_row;
  uint n_features;
  uint cells;
  uint named_stride;
  uint packed_stride;
  uint width;  // nodes per model in `slots`
  uint cat_stride;
  uint rows_per_group;
  uint slots_per_chunk;
  uint chunks;  // slot chunks per model
};

// Integer (sum q, count) per (slot, cell) over in-bag rows whose node has a slot, for the given features (named
// and categorical features here). One thread per (row, model), device atomics. Model k's slots start at slot_base[k].
kernel void build_histograms(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                             device const int *offsets [[buffer(2)]], device const uchar *named [[buffer(3)]],
                             device const uchar *packed [[buffer(4)]], device const uchar *country [[buffer(5)]],
                             device const uchar *bag [[buffer(6)]], device const uchar *node [[buffer(7)]],
                             device const char *q [[buffer(8)]], device const int *slots [[buffer(9)]],
                             device const int *slot_base [[buffer(10)]], device atomic_int *hist [[buffer(11)]],
                             constant HistogramParams &p [[buffer(12)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n || bag[at.x] == 0) {
    return;
  }
  ulong index = ulong(at.y) * p.n + at.x;
  int slot = slots[at.y * p.width + node[index]];
  if (slot < 0) {
    return;
  }
  uint row = p.first_row + at.x;
  int qi = int(q[index]);
  device atomic_int *base = hist + ulong(slot_base[at.y] + slot) * p.cells * 2;
  for (uint j = 0; j < p.n_features; ++j) {
    uint cell = uint(offsets[j]) + bin_of(kinds[j], columns[j], row, named, p.named_stride, packed, p.packed_stride,
                                          country, p.cat_stride);
    atomic_fetch_add_explicit(base + 2 * cell, qi, memory_order_relaxed);
    atomic_fetch_add_explicit(base + 2 * cell + 1, 1, memory_order_relaxed);
  }
}

// build_histograms' sums, a threadgroup's in threadgroup memory first: group (row group, feature, model * chunks +
// chunk) counts the rows of one row group whose slot is in the chunk, for one named or categorical feature of
// `widths[feature]` cells, with integer atomics in threadgroup memory, then adds its cells that hold a row to the
// histogram. Integer sums: the same in any order.
kernel void build_histograms_shared(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                                    device const int *offsets [[buffer(2)]], device const int *widths [[buffer(3)]],
                                    device const uchar *named [[buffer(4)]], device const uchar *packed [[buffer(5)]],
                                    device const uchar *country [[buffer(6)]], device const uchar *bag [[buffer(7)]],
                                    device const uchar *node [[buffer(8)]], device const char *q [[buffer(9)]],
                                    device const int *slots [[buffer(10)]], device const int *slot_base [[buffer(11)]],
                                    device atomic_int *hist [[buffer(12)]],
                                    constant SharedHistogramParams &p [[buffer(13)]],
                                    threadgroup atomic_int *local [[threadgroup(0)]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint index [[thread_index_in_threadgroup]],
                                    uint3 size [[threads_per_threadgroup]]) {
  uint j = group.y;
  uint model = group.z / p.chunks;
  int first_slot = int((group.z % p.chunks) * p.slots_per_chunk);
  uint bins = uint(widths[j]);
  uint span = p.slots_per_chunk * bins * 2;  // (slot, bin, {sum, count})
  for (uint k = index; k < span; k += size.x) {
    atomic_store_explicit(local + k, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint stop = min((group.x + 1) * p.rows_per_group, p.n);
  for (uint i = group.x * p.rows_per_group + index; i < stop; i += size.x) {
    if (bag[i] == 0) {
      continue;
    }
    ulong index = ulong(model) * p.n + i;
    int slot = slots[model * p.width + node[index]] - first_slot;
    if (slot < 0 || slot >= int(p.slots_per_chunk)) {
      continue;
    }
    uint bin = bin_of(kinds[j], columns[j], p.first_row + i, named, p.named_stride, packed, p.packed_stride, country,
                      p.cat_stride);
    uint cell = (uint(slot) * bins + bin) * 2;
    atomic_fetch_add_explicit(local + cell, int(q[index]), memory_order_relaxed);
    atomic_fetch_add_explicit(local + cell + 1, 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device atomic_int *base = hist + (ulong(slot_base[model]) + ulong(first_slot)) * p.cells * 2 + ulong(offsets[j]) * 2;
  for (uint k = index; k < span / 2; k += size.x) {
    int count = atomic_load_explicit(local + 2 * k + 1, memory_order_relaxed);
    if (count != 0) {
      device atomic_int *target = base + (ulong(k / bins) * p.cells + k % bins) * 2;
      atomic_fetch_add_explicit(target, atomic_load_explicit(local + 2 * k, memory_order_relaxed),
                                memory_order_relaxed);
      atomic_fetch_add_explicit(target + 1, count, memory_order_relaxed);
    }
  }
}

struct ObfuscatedParams {
  uint n;
  uint first_row;
  uint n_features;
  uint cells;
  uint packed_stride;
  uint rows_per_group;
  uint width;      // nodes per model in `slots`
  uint q_bits;     // bits of |q|: bit_length(half_bins)
  uint popcount;   // 1: bitmask path (every entry has at most 2 slots), 0: per-row path
  uint max_pairs;  // (model, slot) pairs per chunk at most: sizes the threadgroup memory
};

// One entry of a chunk: model `model`'s slots [first_slot, end_slot), stored from local pair `pair_base` on.
struct Entry {
  uint model;
  uint first_slot;
  uint end_slot;
  uint pair_base;
};

// Rows of a 32-row block whose code is > t, t = 0..3, from 16 bytes of nibbles (two rows per byte): bit j = row j.
inline uint above(uint4 codes, uint t) {
  uint mask = 0;
  for (uint w = 0; w < 4; ++w) {
    uint x = codes[w];
    uint hit = t == 0   ? (x | (x >> 1) | (x >> 2) | (x >> 3))
               : t == 1 ? ((x >> 1) | (x >> 2) | (x >> 3))
               : t == 2 ? (((x >> 1) & x) | (x >> 2) | (x >> 3))
                        : ((x >> 2) | (x >> 3));
    hit &= 0x11111111u;  // one bit per nibble, at 4k; compress to 8 contiguous bits
    hit = (hit | (hit >> 3)) & 0x03030303u;
    hit = (hit | (hit >> 6)) & 0x000F000Fu;
    hit = (hit | (hit >> 12)) & 0x000000FFu;
    mask |= hit << (8 * w);
  }
  return mask;
}

// Histograms of obfuscated features for K models, one feature per SIMD lane ("lane owns column"). A threadgroup is
// one SIMD group: its 32 lanes take 32 sampled features and walk the same rows in 32-row blocks aligned on absolute
// rows, for every (model, slot) pair of one chunk (`entries` [chunk_start[c], chunk_start[c + 1])). Per block, lane j
// writes row j's word for each entry to threadgroup memory (-1 = skip, else local pair * 256 + q + 128) and loads its
// feature's 32 codes once with a 16-byte load: the codes, and on the bitmask path the "code > t" masks, are shared by
// every model of the chunk. Paths:
//   popcount: per pair, ballots give its rows and the bits of |q| by sign; count and sum q of (pair, code > t),
//     t = -1..3, are popcounts accumulated once per block; bins follow by differences.
//   per row: for each row (unrolled) and entry, the word is read as a broadcast and each lane adds q * 2^16 + 1 to
//     its (pair, bin) counter: sum q high, count low, exact while rows_per_group * half_bins <= 32767.
// Counters are lane-minor in threadgroup memory (no atomics, no bank conflicts); lanes finally add their non-zero sums
// and counts to the device histogram with integer atomics. Grid: (feature groups, row groups, chunks). Threadgroup
// memory: max_pairs * (10 or 5) * 32 counters, then max_pairs * 32 words. Columns must be 16-byte aligned, with 16
// bytes readable past the last row.
kernel void histograms_obfuscated(device const int *columns [[buffer(0)]], device const int *offsets [[buffer(1)]],
                                  device const uchar *packed [[buffer(2)]], device const uchar *bag [[buffer(3)]],
                                  device const uchar *node [[buffer(4)]], device const char *q [[buffer(5)]],
                                  device const int *slots [[buffer(6)]], device const int *slot_base [[buffer(7)]],
                                  device const Entry *entries [[buffer(8)]],
                                  device const uint *chunk_start [[buffer(9)]], device atomic_int *hist [[buffer(10)]],
                                  constant ObfuscatedParams &p [[buffer(11)]],
                                  threadgroup int *local [[threadgroup(0)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Entry chunk[MAX_ENTRIES];
  uint feature = group.x * LANES + lane;
  uint first_entry = chunk_start[group.z];
  uint n_entries = chunk_start[group.z + 1] - first_entry;
  if (lane < n_entries) {
    chunk[lane] = entries[first_entry + lane];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  Entry last = chunk[n_entries - 1];
  uint n_pairs = last.pair_base + (last.end_slot - last.first_slot);
  uint per_pair = p.popcount != 0 ? OBFUSCATED_BINS * 2 : OBFUSCATED_BINS;
  threadgroup int *words = local + p.max_pairs * per_pair * LANES;
  for (uint k = 0; k < n_pairs * per_pair; ++k) {
    local[k * LANES + lane] = 0;
  }
  bool active = feature < p.n_features;
  device const uchar *column = packed + (active ? ulong(columns[feature]) * p.packed_stride : 0ul);
  uint start = p.first_row + group.y * p.rows_per_group;  // absolute rows [start, stop)
  uint stop = p.first_row + min((group.y + 1) * p.rows_per_group, p.n);
  for (uint block = start & ~31u; block < stop; block += 32) {
    uint row = block + lane;
    bool inside = row >= start && row < stop;
    uint i = row - p.first_row;
    bool in_bag = inside && bag[i] != 0;
    bool any_row = false;
    int word0 = -1;  // entry 0's word stays in a register: a chunk of one entry (every single fit) needs no memory
    for (uint e = 0; e < n_entries; ++e) {
      Entry entry = chunk[e];
      int word = -1;
      if (in_bag) {
        ulong index = ulong(entry.model) * p.n + i;
        int slot = slots[entry.model * p.width + node[index]];
        if (slot >= int(entry.first_slot) && slot < int(entry.end_slot)) {
          word = int(entry.pair_base + uint(slot) - entry.first_slot) * 256 + int(q[index]) + 128;
        }
      }
      if (e == 0) {
        word0 = word;
      }
      if (n_entries > 1) {
        words[e * LANES + lane] = word;
      }
      any_row = any_row || word >= 0;
    }
    if (n_entries > 1) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!simd_any(any_row)) {
      continue;
    }
    uint4 codes = active ? *(device const uint4 *)(column + (block >> 1)) : uint4(0);
    if (p.popcount != 0) {
      uint gt[OBFUSCATED_BINS] = {0xFFFFFFFFu, above(codes, 0), above(codes, 1), above(codes, 2), above(codes, 3)};
      for (uint e = 0; e < n_entries; ++e) {
        int word = e == 0 ? word0 : words[e * LANES + lane];
        int value = (word & 255) - 128;
        uint magnitude = uint(abs(value));
        Entry entry = chunk[e];
        for (uint pair = entry.pair_base; pair < entry.pair_base + (entry.end_slot - entry.first_slot); ++pair) {
          bool here = word >= 0 && uint(word >> 8) == pair;
          uint rows = uint(ulong(simd_ballot(here)));
          if (rows == 0) {
            continue;
          }
          threadgroup int *counters = local + pair * (OBFUSCATED_BINS * 2) * LANES + lane;
          for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
            counters[(2 * t) * LANES] += int(popcount(gt[t] & rows));
          }
          for (uint b = 0; b < p.q_bits; ++b) {
            uint positive = uint(ulong(simd_ballot(here && value > 0 && ((magnitude >> b) & 1u) != 0)));
            uint negative = uint(ulong(simd_ballot(here && value < 0 && ((magnitude >> b) & 1u) != 0)));
            for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
              counters[(2 * t + 1) * LANES] += (int(popcount(gt[t] & positive)) - int(popcount(gt[t] & negative))) << b;
            }
          }
        }
      }
    } else if (n_entries == 1) {
#pragma unroll
      for (uint t = 0; t < 32; ++t) {
        int w = simd_shuffle(word0, ushort(t));
        if (w >= 0) {
          uint bin = (codes[t >> 3] >> ((t & 7u) * 4u)) & 15u;
          local[(uint(w >> 8) * OBFUSCATED_BINS + bin) * LANES + lane] += ((w & 255) - 128) * 65536 + 1;
        }
      }
    } else {
#pragma unroll
      for (uint t = 0; t < 32; ++t) {
        uint bin = (codes[t >> 3] >> ((t & 7u) * 4u)) & 15u;
        for (uint e = 0; e < n_entries; ++e) {
          int w = words[e * LANES + t];
          if (w >= 0) {
            local[(uint(w >> 8) * OBFUSCATED_BINS + bin) * LANES + lane] += ((w & 255) - 128) * 65536 + 1;
          }
        }
      }
    }
    if (n_entries > 1) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (!active) {
    return;
  }
  for (uint e = 0; e < n_entries; ++e) {
    Entry entry = chunk[e];
    for (uint s = entry.first_slot; s < entry.end_slot; ++s) {
      uint pair = entry.pair_base + s - entry.first_slot;
      for (uint b = 0; b < OBFUSCATED_BINS; ++b) {
        int count;
        int sum;
        if (p.popcount != 0) {
          threadgroup int *counters = local + pair * (OBFUSCATED_BINS * 2) * LANES + lane;
          int above_count = b + 1 < OBFUSCATED_BINS ? counters[(2 * (b + 1)) * LANES] : 0;
          int above_sum = b + 1 < OBFUSCATED_BINS ? counters[(2 * (b + 1) + 1) * LANES] : 0;
          count = counters[(2 * b) * LANES] - above_count;
          sum = counters[(2 * b + 1) * LANES] - above_sum;
        } else {
          int packed_counter = local[(pair * OBFUSCATED_BINS + b) * LANES + lane];
          count = packed_counter & 0xFFFF;
          sum = (packed_counter - count) >> 16;
        }
        if (count != 0) {
          ulong target = (ulong(slot_base[entry.model] + int(s)) * p.cells + uint(offsets[feature]) + b) * 2;
          atomic_fetch_add_explicit(hist + target, sum, memory_order_relaxed);
          atomic_fetch_add_explicit(hist + target + 1, count, memory_order_relaxed);
        }
      }
    }
  }
}

struct SingleObfuscatedParams {
  uint n;
  uint first_row;
  uint n_features;
  uint cells;
  uint packed_stride;
  uint rows_per_group;
  uint slots_per_chunk;
  uint n_slots;
  uint q_bits;    // bits of |q|: bit_length(half_bins)
  uint popcount;  // 1: bitmask path (few slots), 0: per-row path
};

// One model's histograms of obfuscated features (used for K = 1: it keeps no per-model words in threadgroup memory,
// and measured about 20 % faster for single fits than the K-model kernel).
// One feature per SIMD lane ("lane owns column").
// A threadgroup is one SIMD group; its 32 lanes take 32 sampled features and walk the same rows in 32-row blocks
// aligned on absolute rows. Per block, lane j loads row j's bag, node and q (coalesced) and turns them into one word
// (-1 = skip, else local slot * 256 + q + 128), and loads its own feature's 32 codes with one 16-byte load (two rows
// per byte). A block with no selected row is skipped (one vote). Two accumulation paths, chosen by the host:
//   popcount (few slots, levels 0-3): per slot, ballots give the block's rows of that slot and the bits of |q| by
//     sign; per lane, `above` gives the rows whose code is > t. Count and sum q of (slot, code > t), t = -1..3, are
//     popcounts, accumulated once per block. Bins follow by differences: bin b = (> b-1) - (> b).
//   per row (many slots): for each of the 32 rows (unrolled) the word is broadcast with simd_shuffle and each lane adds
//     q * 2^16 + 1 to its own (slot, bin) counter, sum q in the high and the count in the low 16 bits, exact while
//     rows_per_group * half_bins <= 32767 (the host guarantees it).
// Counters are lane-minor in threadgroup memory: no atomics, no bank conflicts. Lanes finally add their non-zero sums
// and counts to the device histogram (integer atomics, so the order does not matter). Grid: (feature groups, row
// groups, slot chunks); threadgroup memory sized by the host. Columns must be 16-byte aligned, with 16 bytes readable
// past the last row.
kernel void histograms_obfuscated_one(device const int *columns [[buffer(0)]], device const int *offsets [[buffer(1)]],
                                      device const uchar *packed [[buffer(2)]], device const uchar *bag [[buffer(3)]],
                                      device const uchar *node [[buffer(4)]], device const char *q [[buffer(5)]],
                                      device const int *slots [[buffer(6)]], device atomic_int *hist [[buffer(7)]],
                                      constant SingleObfuscatedParams &p [[buffer(8)]],
                                      threadgroup int *local [[threadgroup(0)]],
                                      uint3 group [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_threadgroup]]) {
  uint feature = group.x * LANES + lane;
  uint first_slot = group.z * p.slots_per_chunk;
  uint end_slot = min(first_slot + p.slots_per_chunk, p.n_slots);
  uint n_local = end_slot - first_slot;
  // popcount: (slot, t + 1 for t = -1..3, {count, sum}); per row: (slot, bin) packed
  uint span = p.popcount != 0 ? n_local * (OBFUSCATED_BINS * 2) : n_local * OBFUSCATED_BINS;
  for (uint k = 0; k < span; ++k) {
    local[k * LANES + lane] = 0;
  }
  bool active = feature < p.n_features;
  device const uchar *column = packed + (active ? ulong(columns[feature]) * p.packed_stride : 0ul);
  uint start = p.first_row + group.y * p.rows_per_group;  // absolute rows [start, stop)
  uint stop = p.first_row + min((group.y + 1) * p.rows_per_group, p.n);
  for (uint block = start & ~31u; block < stop; block += 32) {
    uint row = block + lane;
    int word = -1;
    if (row >= start && row < stop) {
      uint i = row - p.first_row;
      if (bag[i] != 0) {
        int slot = slots[node[i]];
        if (slot >= int(first_slot) && slot < int(end_slot)) {
          word = (slot - int(first_slot)) * 256 + int(q[i]) + 128;
        }
      }
    }
    if (!simd_any(word >= 0)) {
      continue;
    }
    uint4 codes = active ? *(device const uint4 *)(column + (block >> 1)) : uint4(0);
    if (p.popcount != 0) {
      uint gt[OBFUSCATED_BINS] = {0xFFFFFFFFu, above(codes, 0), above(codes, 1), above(codes, 2), above(codes, 3)};
      int value = (word & 255) - 128;
      uint magnitude = uint(abs(value));
      for (uint s = 0; s < n_local; ++s) {
        bool here = word >= 0 && uint(word >> 8) == s;
        uint rows = uint(ulong(simd_ballot(here)));
        if (rows == 0) {
          continue;
        }
        threadgroup int *counters = local + s * (OBFUSCATED_BINS * 2) * LANES + lane;
        for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
          counters[(2 * t) * LANES] += int(popcount(gt[t] & rows));
        }
        for (uint b = 0; b < p.q_bits; ++b) {
          uint positive = uint(ulong(simd_ballot(here && value > 0 && ((magnitude >> b) & 1u) != 0)));
          uint negative = uint(ulong(simd_ballot(here && value < 0 && ((magnitude >> b) & 1u) != 0)));
          for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
            counters[(2 * t + 1) * LANES] += (int(popcount(gt[t] & positive)) - int(popcount(gt[t] & negative))) << b;
          }
        }
      }
      continue;
    }
#pragma unroll
    for (uint t = 0; t < 32; ++t) {
      int w = simd_shuffle(word, ushort(t));
      if (w >= 0) {
        uint bin = (codes[t >> 3] >> ((t & 7u) * 4u)) & 15u;
        local[(uint(w >> 8) * OBFUSCATED_BINS + bin) * LANES + lane] += ((w & 255) - 128) * 65536 + 1;
      }
    }
  }
  if (!active) {
    return;
  }
  for (uint s = 0; s < n_local; ++s) {
    for (uint b = 0; b < OBFUSCATED_BINS; ++b) {
      int count;
      int sum;
      if (p.popcount != 0) {
        threadgroup int *counters = local + s * (OBFUSCATED_BINS * 2) * LANES + lane;
        int above_count = b + 1 < OBFUSCATED_BINS ? counters[(2 * (b + 1)) * LANES] : 0;
        int above_sum = b + 1 < OBFUSCATED_BINS ? counters[(2 * (b + 1) + 1) * LANES] : 0;
        count = counters[(2 * b) * LANES] - above_count;
        sum = counters[(2 * b + 1) * LANES] - above_sum;
      } else {
        int packed_counter = local[(s * OBFUSCATED_BINS + b) * LANES + lane];
        count = packed_counter & 0xFFFF;
        sum = (packed_counter - count) >> 16;
      }
      if (count != 0) {
        ulong target = (ulong(first_slot + s) * p.cells + uint(offsets[feature]) + b) * 2;
        atomic_fetch_add_explicit(hist + target, sum, memory_order_relaxed);
        atomic_fetch_add_explicit(hist + target + 1, count, memory_order_relaxed);
      }
    }
  }
}

struct PartitionParams {
  uint n;
  uint first_row;
  uint named_stride;
  uint packed_stride;
  uint width;  // nodes per model on this level
  uint cat_stride;
};

// node = 2 * node + go_right for every row of every model, under that model's level splits.
kernel void partition_rows(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                           device const uchar *split_kind [[buffer(2)]], device const int *split_feature [[buffer(3)]],
                           device const uchar *threshold [[buffer(4)]], device const uchar *missing_left [[buffer(5)]],
                           device const ulong *cat_left [[buffer(6)]], device const uchar *named [[buffer(7)]],
                           device const uchar *packed [[buffer(8)]], device const uchar *country [[buffer(9)]],
                           device uchar *node [[buffer(10)]], constant PartitionParams &p [[buffer(11)]],
                           uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) {
    return;
  }
  ulong index = ulong(at.y) * p.n + at.x;
  uint here = at.y * p.width + node[index];
  int f = split_feature[here];
  uint bin = bin_of(kinds[f], columns[f], p.first_row + at.x, named, p.named_stride, packed, p.packed_stride, country,
                    p.cat_stride);
  node[index] = uchar(2 * uint(node[index]) +
                      goes_right(split_kind[here], bin, threshold[here], missing_left[here], cat_left[here]));
}

// Multi-target own last level: node_own[t * n + i] = 2 * node[i] + go_right under target t's own splits of the
// deepest level; the shared node array is left as it is. Grid: (rows, targets).
kernel void partition_own(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                          device const uchar *split_kind [[buffer(2)]], device const int *split_feature [[buffer(3)]],
                          device const uchar *threshold [[buffer(4)]], device const uchar *missing_left [[buffer(5)]],
                          device const ulong *cat_left [[buffer(6)]], device const uchar *named [[buffer(7)]],
                          device const uchar *packed [[buffer(8)]], device const uchar *country [[buffer(9)]],
                          device const uchar *node [[buffer(10)]], device uchar *node_own [[buffer(11)]],
                          constant PartitionParams &p [[buffer(12)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) {
    return;
  }
  uint here = at.y * p.width + node[at.x];
  int f = split_feature[here];
  uint bin = bin_of(kinds[f], columns[f], p.first_row + at.x, named, p.named_stride, packed, p.packed_stride, country,
                    p.cat_stride);
  node_own[ulong(at.y) * p.n + at.x] = uchar(
      2 * uint(node[at.x]) + goes_right(split_kind[here], bin, threshold[here], missing_left[here], cat_left[here]));
}

struct LeafSplitParams {
  uint n;
  uint first_row;
  uint named_stride;
  uint packed_stride;
  uint cat_stride;
};

// Leaf-wise growth: node holds each row's leaf id. Per model, the rows of leaf leaves[2 * model] that go right
// under the model's split (entry `model` of the split arrays) move to leaf leaves[2 * model + 1]; every other
// row keeps its leaf. A model that does not split this step has leaf 256, which no row holds. Grid: (rows, models).
kernel void split_leaf(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                       device const uchar *split_kind [[buffer(2)]], device const int *split_feature [[buffer(3)]],
                       device const uchar *threshold [[buffer(4)]], device const uchar *missing_left [[buffer(5)]],
                       device const ulong *cat_left [[buffer(6)]], device const uint *leaves [[buffer(7)]],
                       device const uchar *named [[buffer(8)]], device const uchar *packed [[buffer(9)]],
                       device const uchar *country [[buffer(10)]], device uchar *node [[buffer(11)]],
                       constant LeafSplitParams &p [[buffer(12)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) {
    return;
  }
  ulong index = ulong(at.y) * p.n + at.x;
  if (uint(node[index]) != leaves[2 * at.y]) {
    return;
  }
  int f = split_feature[at.y];
  uint bin = bin_of(kinds[f], columns[f], p.first_row + at.x, named, p.named_stride, packed, p.packed_stride, country,
                    p.cat_stride);
  if (goes_right(split_kind[at.y], bin, threshold[at.y], missing_left[at.y], cat_left[at.y]) != 0) {
    node[index] = uchar(leaves[2 * at.y + 1]);
  }
}

struct LeafParams {
  uint n;
  uint n_leaves;
  uint rows_per_group;
  uint groups;
  uint node_stride;  // n: a node array per model; 0: one node array shared by every model (multi-target)
  uint residual;     // 1: the value of a row is a[i] - b[i] (multi-target: pred, y); 0: a[i] itself (g, or h)
};

// Exact per-leaf sums of rint(value * 2^32) over in-bag rows, per model and threadgroup chunk of rows; a row's value
// is its gradient (or its hessian) as stored, or pred - y (`residual`). Metal has no 64-bit atomic add, so each value
// is split as hi * 2^32 + mid * 2^16 + low with int hi and 16-bit mid and low. The host keeps max |value| * rows below
// 2^31, so with rows_per_group <= 4096 no 32-bit partial can overflow (|sum hi| < 2^31, sum mid, sum low < 2^28). The
// host adds the chunks' int64 partials. Grid: (row chunks, models).
kernel void leaf_partials(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                          device const uchar *bag [[buffer(2)]], device const uchar *node [[buffer(3)]],
                          device uint *sums [[buffer(4)]], device uint *counts [[buffer(5)]],
                          constant LeafParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                          uint2 position [[thread_position_in_threadgroup]], uint2 size [[threads_per_threadgroup]]) {
  threadgroup atomic_int hi[MAX_LEAVES];
  threadgroup atomic_uint mid[MAX_LEAVES];
  threadgroup atomic_uint low[MAX_LEAVES];
  threadgroup atomic_uint count[MAX_LEAVES];
  uint tid = position.x;
  uint width = size.x;
  for (uint k = tid; k < p.n_leaves; k += width) {
    atomic_store_explicit(&hi[k], 0, memory_order_relaxed);
    atomic_store_explicit(&mid[k], 0u, memory_order_relaxed);
    atomic_store_explicit(&low[k], 0u, memory_order_relaxed);
    atomic_store_explicit(&count[k], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint start = group.x * p.rows_per_group;
  uint stop = min(start + p.rows_per_group, p.n);
  ulong model = ulong(group.y) * p.n;
  ulong nodes = ulong(group.y) * p.node_stride;
  for (uint i = start + tid; i < stop; i += width) {
    if (bag[i] == 0) {
      continue;
    }
    float value = p.residual != 0 ? a[model + i] - b[model + i] : a[model + i];
    long fixed = long(rint(value * 0x1p32f));
    uint leaf = node[nodes + i];
    uint bits = uint(fixed & 0xFFFFFFFFl);
    atomic_fetch_add_explicit(&hi[leaf], int(fixed >> 32), memory_order_relaxed);
    atomic_fetch_add_explicit(&mid[leaf], bits >> 16, memory_order_relaxed);
    atomic_fetch_add_explicit(&low[leaf], bits & 0xFFFFu, memory_order_relaxed);
    atomic_fetch_add_explicit(&count[leaf], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong out = (ulong(group.y) * p.groups + group.x) * p.n_leaves;
  for (uint k = tid; k < p.n_leaves; k += width) {
    // hi (as uint), mid, low: the host adds them in int64 (Apple7's shader compiler crashes on a 64-bit store here)
    sums[3 * (out + k)] = uint(atomic_load_explicit(&hi[k], memory_order_relaxed));
    sums[3 * (out + k) + 1] = atomic_load_explicit(&mid[k], memory_order_relaxed);
    sums[3 * (out + k) + 2] = atomic_load_explicit(&low[k], memory_order_relaxed);
    counts[out + k] = atomic_load_explicit(&count[k], memory_order_relaxed);
  }
}

struct UpdateParams {
  uint n;
  uint n_leaves;
  uint node_stride;  // as LeafParams
};

// pred += leaf_values[model, node]: one float32 addition per row and model.
kernel void update_predictions(device float *pred [[buffer(0)]], device const uchar *node [[buffer(1)]],
                               device const float *leaf_values [[buffer(2)]], constant UpdateParams &p [[buffer(3)]],
                               uint2 at [[thread_position_in_grid]]) {
  if (at.x < p.n) {
    ulong index = ulong(at.y) * p.n + at.x;
    pred[index] = pred[index] + leaf_values[at.y * p.n_leaves + node[ulong(at.y) * p.node_stride + at.x]];
  }
}

struct PredictParams {
  uint n;
  uint first_row;
  uint n_trees;
  uint depth;
  uint named_stride;
  uint packed_stride;
  uint cat_stride;
};

// Forest output per row: heap walk per tree, leaves added in tree order from 0.0.
kernel void predict_trees(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                          device const uchar *kind [[buffer(2)]], device const int *feature [[buffer(3)]],
                          device const uchar *threshold [[buffer(4)]], device const uchar *missing_left [[buffer(5)]],
                          device const ulong *cat_left [[buffer(6)]], device const float *leaf_value [[buffer(7)]],
                          device const uchar *named [[buffer(8)]], device const uchar *packed [[buffer(9)]],
                          device const uchar *country [[buffer(10)]], device float *out [[buffer(11)]],
                          constant PredictParams &p [[buffer(12)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) {
    return;
  }
  uint internal = (1u << p.depth) - 1u;
  uint row = p.first_row + i;
  float acc = 0.0f;
  for (uint t = 0; t < p.n_trees; ++t) {
    ulong base = ulong(t) * internal;
    uint h = 0;
    for (uint d = 0; d < p.depth; ++d) {
      int f = feature[base + h];
      uint bin =
          bin_of(kinds[f], columns[f], row, named, p.named_stride, packed, p.packed_stride, country, p.cat_stride);
      h = 2 * h + 1 + goes_right(kind[base + h], bin, threshold[base + h], missing_left[base + h], cat_left[base + h]);
    }
    acc = acc + leaf_value[ulong(t) * (internal + 1u) + (h - internal)];
  }
  out[i] = acc;
}

struct NodePredictParams {
  uint n;
  uint first_row;
  uint n_trees;
  uint max_leaves;
  uint named_stride;
  uint packed_stride;
  uint cat_stride;
};

// Forest output per row for leaf-wise trees (node lists, idun.gbm.tree.NodeForest): from node 0, follow left or
// right until a child is a leaf (< 0, leaf ~child); every child node comes after its parent, so a walk ends within
// max_leaves - 1 steps. Leaves added in tree order from 0.0, as predict_trees.
kernel void predict_nodes(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                          device const uchar *kind [[buffer(2)]], device const int *feature [[buffer(3)]],
                          device const uchar *threshold [[buffer(4)]], device const uchar *missing_left [[buffer(5)]],
                          device const ulong *cat_left [[buffer(6)]], device const short *left [[buffer(7)]],
                          device const short *right [[buffer(8)]], device const float *leaf_value [[buffer(9)]],
                          device const uchar *named [[buffer(10)]], device const uchar *packed [[buffer(11)]],
                          device const uchar *country [[buffer(12)]], device float *out [[buffer(13)]],
                          constant NodePredictParams &p [[buffer(14)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.n) {
    return;
  }
  uint internal = p.max_leaves - 1u;
  uint row = p.first_row + i;
  float acc = 0.0f;
  for (uint t = 0; t < p.n_trees; ++t) {
    ulong base = ulong(t) * internal;
    int h = 0;
    int child = -1;
    for (uint step = 0; step < internal; ++step) {
      ulong at = base + uint(h);
      int f = feature[at];
      uint bin =
          bin_of(kinds[f], columns[f], row, named, p.named_stride, packed, p.packed_stride, country, p.cat_stride);
      child = goes_right(kind[at], bin, threshold[at], missing_left[at], cat_left[at]) != 0 ? int(right[at])
                                                                                            : int(left[at]);
      if (child < 0) {
        break;
      }
      h = child;
    }
    acc = acc + leaf_value[ulong(t) * p.max_leaves + uint(~child)];
  }
  out[i] = acc;
}

struct MultiParams {
  uint n;
  uint first_row;
  uint n_features;
  uint cells;
  uint named_stride;
  uint packed_stride;
  uint rows_per_group;
  uint k;  // targets
  uint half_bins;
  uint slots_per_chunk;
  uint n_slots;
  uint targets_per_chunk;  // histograms_multi: at most 15 (with the count, four words of 16-bit fields)
  uint target_chunks;
  uint cat_stride;
};

// Multi-target histograms of named features and the country (experimental): one thread per row, one node array for
// every target. hist is (slots, cells, K + 1): the K targets' sums of q, then the count. Device atomics.
kernel void build_histograms_multi(device const int *kinds [[buffer(0)]], device const int *columns [[buffer(1)]],
                                   device const int *offsets [[buffer(2)]], device const uchar *named [[buffer(3)]],
                                   device const uchar *packed [[buffer(4)]], device const uchar *country [[buffer(5)]],
                                   device const uchar *bag [[buffer(6)]], device const uchar *node [[buffer(7)]],
                                   device const char *q [[buffer(8)]], device const int *slots [[buffer(9)]],
                                   device atomic_int *hist [[buffer(10)]], constant MultiParams &p [[buffer(11)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i >= p.n || bag[i] == 0) {
    return;
  }
  int slot = slots[node[i]];
  if (slot < 0) {
    return;
  }
  uint row = p.first_row + i;
  for (uint j = 0; j < p.n_features; ++j) {
    uint cell = uint(offsets[j]) + bin_of(kinds[j], columns[j], row, named, p.named_stride, packed, p.packed_stride,
                                          country, p.cat_stride);
    device atomic_int *base = hist + (ulong(slot) * p.cells + cell) * (p.k + 1);
    for (uint t = 0; t < p.k; ++t) {
      atomic_fetch_add_explicit(base + t, int(q[ulong(t) * p.n + i]), memory_order_relaxed);
    }
    atomic_fetch_add_explicit(base + p.k, 1, memory_order_relaxed);
  }
}

// Per training row, its slot at this level for histograms_multi: -1 out of the bag or in a skipped node (slots < 128).
kernel void row_slots(device const uchar *bag [[buffer(0)]], device const uchar *node [[buffer(1)]],
                      device const int *slots [[buffer(2)]], device char *row_slot [[buffer(3)]],
                      constant CountParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) {
    row_slot[i] = bag[i] != 0 ? char(slots[node[i]]) : char(-1);
  }
}

// Multi-target histograms of obfuscated features (experimental), one feature per SIMD lane over 32-row blocks as
// histograms_obfuscated, with one node array for every target. A threadgroup takes one chunk of slots and one chunk of
// at most 15 targets [t_lo, t_lo + kc): per (slot, bin) a lane keeps kc + 1 16-bit fields packed four to a 64-bit word;
// field t < kc adds q + half_bins of target t_lo + t (never negative), field kc counts the row; field f sits in bits
// 16 (f % 4) .. of word f / 4. The lane owning a row packs that row's words once; the 32 lanes receive the words they
// need by simd_shuffle, so one 64-bit add updates four fields. Exact while rows_per_group * 2 * half_bins <= 65535 (the
// host guarantees it); a sum of q is its field minus half_bins * count. Every target chunk unbiases with its own count;
// the first writes the count. Grid: (slot chunks x target chunks, target chunk fastest, so the chunks of a block share
// its codes in the cache; feature groups; row groups). Threadgroup memory: slots_per_chunk * 5 * words * 32 ulongs.
kernel void histograms_multi(device const int *columns [[buffer(0)]], device const int *offsets [[buffer(1)]],
                             device const uchar *packed [[buffer(2)]], device const char *row_slot [[buffer(3)]],
                             device const char *q [[buffer(4)]], device atomic_int *hist [[buffer(5)]],
                             constant MultiParams &p [[buffer(6)]], threadgroup ulong *local [[threadgroup(0)]],
                             uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  uint target_chunk = group.x % p.target_chunks;
  uint t_lo = target_chunk * p.targets_per_chunk;
  uint kc = min(p.targets_per_chunk, p.k - t_lo);
  uint words = (kc + 4) / 4;
  uint feature = group.y * LANES + lane;
  uint first_slot = (group.x / p.target_chunks) * p.slots_per_chunk;
  uint end_slot = min(first_slot + p.slots_per_chunk, p.n_slots);
  uint span = (end_slot - first_slot) * OBFUSCATED_BINS * words;
  for (uint c = 0; c < span; ++c) {
    local[c * LANES + lane] = 0;
  }
  bool active = feature < p.n_features;
  device const uchar *column = packed + (active ? ulong(columns[feature]) * p.packed_stride : 0ul);
  uint start = p.first_row + group.z * p.rows_per_group;
  uint stop = p.first_row + min((group.z + 1) * p.rows_per_group, p.n);
  for (uint block = start & ~31u; block < stop; block += 32) {
    uint row = block + lane;
    int slot = -1;
    ulong w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    if (row >= start && row < stop) {
      uint i = row - p.first_row;
      {
        int s = row_slot[i];
        if (s >= int(first_slot) && s < int(end_slot)) {
          slot = s - int(first_slot);
          for (uint t = 0; t < kc; ++t) {
            ulong v = ulong(int(q[ulong(t_lo + t) * p.n + i]) + int(p.half_bins)) << (16 * (t % 4));
            uint which = t / 4;
            w0 += which == 0 ? v : 0;
            w1 += which == 1 ? v : 0;
            w2 += which == 2 ? v : 0;
            w3 += which == 3 ? v : 0;
          }
          ulong one = 1ul << (16 * (kc % 4));
          uint which = kc / 4;
          w0 += which == 0 ? one : 0;
          w1 += which == 1 ? one : 0;
          w2 += which == 2 ? one : 0;
          w3 += which == 3 ? one : 0;
        }
      }
    }
    uint present = uint(ulong(simd_ballot(slot >= 0)));  // the block's rows of this chunk: a uniform loop over them
    if (present == 0) {
      continue;
    }
    uint4 codes = active ? *(device const uint4 *)(column + (block >> 1)) : uint4(0);
    while (present != 0) {
      uint t = ctz(present);
      present &= present - 1;
      int s = simd_shuffle(slot, ushort(t));
      uint bin = (codes[t >> 3] >> ((t & 7u) * 4u)) & 15u;
      threadgroup ulong *base = local + ((uint(s) * OBFUSCATED_BINS + bin) * words) * LANES + lane;
      base[0] += as_type<ulong>(simd_shuffle(as_type<uint2>(w0), ushort(t)));  // inactive lanes write their own unused
      if (words > 1) {  // counters (no divergence around shuffles)
        base[LANES] += as_type<ulong>(simd_shuffle(as_type<uint2>(w1), ushort(t)));
      }
      if (words > 2) {
        base[2 * LANES] += as_type<ulong>(simd_shuffle(as_type<uint2>(w2), ushort(t)));
      }
      if (words > 3) {
        base[3 * LANES] += as_type<ulong>(simd_shuffle(as_type<uint2>(w3), ushort(t)));
      }
    }
  }
  if (!active) {
    return;
  }
  for (uint s = 0; s < end_slot - first_slot; ++s) {
    for (uint b = 0; b < OBFUSCATED_BINS; ++b) {
      threadgroup ulong *base = local + ((s * OBFUSCATED_BINS + b) * words) * LANES + lane;
      int count = int((base[(kc / 4) * LANES] >> (16 * (kc % 4))) & 0xFFFFul);
      if (count == 0) {
        continue;
      }
      device atomic_int *target = hist + (ulong(first_slot + s) * p.cells + uint(offsets[feature]) + b) * (p.k + 1);
      for (uint t = 0; t < kc; ++t) {
        int field = int((base[(t / 4) * LANES] >> (16 * (t % 4))) & 0xFFFFul);
        atomic_fetch_add_explicit(target + t_lo + t, field - int(p.half_bins) * count, memory_order_relaxed);
      }
      if (target_chunk == 0) {
        atomic_fetch_add_explicit(target + p.k, count, memory_order_relaxed);
      }
    }
  }
}
