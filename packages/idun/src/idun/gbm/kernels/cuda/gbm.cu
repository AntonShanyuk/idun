// gbm.cu - idun_gbm kernels for the CUDA backend: the CUDA C++ twins of kernels/metal/gbm.metal.
//
// Each kernel mirrors the Metal kernel of the same name operation for operation, and through it the numba kernel of
// idun/gbm/backends/cpu.py, so the backends produce identical bits (docs/architecture/determinism.md). Rules: integer
// atomics only; floats only in per-row `+`, `*` and `rint` (the runtime compiles with --fmad=false: a product and a
// sum round separately); draws from the row's absolute index. A kernel's arguments are its buffers in the order of
// the kernel table in idun/gbm/backends/device.py (`const` for the ones it only reads), then its params struct by
// value, mirrored by a ctypes.Structure there. Kernels are `extern "C"`: the module exports their plain names.
//
// Metal's threadgroup is a block here, its threadgroup memory is shared memory (the host sizes the dynamic part
// per launch), and its SIMD group is a warp: 32 lanes on both. The lane kernels run one warp per block; every
// vote and shuffle names all 32 lanes, so no lane leaves a kernel before its last one.

typedef unsigned int uint;
typedef unsigned char uchar;
typedef signed char schar;  // plain `char` is unsigned on some hosts: q and the row slots are signed bytes
typedef unsigned long long ulong;

static const uint NAMED = 0;
static const uint CATEGORICAL = 2;
static const uint MISSING_BIN = 255;
static const uint MAX_LEAVES = 256;
static const uint LANES = 32;
static const uint OBFUSCATED_BINS = 5;
static const uint MAX_ENTRIES = 16;    // entries per chunk: at most one per (model, slot) pair
static const uint WARP = 0xFFFFFFFFu;  // every lane of a warp

// A block's dynamic shared memory, sized by the host per launch: 32-bit counters, or 64-bit ones (histograms_multi).
extern __shared__ int local[];
extern __shared__ ulong local_wide[];

static __device__ __forceinline__ uint smaller(uint a, uint b) { return a < b ? a : b; }

// lowbias32 (Chris Wellons, public domain): idun.rng.
static __device__ __forceinline__ uint lowbias32(uint x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

static __device__ __forceinline__ uint row_bits(uint key, uint row) { return lowbias32(key ^ row) >> 8; }

// The largest value among a warp's 32 lanes (every lane gets it).
static __device__ __forceinline__ uint warp_max(uint x) {
  for (uint step = 16; step > 0; step >>= 1) {
    uint other = __shfl_xor_sync(WARP, x, step);
    x = other > x ? other : x;
  }
  return x;
}

// Bin of panel row `row` in one feature: named bins and categorical codes are bytes, obfuscated codes are nibbles.
// Categorical column c starts at c * cat_stride (0 for a single column).
static __device__ __forceinline__ uint bin_of(int kind, int column, uint row, const uchar *named, uint named_stride,
                                              const uchar *packed, uint packed_stride, const uchar *country,
                                              uint cat_stride) {
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
static __device__ __forceinline__ uint goes_right(uint kind, uint bin, uint threshold, uint missing_left,
                                                  ulong cat_left) {
  if (kind == 0) {
    return 0;
  }
  if (kind == 2) {
    // The shift takes the low six bits of the bin, as Metal's and the CPU's do.
    return ((cat_left >> (ulong(bin) & 63ull)) & 1ull) != 0 ? 0 : 1;
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
extern "C" __global__ void rng_uniform_bits(uint *out, RngParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.n) {
    out[i] = row_bits(p.key, p.first_row + i);
  }
}

// Test kernel: `a * b + c` must round twice, as a separate multiply and add.
extern "C" __global__ void fma_probe(const float *a, const float *b, const float *c, float *out, CountParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.n) {
    out[i] = a[i] * b[i] + c[i];
  }
}

// Every per-row array of a fit holds K models back to back: element (model, i) is at model * n + i, and kernels
// that touch them run on a (rows, models) grid: row = blockIdx.x * blockDim.x + threadIdx.x, model = blockIdx.y.
// The bag and the feature columns are shared by the K models (the models of one seed share the bag and the feature
// sample). K = 1 is a single fit.

struct ModelParams {
  uint n;
  uint k;
};

// g = pred - y per model; max |g| per model into max_bits[model] (the bits of a non-negative float order like it).
extern "C" __global__ void compute_gradients(const float *pred, const float *y, float *g, uint *max_bits,
                                             ModelParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  float value = 0.0f;
  ulong index = ulong(model) * p.n + i;
  if (i < p.n) {
    value = pred[index] - y[index];
    g[index] = value;
  }
  uint largest = warp_max(__float_as_uint(fabsf(value)));
  if (threadIdx.x % 32 == 0) {
    atomicMax(max_bits + model, largest);
  }
}

struct QuantizeParams {
  uint n;
  uint first_row;
  uint key;
  int half_bins;
};

// Stochastic rounding toward zero, clamped (determinism.md §2), of one in-bag row's gradient.
static __device__ __forceinline__ schar round_gradient(float g, float inv_scale, const QuantizeParams &p, uint i) {
  float u = float(row_bits(p.key, p.first_row + i)) * 0x1p-24f;
  float x = g * inv_scale;
  float v = g >= 0.0f ? x + u : x - u;
  int rounded = int(v);
  return schar(rounded < -p.half_bins ? -p.half_bins : rounded > p.half_bins ? p.half_bins : rounded);
}

// q = the rounded g; 0 for rows out of the bag. One key for every model.
extern "C" __global__ void quantize_gradients(const float *g, const uchar *bag, schar *q, const float *inv_scale,
                                              QuantizeParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i >= p.n) {
    return;
  }
  ulong index = ulong(model) * p.n + i;
  q[index] = bag[i] == 0 ? schar(0) : round_gradient(g[index], inv_scale[model], p, i);
}

// Multi-target (no g buffer): max |pred - y| per target, as compute_gradients without storing g.
extern "C" __global__ void max_residuals(const float *pred, const float *y, uint *max_bits, ModelParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  float value = 0.0f;
  ulong index = ulong(model) * p.n + i;
  if (i < p.n) {
    value = pred[index] - y[index];
  }
  uint largest = warp_max(__float_as_uint(fabsf(value)));
  if (threadIdx.x % 32 == 0) {
    atomicMax(max_bits + model, largest);
  }
}

// Multi-target (no g buffer): quantize_gradients of g = pred - y, the same float32 subtraction as compute_gradients.
extern "C" __global__ void quantize_residuals(const float *pred, const float *y, const uchar *bag, schar *q,
                                              const float *inv_scale, QuantizeParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i >= p.n) {
    return;
  }
  ulong index = ulong(model) * p.n + i;
  q[index] = bag[i] == 0 ? schar(0) : round_gradient(pred[index] - y[index], inv_scale[model], p, i);
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
extern "C" __global__ void build_histograms(const int *kinds, const int *columns, const int *offsets,
                                            const uchar *named, const uchar *packed, const uchar *country,
                                            const uchar *bag, const uchar *node, const schar *q, const int *slots,
                                            const int *slot_base, int *hist, HistogramParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i >= p.n || bag[i] == 0) {
    return;
  }
  ulong index = ulong(model) * p.n + i;
  int slot = slots[model * p.width + node[index]];
  if (slot < 0) {
    return;
  }
  uint row = p.first_row + i;
  int qi = int(q[index]);
  int *base = hist + ulong(slot_base[model] + slot) * p.cells * 2;
  for (uint j = 0; j < p.n_features; ++j) {
    uint cell = uint(offsets[j]) + bin_of(kinds[j], columns[j], row, named, p.named_stride, packed, p.packed_stride,
                                          country, p.cat_stride);
    atomicAdd(base + 2 * cell, qi);
    atomicAdd(base + 2 * cell + 1, 1);
  }
}

// build_histograms' sums, a block's in shared memory first: block (group, feature, model * chunks + chunk) counts the
// rows of one row group whose slot is in the chunk, for one named or categorical feature of `widths[feature]` cells,
// with integer atomics in shared memory, then adds its cells that hold a row to the histogram. Integer sums: the same
// in any order.
extern "C" __global__ void build_histograms_shared(const int *kinds, const int *columns, const int *offsets,
                                                   const int *widths, const uchar *named, const uchar *packed,
                                                   const uchar *country, const uchar *bag, const uchar *node,
                                                   const schar *q, const int *slots, const int *slot_base, int *hist,
                                                   SharedHistogramParams p) {
  uint j = blockIdx.y;
  uint model = blockIdx.z / p.chunks;
  int first_slot = int((blockIdx.z % p.chunks) * p.slots_per_chunk);
  uint bins = uint(widths[j]);
  uint span = p.slots_per_chunk * bins * 2;  // (slot, bin, {sum, count})
  for (uint k = threadIdx.x; k < span; k += blockDim.x) {
    local[k] = 0;
  }
  __syncthreads();
  uint stop = smaller((blockIdx.x + 1) * p.rows_per_group, p.n);
  for (uint i = blockIdx.x * p.rows_per_group + threadIdx.x; i < stop; i += blockDim.x) {
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
    atomicAdd(local + cell, int(q[index]));
    atomicAdd(local + cell + 1, 1);
  }
  __syncthreads();
  int *base = hist + (ulong(slot_base[model]) + ulong(first_slot)) * p.cells * 2 + ulong(offsets[j]) * 2;
  for (uint k = threadIdx.x; k < span / 2; k += blockDim.x) {
    int count = local[2 * k + 1];
    if (count != 0) {
      int *target = base + (ulong(k / bins) * p.cells + k % bins) * 2;
      atomicAdd(target, local[2 * k]);
      atomicAdd(target + 1, count);
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
  uint max_pairs;  // (model, slot) pairs per chunk at most: sizes the shared memory
};

// One entry of a chunk: model `model`'s slots [first_slot, end_slot), stored from local pair `pair_base` on.
struct Entry {
  uint model;
  uint first_slot;
  uint end_slot;
  uint pair_base;
};

// The codes of a 32-row block of one column: 16 bytes of nibbles (two rows per byte) from a 16-byte aligned address.
struct Codes {
  uint word[4];
};

static __device__ __forceinline__ Codes load_codes(const uchar *column, uint block, bool active) {
  Codes codes = {{0, 0, 0, 0}};
  if (active) {
    const uint *words = (const uint *)(column + (block >> 1));
    codes.word[0] = words[0];
    codes.word[1] = words[1];
    codes.word[2] = words[2];
    codes.word[3] = words[3];
  }
  return codes;
}

// Row t's code in a block.
static __device__ __forceinline__ uint code_of(const Codes &codes, uint t) {
  return (codes.word[t >> 3] >> ((t & 7u) * 4u)) & 15u;
}

// Rows of a 32-row block whose code is > t, t = 0..3: bit j = row j.
static __device__ __forceinline__ uint above(const Codes &codes, uint t) {
  uint mask = 0;
  for (uint w = 0; w < 4; ++w) {
    uint x = codes.word[w];
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

// Histograms of obfuscated features for K models, one feature per lane ("lane owns column"). A block is one warp:
// its 32 lanes take 32 sampled features and walk the same rows in 32-row blocks aligned on absolute rows, for every
// (model, slot) pair of one chunk (`entries` [chunk_start[c], chunk_start[c + 1])). Per row block, lane j writes row
// j's word for each entry to shared memory (-1 = skip, else local pair * 256 + q + 128) and loads its feature's 32
// codes once with a 16-byte load: the codes, and on the bitmask path the "code > t" masks, are shared by every model
// of the chunk. Paths:
//   popcount: per pair, ballots give its rows and the bits of |q| by sign; count and sum q of (pair, code > t),
//     t = -1..3, are popcounts accumulated once per row block; bins follow by differences.
//   per row: for each row (unrolled) and entry, the word is read as a broadcast and each lane adds q * 2^16 + 1 to
//     its (pair, bin) counter: sum q high, count low, exact while rows_per_group * half_bins <= 32767.
// Counters are lane-minor in shared memory (no atomics, no bank conflicts); lanes finally add their non-zero sums and
// counts to the device histogram with integer atomics. Grid: (feature groups, row groups, chunks). Shared memory:
// max_pairs * (10 or 5) * 32 counters, then max_pairs * 32 words. Columns must be 16-byte aligned, with 16 bytes
// readable past the last row.
extern "C" __global__ void histograms_obfuscated(const int *columns, const int *offsets, const uchar *packed,
                                                 const uchar *bag, const uchar *node, const schar *q, const int *slots,
                                                 const int *slot_base, const Entry *entries, const uint *chunk_start,
                                                 int *hist, ObfuscatedParams p) {
  __shared__ Entry chunk[MAX_ENTRIES];
  uint lane = threadIdx.x;
  uint feature = blockIdx.x * LANES + lane;
  uint first_entry = chunk_start[blockIdx.z];
  uint n_entries = chunk_start[blockIdx.z + 1] - first_entry;
  if (lane < n_entries) {
    chunk[lane] = entries[first_entry + lane];
  }
  __syncwarp();
  Entry last = chunk[n_entries - 1];
  uint n_pairs = last.pair_base + (last.end_slot - last.first_slot);
  uint per_pair = p.popcount != 0 ? OBFUSCATED_BINS * 2 : OBFUSCATED_BINS;
  int *words = local + p.max_pairs * per_pair * LANES;
  for (uint k = 0; k < n_pairs * per_pair; ++k) {
    local[k * LANES + lane] = 0;
  }
  bool active = feature < p.n_features;
  const uchar *column = packed + (active ? ulong(columns[feature]) * p.packed_stride : 0ull);
  uint start = p.first_row + blockIdx.y * p.rows_per_group;  // absolute rows [start, stop)
  uint stop = p.first_row + smaller((blockIdx.y + 1) * p.rows_per_group, p.n);
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
      __syncwarp();
    }
    if (__ballot_sync(WARP, any_row) == 0) {
      continue;
    }
    Codes codes = load_codes(column, block, active);
    if (p.popcount != 0) {
      uint gt[OBFUSCATED_BINS] = {0xFFFFFFFFu, above(codes, 0), above(codes, 1), above(codes, 2), above(codes, 3)};
      for (uint e = 0; e < n_entries; ++e) {
        int word = e == 0 ? word0 : words[e * LANES + lane];
        int value = (word & 255) - 128;
        uint magnitude = uint(value < 0 ? -value : value);
        Entry entry = chunk[e];
        for (uint pair = entry.pair_base; pair < entry.pair_base + (entry.end_slot - entry.first_slot); ++pair) {
          bool here = word >= 0 && uint(word >> 8) == pair;
          uint rows = __ballot_sync(WARP, here);
          if (rows == 0) {
            continue;
          }
          int *counters = local + pair * (OBFUSCATED_BINS * 2) * LANES + lane;
          for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
            counters[(2 * t) * LANES] += __popc(gt[t] & rows);
          }
          for (uint b = 0; b < p.q_bits; ++b) {
            uint positive = __ballot_sync(WARP, here && value > 0 && ((magnitude >> b) & 1u) != 0);
            uint negative = __ballot_sync(WARP, here && value < 0 && ((magnitude >> b) & 1u) != 0);
            for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
              counters[(2 * t + 1) * LANES] += (__popc(gt[t] & positive) - __popc(gt[t] & negative)) * (1 << b);
            }
          }
        }
      }
    } else if (n_entries == 1) {
#pragma unroll
      for (uint t = 0; t < 32; ++t) {
        int w = __shfl_sync(WARP, word0, t);
        if (w >= 0) {
          local[(uint(w >> 8) * OBFUSCATED_BINS + code_of(codes, t)) * LANES + lane] += ((w & 255) - 128) * 65536 + 1;
        }
      }
    } else {
#pragma unroll
      for (uint t = 0; t < 32; ++t) {
        uint bin = code_of(codes, t);
        for (uint e = 0; e < n_entries; ++e) {
          int w = words[e * LANES + t];
          if (w >= 0) {
            local[(uint(w >> 8) * OBFUSCATED_BINS + bin) * LANES + lane] += ((w & 255) - 128) * 65536 + 1;
          }
        }
      }
    }
    if (n_entries > 1) {
      __syncwarp();
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
          int *counters = local + pair * (OBFUSCATED_BINS * 2) * LANES + lane;
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
          atomicAdd(hist + target, sum);
          atomicAdd(hist + target + 1, count);
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

// One model's histograms of obfuscated features (used for K = 1: it keeps no per-model words in shared memory).
// One feature per lane ("lane owns column"). A block is one warp; its 32 lanes take 32 sampled features and walk the
// same rows in 32-row blocks aligned on absolute rows. Per row block, lane j loads row j's bag, node and q
// (coalesced) and turns them into one word (-1 = skip, else local slot * 256 + q + 128), and loads its own feature's
// 32 codes with one 16-byte load (two rows per byte). A row block with no selected row is skipped (one vote). Two
// accumulation paths, chosen by the host:
//   popcount (few slots, levels 0-3): per slot, ballots give the block's rows of that slot and the bits of |q| by
//     sign; per lane, `above` gives the rows whose code is > t. Count and sum q of (slot, code > t), t = -1..3, are
//     popcounts, accumulated once per row block. Bins follow by differences: bin b = (> b-1) - (> b).
//   per row (many slots): for each of the 32 rows (unrolled) the word is broadcast with a shuffle and each lane adds
//     q * 2^16 + 1 to its own (slot, bin) counter, sum q in the high and the count in the low 16 bits, exact while
//     rows_per_group * half_bins <= 32767 (the host guarantees it).
// Counters are lane-minor in shared memory: no atomics, no bank conflicts. Lanes finally add their non-zero sums and
// counts to the device histogram (integer atomics, so the order does not matter). Grid: (feature groups, row groups,
// slot chunks); shared memory sized by the host. Columns must be 16-byte aligned, with 16 bytes readable past the
// last row.
extern "C" __global__ void histograms_obfuscated_one(const int *columns, const int *offsets, const uchar *packed,
                                                     const uchar *bag, const uchar *node, const schar *q,
                                                     const int *slots, int *hist, SingleObfuscatedParams p) {
  uint lane = threadIdx.x;
  uint feature = blockIdx.x * LANES + lane;
  uint first_slot = blockIdx.z * p.slots_per_chunk;
  uint end_slot = smaller(first_slot + p.slots_per_chunk, p.n_slots);
  uint n_local = end_slot - first_slot;
  // popcount: (slot, t + 1 for t = -1..3, {count, sum}); per row: (slot, bin) packed
  uint span = p.popcount != 0 ? n_local * (OBFUSCATED_BINS * 2) : n_local * OBFUSCATED_BINS;
  for (uint k = 0; k < span; ++k) {
    local[k * LANES + lane] = 0;
  }
  bool active = feature < p.n_features;
  const uchar *column = packed + (active ? ulong(columns[feature]) * p.packed_stride : 0ull);
  uint start = p.first_row + blockIdx.y * p.rows_per_group;  // absolute rows [start, stop)
  uint stop = p.first_row + smaller((blockIdx.y + 1) * p.rows_per_group, p.n);
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
    if (__ballot_sync(WARP, word >= 0) == 0) {
      continue;
    }
    Codes codes = load_codes(column, block, active);
    if (p.popcount != 0) {
      uint gt[OBFUSCATED_BINS] = {0xFFFFFFFFu, above(codes, 0), above(codes, 1), above(codes, 2), above(codes, 3)};
      int value = (word & 255) - 128;
      uint magnitude = uint(value < 0 ? -value : value);
      for (uint s = 0; s < n_local; ++s) {
        bool here = word >= 0 && uint(word >> 8) == s;
        uint rows = __ballot_sync(WARP, here);
        if (rows == 0) {
          continue;
        }
        int *counters = local + s * (OBFUSCATED_BINS * 2) * LANES + lane;
        for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
          counters[(2 * t) * LANES] += __popc(gt[t] & rows);
        }
        for (uint b = 0; b < p.q_bits; ++b) {
          uint positive = __ballot_sync(WARP, here && value > 0 && ((magnitude >> b) & 1u) != 0);
          uint negative = __ballot_sync(WARP, here && value < 0 && ((magnitude >> b) & 1u) != 0);
          for (uint t = 0; t < OBFUSCATED_BINS; ++t) {
            counters[(2 * t + 1) * LANES] += (__popc(gt[t] & positive) - __popc(gt[t] & negative)) * (1 << b);
          }
        }
      }
      continue;
    }
#pragma unroll
    for (uint t = 0; t < 32; ++t) {
      int w = __shfl_sync(WARP, word, t);
      if (w >= 0) {
        local[(uint(w >> 8) * OBFUSCATED_BINS + code_of(codes, t)) * LANES + lane] += ((w & 255) - 128) * 65536 + 1;
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
        int *counters = local + s * (OBFUSCATED_BINS * 2) * LANES + lane;
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
        atomicAdd(hist + target, sum);
        atomicAdd(hist + target + 1, count);
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
extern "C" __global__ void partition_rows(const int *kinds, const int *columns, const uchar *split_kind,
                                          const int *split_feature, const uchar *threshold, const uchar *missing_left,
                                          const ulong *cat_left, const uchar *named, const uchar *packed,
                                          const uchar *country, uchar *node, PartitionParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i >= p.n) {
    return;
  }
  ulong index = ulong(model) * p.n + i;
  uint here = model * p.width + node[index];
  int f = split_feature[here];
  uint bin = bin_of(kinds[f], columns[f], p.first_row + i, named, p.named_stride, packed, p.packed_stride, country,
                    p.cat_stride);
  node[index] = uchar(2 * uint(node[index]) +
                      goes_right(split_kind[here], bin, threshold[here], missing_left[here], cat_left[here]));
}

// Multi-target own last level: node_own[t * n + i] = 2 * node[i] + go_right under target t's own splits of the
// deepest level; the shared node array is left as it is. Grid: (rows, targets).
extern "C" __global__ void partition_own(const int *kinds, const int *columns, const uchar *split_kind,
                                         const int *split_feature, const uchar *threshold, const uchar *missing_left,
                                         const ulong *cat_left, const uchar *named, const uchar *packed,
                                         const uchar *country, const uchar *node, uchar *node_own, PartitionParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i >= p.n) {
    return;
  }
  uint here = model * p.width + node[i];
  int f = split_feature[here];
  uint bin = bin_of(kinds[f], columns[f], p.first_row + i, named, p.named_stride, packed, p.packed_stride, country,
                    p.cat_stride);
  node_own[ulong(model) * p.n + i] =
      uchar(2 * uint(node[i]) + goes_right(split_kind[here], bin, threshold[here], missing_left[here], cat_left[here]));
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
extern "C" __global__ void split_leaf(const int *kinds, const int *columns, const uchar *split_kind,
                                      const int *split_feature, const uchar *threshold, const uchar *missing_left,
                                      const ulong *cat_left, const uint *leaves, const uchar *named,
                                      const uchar *packed, const uchar *country, uchar *node, LeafSplitParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i >= p.n) {
    return;
  }
  ulong index = ulong(model) * p.n + i;
  if (uint(node[index]) != leaves[2 * model]) {
    return;
  }
  int f = split_feature[model];
  uint bin = bin_of(kinds[f], columns[f], p.first_row + i, named, p.named_stride, packed, p.packed_stride, country,
                    p.cat_stride);
  if (goes_right(split_kind[model], bin, threshold[model], missing_left[model], cat_left[model]) != 0) {
    node[index] = uchar(leaves[2 * model + 1]);
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

// Exact per-leaf sums of rint(value * 2^32) over in-bag rows, per model and block of rows; a row's value is its
// gradient (or its hessian) as stored, or pred - y (`residual`). As on Metal, each value is split as
// hi * 2^32 + mid * 2^16 + low with int hi and 16-bit mid and low, added with 32-bit atomics in shared memory. The
// host keeps max |value| * rows below 2^31, so with rows_per_group <= 4096 no 32-bit partial can overflow
// (|sum hi| < 2^31, sum mid, sum low < 2^28). The host adds the blocks' int64 partials. Grid: (row chunks, models).
extern "C" __global__ void leaf_partials(const float *a, const float *b, const uchar *bag, const uchar *node,
                                         uint *sums, uint *counts, LeafParams p) {
  __shared__ int hi[MAX_LEAVES];
  __shared__ uint mid[MAX_LEAVES];
  __shared__ uint low[MAX_LEAVES];
  __shared__ uint count[MAX_LEAVES];
  uint tid = threadIdx.x;
  uint width = blockDim.x;
  for (uint k = tid; k < p.n_leaves; k += width) {
    hi[k] = 0;
    mid[k] = 0u;
    low[k] = 0u;
    count[k] = 0u;
  }
  __syncthreads();
  uint start = blockIdx.x * p.rows_per_group;
  uint stop = smaller(start + p.rows_per_group, p.n);
  ulong model = ulong(blockIdx.y) * p.n;
  ulong nodes = ulong(blockIdx.y) * p.node_stride;
  for (uint i = start + tid; i < stop; i += width) {
    if (bag[i] == 0) {
      continue;
    }
    float value = p.residual != 0 ? a[model + i] - b[model + i] : a[model + i];
    long long fixed = (long long)rintf(value * 0x1p32f);
    uint leaf = node[nodes + i];
    uint bits = uint(fixed & 0xFFFFFFFFll);
    atomicAdd(&hi[leaf], int(fixed >> 32));
    atomicAdd(&mid[leaf], bits >> 16);
    atomicAdd(&low[leaf], bits & 0xFFFFu);
    atomicAdd(&count[leaf], 1u);
  }
  __syncthreads();
  ulong out = (ulong(blockIdx.y) * p.groups + blockIdx.x) * p.n_leaves;
  for (uint k = tid; k < p.n_leaves; k += width) {
    sums[3 * (out + k)] = uint(hi[k]);  // hi (as uint), mid, low: the host adds them in int64
    sums[3 * (out + k) + 1] = mid[k];
    sums[3 * (out + k) + 2] = low[k];
    counts[out + k] = count[k];
  }
}

struct UpdateParams {
  uint n;
  uint n_leaves;
  uint node_stride;  // as LeafParams
};

// pred += leaf_values[model, node]: one float32 addition per row and model.
extern "C" __global__ void update_predictions(float *pred, const uchar *node, const float *leaf_values,
                                              UpdateParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  uint model = blockIdx.y;
  if (i < p.n) {
    ulong index = ulong(model) * p.n + i;
    pred[index] = pred[index] + leaf_values[model * p.n_leaves + node[ulong(model) * p.node_stride + i]];
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
extern "C" __global__ void predict_trees(const int *kinds, const int *columns, const uchar *kind, const int *feature,
                                         const uchar *threshold, const uchar *missing_left, const ulong *cat_left,
                                         const float *leaf_value, const uchar *named, const uchar *packed,
                                         const uchar *country, float *out, PredictParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
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
extern "C" __global__ void predict_nodes(const int *kinds, const int *columns, const uchar *kind, const int *feature,
                                         const uchar *threshold, const uchar *missing_left, const ulong *cat_left,
                                         const short *left, const short *right, const float *leaf_value,
                                         const uchar *named, const uchar *packed, const uchar *country, float *out,
                                         NodePredictParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
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
extern "C" __global__ void build_histograms_multi(const int *kinds, const int *columns, const int *offsets,
                                                  const uchar *named, const uchar *packed, const uchar *country,
                                                  const uchar *bag, const uchar *node, const schar *q, const int *slots,
                                                  int *hist, MultiParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
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
    int *base = hist + (ulong(slot) * p.cells + cell) * (p.k + 1);
    for (uint t = 0; t < p.k; ++t) {
      atomicAdd(base + t, int(q[ulong(t) * p.n + i]));
    }
    atomicAdd(base + p.k, 1);
  }
}

// Per training row, its slot at this level for histograms_multi: -1 out of the bag or in a skipped node (slots < 128).
extern "C" __global__ void row_slots(const uchar *bag, const uchar *node, const int *slots, schar *row_slot,
                                     CountParams p) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.n) {
    row_slot[i] = bag[i] != 0 ? schar(slots[node[i]]) : schar(-1);
  }
}

// Multi-target histograms of obfuscated features (experimental), one feature per lane over 32-row blocks as
// histograms_obfuscated, with one node array for every target. A block (one warp) takes one chunk of slots and one
// chunk of at most 15 targets [t_lo, t_lo + kc): per (slot, bin) a lane keeps kc + 1 16-bit fields packed four to a
// 64-bit word; field t < kc adds q + half_bins of target t_lo + t (never negative), field kc counts the row; field f
// sits in bits 16 (f % 4) .. of word f / 4. The lane owning a row packs that row's words once; the 32 lanes receive
// the words they need by a shuffle, so one 64-bit add updates four fields. Exact while
// rows_per_group * 2 * half_bins <= 65535 (the host guarantees it); a sum of q is its field minus half_bins * count.
// Every target chunk unbiases with its own count; the first writes the count. Grid: (slot chunks x target chunks,
// target chunk fastest; feature groups; row groups). Shared memory: slots_per_chunk * 5 * words * 32 64-bit words.
extern "C" __global__ void histograms_multi(const int *columns, const int *offsets, const uchar *packed,
                                            const schar *row_slot, const schar *q, int *hist, MultiParams p) {
  uint lane = threadIdx.x;
  uint target_chunk = blockIdx.x % p.target_chunks;
  uint t_lo = target_chunk * p.targets_per_chunk;
  uint kc = smaller(p.targets_per_chunk, p.k - t_lo);
  uint words = (kc + 4) / 4;
  uint feature = blockIdx.y * LANES + lane;
  uint first_slot = (blockIdx.x / p.target_chunks) * p.slots_per_chunk;
  uint end_slot = smaller(first_slot + p.slots_per_chunk, p.n_slots);
  uint span = (end_slot - first_slot) * OBFUSCATED_BINS * words;
  for (uint c = 0; c < span; ++c) {
    local_wide[c * LANES + lane] = 0;
  }
  bool active = feature < p.n_features;
  const uchar *column = packed + (active ? ulong(columns[feature]) * p.packed_stride : 0ull);
  uint start = p.first_row + blockIdx.z * p.rows_per_group;
  uint stop = p.first_row + smaller((blockIdx.z + 1) * p.rows_per_group, p.n);
  for (uint block = start & ~31u; block < stop; block += 32) {
    uint row = block + lane;
    int slot = -1;
    ulong w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    if (row >= start && row < stop) {
      uint i = row - p.first_row;
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
        ulong one = 1ull << (16 * (kc % 4));
        uint which = kc / 4;
        w0 += which == 0 ? one : 0;
        w1 += which == 1 ? one : 0;
        w2 += which == 2 ? one : 0;
        w3 += which == 3 ? one : 0;
      }
    }
    uint present = __ballot_sync(WARP, slot >= 0);  // the block's rows of this chunk: a uniform loop over them
    if (present == 0) {
      continue;
    }
    Codes codes = load_codes(column, block, active);
    while (present != 0) {
      uint t = uint(__ffs(int(present))) - 1u;
      present &= present - 1;
      int s = __shfl_sync(WARP, slot, t);
      ulong *base = local_wide + ((uint(s) * OBFUSCATED_BINS + code_of(codes, t)) * words) * LANES + lane;
      base[0] += __shfl_sync(WARP, w0, t);  // inactive lanes write their own unused counters (no divergence
      if (words > 1) {                      // around shuffles)
        base[LANES] += __shfl_sync(WARP, w1, t);
      }
      if (words > 2) {
        base[2 * LANES] += __shfl_sync(WARP, w2, t);
      }
      if (words > 3) {
        base[3 * LANES] += __shfl_sync(WARP, w3, t);
      }
    }
  }
  if (!active) {
    return;
  }
  for (uint s = 0; s < end_slot - first_slot; ++s) {
    for (uint b = 0; b < OBFUSCATED_BINS; ++b) {
      ulong *base = local_wide + ((s * OBFUSCATED_BINS + b) * words) * LANES + lane;
      int count = int((base[(kc / 4) * LANES] >> (16 * (kc % 4))) & 0xFFFFull);
      if (count == 0) {
        continue;
      }
      int *target = hist + (ulong(first_slot + s) * p.cells + uint(offsets[feature]) + b) * (p.k + 1);
      for (uint t = 0; t < kc; ++t) {
        int field = int((base[(t / 4) * LANES] >> (16 * (t % 4))) & 0xFFFFull);
        atomicAdd(target + t_lo + t, field - int(p.half_bins) * count);
      }
      if (target_chunk == 0) {
        atomicAdd(target + p.k, count);
      }
    }
  }
}
