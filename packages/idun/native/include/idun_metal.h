/*
 * idun_metal.h - C ABI of the Idun Metal runtime (libidun_metal.dylib).
 *
 * Python binds this header through ctypes (src/idun/metal/runtime.py). Conventions, which the CUDA runtime's ctypes
 * binding follows too (docs/architecture/backend-interface.md):
 *   - every function is extern "C", takes and returns only C types, and never lets an exception or Objective-C
 *     exception cross the boundary;
 *   - fallible functions return IDUN_OK (0) on success, or a negative IDUN_ERR_* code with a NUL-terminated message
 *     written to (err, err_len); err may be NULL when err_len is 0; functions returning a handle return NULL on
 *     failure and write the message the same way;
 *   - the caller owns the memory it passes in; the library owns the handles it returns until the matching
 *     _destroy/_free; kernels belong to their context;
 *   - IDUN_ABI_VERSION changes on every incompatible change, and the binding refuses to load a mismatched library.
 */
#ifndef IDUN_METAL_H
#define IDUN_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IDUN_ABI_VERSION 6

#define IDUN_OK 0
#define IDUN_ERR_NO_DEVICE (-1)
#define IDUN_ERR_COMPILE (-2)
#define IDUN_ERR_ARGUMENT (-3)
#define IDUN_ERR_STATE (-4) /* a call out of order: dispatch without a batch, a kernel before a library, ... */
#define IDUN_ERR_GPU (-5)   /* a command buffer failed on the device */

#define IDUN_NAME_LEN 128
#define IDUN_MAX_BUFFERS 30  /* buffer indices 0..29; the params block goes to index n_buffers */
#define IDUN_MAX_PARAMS 4096 /* setBytes limit */

typedef struct idun_device_info {
  char name[IDUN_NAME_LEN];
  uint64_t max_threadgroup_memory_bytes;
  uint64_t recommended_working_set_bytes;
  uint64_t max_buffer_bytes;
  uint32_t max_threads_per_threadgroup;
  uint32_t page_size_bytes;
  uint32_t has_unified_memory;
  uint32_t supports_apple7; /* M1 family and newer: what the idun_gbm Metal backend needs */
  uint32_t supports_apple9; /* M3 and M4 families: SIMD width 32, 32 KiB threadgroup memory */
  uint32_t supports_metal4;
} idun_device_info;

typedef struct idun_context idun_context; /* device, queue, library, kernels, the open command batch */
typedef struct idun_kernel idun_kernel;   /* a compute pipeline, owned by its context */
typedef struct idun_buffer idun_buffer;   /* a shared-storage MTLBuffer */

/* IDUN_ABI_VERSION of the compiled library. */
int32_t idun_abi_version(void);

/* The source text prepended to every kernel library before compilation (determinism.md: FP contraction off). */
const char *idun_source_prelude(void);

/* Describe the system default Metal device. */
int32_t idun_probe(idun_device_info *out, char *err, size_t err_len);

/* Compile MSL `source` with Idun's fixed compile options and prelude; on failure the compiler log goes to err. */
int32_t idun_compile_check(const char *source, char *err, size_t err_len);

/* A context on the system default device: one command queue, at most one library, at most one open batch. */
idun_context *idun_context_create(char *err, size_t err_len);
/* Destroy a context and its kernels (buffers are freed separately). Commits nothing: an open batch is dropped;
 * committed batches are waited for (their errors are not reported). */
void idun_context_destroy(idun_context *ctx);

/* Compile `source` (with the prelude and fixed options) as the context's library; replaces kernels of a previous one.
 */
int32_t idun_library_load(idun_context *ctx, const char *source, char *err, size_t err_len);
/* The compute pipeline of kernel function `name` of the loaded library. */
idun_kernel *idun_kernel_get(idun_context *ctx, const char *name, char *err, size_t err_len);
/* Largest threadgroup the pipeline accepts, and its SIMD width. */
uint32_t idun_kernel_max_threads(const idun_kernel *kernel);
uint32_t idun_kernel_simd_width(const idun_kernel *kernel);

/* A new zero-filled shared buffer of `bytes` (> 0); its CPU address goes to *contents. */
idun_buffer *idun_buffer_alloc(idun_context *ctx, size_t bytes, void **contents, char *err, size_t err_len);
/* Wrap caller memory without copying: `page_aligned` and `bytes` must be multiples of the page size. The memory must
 * stay valid and mapped until idun_buffer_free; the GPU may read it (and write it, if the mapping is writable). */
idun_buffer *idun_buffer_wrap(idun_context *ctx, void *page_aligned, size_t bytes, char *err, size_t err_len);
void idun_buffer_free(idun_buffer *buffer);

/* Open a command batch: one command buffer with one serial compute encoder. */
int32_t idun_batch_begin(idun_context *ctx, char *err, size_t err_len);
/* Encode one dispatch into the open batch: buffers at indices 0..n_buffers-1, `params` (if any) at index n_buffers,
 * `threadgroup_bytes` (a multiple of 16; 0 for none) of dynamic threadgroup memory at threadgroup index 0, `groups`
 * threadgroups of `threads` threads. Dispatches of one batch run in order. */
int32_t idun_dispatch(idun_context *ctx, const idun_kernel *kernel, idun_buffer *const *buffers, uint32_t n_buffers,
                      const void *params, size_t params_bytes, uint32_t threadgroup_bytes, const uint32_t groups[3],
                      const uint32_t threads[3], char *err, size_t err_len);
/* Commit the open batch without waiting (v5). Batches run in the order they are committed, each after the one before
 * it finished, so the host can encode the next batch while the GPU runs this one. */
int32_t idun_batch_commit(idun_context *ctx, char *err, size_t err_len);
/* Wait, oldest first, until at most `keep` committed batches are left unwaited (v5); report the GPU time of those
 * waited for (sum of GPUEndTime - GPUStartTime). Returns IDUN_ERR_GPU with the first failure's message if any failed.
 */
int32_t idun_batch_wait(idun_context *ctx, uint32_t keep, double *gpu_seconds, char *err, size_t err_len);
/* Commit the open batch and wait for it and every batch committed before it; report their GPU time. */
int32_t idun_batch_commit_wait(idun_context *ctx, double *gpu_seconds, char *err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* IDUN_METAL_H */
