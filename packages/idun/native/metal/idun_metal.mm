// idun_metal.mm - Objective-C++ implementation of the C ABI in idun_metal.h.
//
// The Metal host layer: device probe, runtime MSL compilation with the fixed options every Idun kernel is built with,
// compute pipelines, shared and no-copy buffers, and command batches with GPU timestamps.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unistd.h>

#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "idun_metal.h"

#define IDUN_API extern "C" __attribute__((visibility("default")))

namespace {

// Metal contracts a * b + c into an FMA by default, even with MTLMathModeSafe; the CPU backend does not. Bit-exact
// parity between backends needs contraction off, so the shim prepends it to every library instead of trusting each
// kernel file to remember (docs/architecture/determinism.md).
constexpr const char *kPrelude = "#pragma METAL fp contract(off)\n";

void set_error(char *err, size_t err_len, const char *message) {
  if (err != nullptr && err_len > 0) {
    strlcpy(err, message, err_len);
  }
}

MTLCompileOptions *compile_options() {
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion3_2;
  options.mathMode = MTLMathModeSafe;
  options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  return options;
}

id<MTLLibrary> compile(id<MTLDevice> device, const char *source, char *err, size_t err_len) {
  NSString *text = [NSString stringWithFormat:@"%s%s", kPrelude, source];
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:text options:compile_options() error:&error];
  if (library == nil) {
    set_error(err, err_len, error.localizedDescription.UTF8String);
  }
  return library;
}

}  // namespace

struct idun_kernel {
  id<MTLComputePipelineState> pipeline;
};

struct idun_buffer {
  id<MTLBuffer> buffer;
};

struct idun_context {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  std::vector<std::unique_ptr<idun_kernel>> kernels;
  id<MTLCommandBuffer> batch;
  id<MTLComputeCommandEncoder> encoder;
  std::deque<id<MTLCommandBuffer>> committed;  // committed and not yet waited for, oldest first
};

// Every function that calls Metal does so inside its own @autoreleasepool. The host (Python) has no run loop to drain
// a pool, so anything Metal autoreleases lands in the thread's top-level pool and lives as long as the process: a
// buffer allocated outside a pool stayed alive after idun_buffer_free and idun_context_destroy (about 1 GB per fit).

IDUN_API int32_t idun_abi_version(void) { return IDUN_ABI_VERSION; }

IDUN_API const char *idun_source_prelude(void) { return kPrelude; }

IDUN_API int32_t idun_probe(idun_device_info *out, char *err, size_t err_len) {
  if (out == nullptr) {
    set_error(err, err_len, "idun_probe: out is NULL");
    return IDUN_ERR_ARGUMENT;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      set_error(err, err_len, "no Metal device: MTLCreateSystemDefaultDevice returned nil");
      return IDUN_ERR_NO_DEVICE;
    }
    std::memset(out, 0, sizeof(*out));
    strlcpy(out->name, device.name.UTF8String, sizeof(out->name));
    out->max_threadgroup_memory_bytes = device.maxThreadgroupMemoryLength;
    out->recommended_working_set_bytes = device.recommendedMaxWorkingSetSize;
    out->max_buffer_bytes = device.maxBufferLength;
    out->max_threads_per_threadgroup = static_cast<uint32_t>(device.maxThreadsPerThreadgroup.width);
    out->page_size_bytes = static_cast<uint32_t>(getpagesize());
    out->has_unified_memory = device.hasUnifiedMemory ? 1U : 0U;
    out->supports_apple7 = [device supportsFamily:MTLGPUFamilyApple7] ? 1U : 0U;
    out->supports_apple9 = [device supportsFamily:MTLGPUFamilyApple9] ? 1U : 0U;
    out->supports_metal4 = 0U;  // the Metal 4 family exists from macOS 26; the library runs from macOS 15
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0  // an older SDK does not declare it
    if (@available(macOS 26.0, *)) {
      out->supports_metal4 = [device supportsFamily:MTLGPUFamilyMetal4] ? 1U : 0U;
    }
#endif
  }
  return IDUN_OK;
}

IDUN_API int32_t idun_compile_check(const char *source, char *err, size_t err_len) {
  if (source == nullptr) {
    set_error(err, err_len, "idun_compile_check: source is NULL");
    return IDUN_ERR_ARGUMENT;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      set_error(err, err_len, "no Metal device: MTLCreateSystemDefaultDevice returned nil");
      return IDUN_ERR_NO_DEVICE;
    }
    return compile(device, source, err, err_len) == nil ? IDUN_ERR_COMPILE : IDUN_OK;
  }
}

IDUN_API idun_context *idun_context_create(char *err, size_t err_len) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      set_error(err, err_len, "no Metal device: MTLCreateSystemDefaultDevice returned nil");
      return nullptr;
    }
    auto ctx = std::make_unique<idun_context>();
    ctx->device = device;
    ctx->queue = [device newCommandQueue];
    return ctx.release();
  }
}

IDUN_API void idun_context_destroy(idun_context *ctx) {
  if (ctx != nullptr) {
    @autoreleasepool {
      if (ctx->encoder != nil) {
        [ctx->encoder endEncoding];
      }
      for (id<MTLCommandBuffer> batch : ctx->committed) {
        [batch waitUntilCompleted];  // nothing reads their results any more, but they may still use the buffers
      }
      delete ctx;
    }
  }
}

IDUN_API int32_t idun_library_load(idun_context *ctx, const char *source, char *err, size_t err_len) {
  if (ctx == nullptr || source == nullptr) {
    set_error(err, err_len, "idun_library_load: ctx or source is NULL");
    return IDUN_ERR_ARGUMENT;
  }
  @autoreleasepool {
    id<MTLLibrary> library = compile(ctx->device, source, err, err_len);
    if (library == nil) {
      return IDUN_ERR_COMPILE;
    }
    ctx->library = library;
    ctx->kernels.clear();
  }
  return IDUN_OK;
}

IDUN_API idun_kernel *idun_kernel_get(idun_context *ctx, const char *name, char *err, size_t err_len) {
  if (ctx == nullptr || name == nullptr) {
    set_error(err, err_len, "idun_kernel_get: ctx or name is NULL");
    return nullptr;
  }
  if (ctx->library == nil) {
    set_error(err, err_len, "idun_kernel_get: no library loaded");
    return nullptr;
  }
  @autoreleasepool {
    id<MTLFunction> function = [ctx->library newFunctionWithName:@(name)];
    if (function == nil) {
      set_error(err, err_len, (std::string("idun_kernel_get: no kernel function ") + name).c_str());
      return nullptr;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      set_error(err, err_len, error.localizedDescription.UTF8String);
      return nullptr;
    }
    auto kernel = std::make_unique<idun_kernel>();
    kernel->pipeline = pipeline;
    ctx->kernels.push_back(std::move(kernel));
    return ctx->kernels.back().get();
  }
}

IDUN_API uint32_t idun_kernel_max_threads(const idun_kernel *kernel) {
  return kernel == nullptr ? 0U : static_cast<uint32_t>(kernel->pipeline.maxTotalThreadsPerThreadgroup);
}

IDUN_API uint32_t idun_kernel_simd_width(const idun_kernel *kernel) {
  return kernel == nullptr ? 0U : static_cast<uint32_t>(kernel->pipeline.threadExecutionWidth);
}

IDUN_API idun_buffer *idun_buffer_alloc(idun_context *ctx, size_t bytes, void **contents, char *err, size_t err_len) {
  if (ctx == nullptr || contents == nullptr || bytes == 0) {
    set_error(err, err_len, "idun_buffer_alloc: ctx or contents is NULL, or bytes is 0");
    return nullptr;
  }
  @autoreleasepool {
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      set_error(err, err_len, "idun_buffer_alloc: the device refused the allocation");
      return nullptr;
    }
    std::memset(buffer.contents, 0, bytes);
    *contents = buffer.contents;
    auto out = std::make_unique<idun_buffer>();
    out->buffer = buffer;
    return out.release();
  }
}

IDUN_API idun_buffer *idun_buffer_wrap(idun_context *ctx, void *page_aligned, size_t bytes, char *err, size_t err_len) {
  const auto page = static_cast<uintptr_t>(getpagesize());
  if (ctx == nullptr || page_aligned == nullptr || bytes == 0 ||
      reinterpret_cast<uintptr_t>(page_aligned) % page != 0 || bytes % page != 0) {
    set_error(err, err_len, "idun_buffer_wrap: needs a context and a non-empty, page-aligned address and length");
    return nullptr;
  }
  @autoreleasepool {
    id<MTLBuffer> buffer = [ctx->device newBufferWithBytesNoCopy:page_aligned
                                                          length:bytes
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    if (buffer == nil) {
      set_error(err, err_len, "idun_buffer_wrap: the device refused the memory");
      return nullptr;
    }
    auto out = std::make_unique<idun_buffer>();
    out->buffer = buffer;
    return out.release();
  }
}

IDUN_API void idun_buffer_free(idun_buffer *buffer) {
  @autoreleasepool {
    delete buffer;
  }
}

IDUN_API int32_t idun_batch_begin(idun_context *ctx, char *err, size_t err_len) {
  if (ctx == nullptr) {
    set_error(err, err_len, "idun_batch_begin: ctx is NULL");
    return IDUN_ERR_ARGUMENT;
  }
  if (ctx->batch != nil) {
    set_error(err, err_len, "idun_batch_begin: a batch is already open");
    return IDUN_ERR_STATE;
  }
  @autoreleasepool {
    ctx->batch = [ctx->queue commandBuffer];
    ctx->encoder = [ctx->batch computeCommandEncoder];
  }
  return IDUN_OK;
}

IDUN_API int32_t idun_dispatch(idun_context *ctx, const idun_kernel *kernel, idun_buffer *const *buffers,
                               uint32_t n_buffers, const void *params, size_t params_bytes, uint32_t threadgroup_bytes,
                               const uint32_t groups[3], const uint32_t threads[3], char *err, size_t err_len) {
  if (ctx == nullptr || kernel == nullptr || groups == nullptr || threads == nullptr ||
      (n_buffers > 0 && buffers == nullptr) || n_buffers > IDUN_MAX_BUFFERS || params_bytes > IDUN_MAX_PARAMS ||
      (params_bytes > 0 && params == nullptr) || threadgroup_bytes % 16 != 0) {
    set_error(err, err_len, "idun_dispatch: bad arguments");
    return IDUN_ERR_ARGUMENT;
  }
  if (ctx->encoder == nil) {
    set_error(err, err_len, "idun_dispatch: no open batch");
    return IDUN_ERR_STATE;
  }
  const uint64_t per_group = static_cast<uint64_t>(threads[0]) * threads[1] * threads[2];
  if (per_group == 0 || per_group > kernel->pipeline.maxTotalThreadsPerThreadgroup) {
    set_error(err, err_len, "idun_dispatch: threads per threadgroup is 0 or above the pipeline's maximum");
    return IDUN_ERR_ARGUMENT;
  }
  if (threadgroup_bytes + kernel->pipeline.staticThreadgroupMemoryLength > ctx->device.maxThreadgroupMemoryLength) {
    set_error(err, err_len, "idun_dispatch: threadgroup memory above the device's maximum");
    return IDUN_ERR_ARGUMENT;
  }
  for (uint32_t i = 0; i < n_buffers; ++i) {
    if (buffers[i] == nullptr) {
      set_error(err, err_len, "idun_dispatch: a buffer is NULL");
      return IDUN_ERR_ARGUMENT;
    }
  }
  @autoreleasepool {
    [ctx->encoder setComputePipelineState:kernel->pipeline];
    for (uint32_t i = 0; i < n_buffers; ++i) {
      [ctx->encoder setBuffer:buffers[i]->buffer offset:0 atIndex:i];
    }
    if (params_bytes > 0) {
      [ctx->encoder setBytes:params length:params_bytes atIndex:n_buffers];
    }
    if (threadgroup_bytes > 0) {
      [ctx->encoder setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    }
    [ctx->encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
                 threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
  }
  return IDUN_OK;
}

namespace {

// Commit the open batch without waiting; `caller` names the API function in error messages.
int32_t commit_open_batch(idun_context *ctx, const char *caller, char *err, size_t err_len) {
  if (ctx == nullptr) {
    set_error(err, err_len, (std::string(caller) + ": ctx is NULL").c_str());
    return IDUN_ERR_ARGUMENT;
  }
  if (ctx->batch == nil) {
    set_error(err, err_len, (std::string(caller) + ": no open batch").c_str());
    return IDUN_ERR_STATE;
  }
  @autoreleasepool {
    [ctx->encoder endEncoding];
    [ctx->batch commit];
    ctx->committed.push_back(ctx->batch);
    ctx->encoder = nil;
    ctx->batch = nil;
  }
  return IDUN_OK;
}

}  // namespace

IDUN_API int32_t idun_batch_commit(idun_context *ctx, char *err, size_t err_len) {
  return commit_open_batch(ctx, "idun_batch_commit", err, err_len);
}

IDUN_API int32_t idun_batch_wait(idun_context *ctx, uint32_t keep, double *gpu_seconds, char *err, size_t err_len) {
  if (ctx == nullptr) {
    set_error(err, err_len, "idun_batch_wait: ctx is NULL");
    return IDUN_ERR_ARGUMENT;
  }
  int32_t status = IDUN_OK;
  double seconds = 0.0;
  @autoreleasepool {
    while (ctx->committed.size() > keep) {
      id<MTLCommandBuffer> batch = ctx->committed.front();
      ctx->committed.pop_front();
      [batch waitUntilCompleted];
      if (batch.status != MTLCommandBufferStatusCompleted) {
        if (status == IDUN_OK) {  // the first failure is the one to report; later batches waited for all the same
          set_error(
              err, err_len,
              batch.error != nil ? batch.error.localizedDescription.UTF8String : "the command buffer did not complete");
        }
        status = IDUN_ERR_GPU;
      } else {
        seconds += batch.GPUEndTime - batch.GPUStartTime;
      }
    }
  }
  if (gpu_seconds != nullptr) {
    *gpu_seconds = seconds;
  }
  return status;
}

IDUN_API int32_t idun_batch_commit_wait(idun_context *ctx, double *gpu_seconds, char *err, size_t err_len) {
  int32_t status = commit_open_batch(ctx, "idun_batch_commit_wait", err, err_len);
  return status != IDUN_OK ? status : idun_batch_wait(ctx, 0, gpu_seconds, err, err_len);
}
