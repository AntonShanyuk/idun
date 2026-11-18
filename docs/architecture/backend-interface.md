# Backend interface

Three layers:

- the **Python `Backend` protocol** that the booster drives (CPU, Metal, CUDA);
- the **`Device` protocol** that a GPU backend's passes drive, whichever GPU runs the kernels;
- the **C ABI** of the native device runtimes underneath the GPU backends.

The protocol below is implemented by the CPU backend and the two GPU backends; Metal and CUDA share one
implementation of it (`idun.gbm.backends.device`), which drives a `Device`.

## 1. Python protocol (`idun.gbm.backends.base.Backend`)

```python
class Backend(Protocol):
    name: str  # "cpu" | "metal" | "cuda"

    @property
    def pred(self) -> NDArray[np.float32]: ...  # per training row; read-only on a GPU backend, until the next pass
    @property
    def g(self) -> NDArray[np.float32]: ...  # to read or write before the next pass (a custom objective writes it)

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, y: NDArray[np.float32]) -> None:
        """Once per fit: bind training rows [lo, lo + len(y)); pred = 0 (Metal: zero-copy wrap of the cache)."""

    def set_predictions(self, pred: NDArray[np.float32]) -> None:
        """After start: pred = these values (continued training, a start from the average); a host write."""

    def gradients(self) -> float:
        """g = pred − y on every training row; return max |g| (the quantization scale's input)."""

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scale: np.float32, half_bins: int) -> None:
        """q = stochastic rounding of g · inv_scale with rng draws of the absolute row; 0 out of the bag."""

    def quantize_hessians(self, key: int, inv_scale: np.float32, half_bins: int) -> None:
        """q_h = the same rounding of h · inv_scale (quantize's bag and draws), for second-order splits."""

    def start_tree(self) -> None:
        """node = 0 for every row."""

    def histograms(
        self, features, offsets, cells: int, slots: NDArray[np.int16], *, hessians: bool = False
    ) -> NDArray[np.int32]:
        """Integer (Σq, count) per (slot, cell) over in-bag rows of the nodes with slots[node] >= 0.

        With hessians, (Σq_h, count): the same kernels over q_h. The K-model driver passes the keyword
        only on that pass, so a backend without it still fits everything but a NewtonObjective's splits.
        """

    def partition(self, splits: LevelSplits) -> None:
        """node = 2·node + go_right for every training row (in and out of the bag)."""

    def split_leaf(self, split: LeafSplit) -> None:
        """Leaf-wise (node = leaf id): rows of leaf split.leaf that go right move to leaf split.right."""

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Σ rint((pred − y) · 2^32) and in-bag count per leaf, exact."""

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """pred += leaf_values[node] for every training row."""

    def predict(self, data: BinnedData, layout: FeatureLayout, lo: int, hi: int, forest: Trees) -> NDArray[np.float32]:
        """Sum of tree outputs per row (heap or node-list walk on bins), trees added in order from 0.0."""
```

Contract for every backend:

- Outputs are **bit-identical** to the CPU backend's. The parity tests compare with `==`
  ([determinism.md](determinism.md)).
- Split finding is **not** in the protocol. It runs host-side in one shared implementation (`idun.gbm.split`), so split
  parity follows from histogram parity. Sibling histograms (parent − smaller child) are subtracted on the host for now.
- Randomness enters only through `iteration_key` (device draws) and host-side masks (bagging, feature sampling).
- Construction raises `BackendUnavailableError` when the backend cannot run (`idun.gbm.backends.get_backend`).
  Nothing selects a backend automatically.

## 2. Device protocol (`idun.device.Device`)

A GPU backend is two parts: the passes of its learner, written once against `Device`
(`idun.gbm.backends.device`: `DeviceMany`, `DeviceBackend`; `idun.tabm.device.DeviceTabM`;
`idun.realmlp.device.DeviceRealMLP`), and a device that runs one kernel library (`idun.metal.device.MetalDevice`,
`idun.cuda.device.CudaDevice`). `MetalBackend` is idun_gbm's passes on a Metal device with `gbm.metal` loaded;
`MetalTabM` and `MetalRealMLP` are the networks on Metal devices with their libraries, `CudaTabM` and `CudaRealMLP` on
CUDA devices with theirs (`nn/kernels/cuda`: the Metal kernels' names, buffers and launch shapes).

```python
class Device(Protocol):
    def alloc(self, nbytes: int) -> Buffer: ...  # zero-filled
    def place(self, name: str, array) -> tuple[Buffer, int]: ...  # data; the row stride in bytes
    def copy_in(self, array) -> Buffer: ...
    def read(self, buffer, dtype, shape) -> NDArray: ...  # current values, read-only
    def write(self, buffer, dtype, shape) -> NDArray: ...  # current values, for the host to change
    def fill(self, buffer, value: int, nbytes: int) -> None: ...  # on the device
    def encode(self, name, buffers, params, groups, threads, local_bytes=0) -> None: ...
    def submit(self) -> None: ...  # start the encoded passes, do not wait
    def flush(self) -> None: ...  # wait for every pass
    def close(self) -> None: ...
```

The host never looks into a buffer. It reaches the values through three calls, and their rules are what a GPU with
memory of its own needs:

| call | what the host gets | valid until |
|---|---|---|
| `read` | the current values, read-only (passes that write the buffer finish first) | the next pass that writes the buffer: ask again |
| `write` | the current values in an array it may change; the changes reach the device before the next pass that uses the buffer | that pass is encoded: ask again |
| `fill` | nothing: the first bytes are set to one value on the device | — |

Data given to `place` is never read or written by the host again.

`submit` starts the encoded passes without waiting for them, so the host can encode the next ones while the GPU runs
these: a later `read` of a buffer they write, or `write` or `fill` of a buffer they use, waits for them (on Metal only
for the batches concerned, oldest first), and `flush` waits for everything. RealMLP's training steps run that way
(`DeviceRealMLP.train_steps_rows`: two row-index and loss buffers in turn, a step's loss read while the next step runs).

Every kernel is declared in the learner's table (`idun.gbm.backends.device.TABLE`) as a `Signature`: the number of
buffers it binds, the indices of the ones it writes, and its parameter structure. `encode` refuses a call that does
not match. A device with memory of its own downloads at `read` only what a kernel declared it writes, and uploads
before a pass only what `write` handed out; a test holds the table to the kernel source (a buffer is written unless
it is `const` there).

On Metal the GPU's memory is the host's: `read` and `write` return views of it, nothing is copied, and the calls
only decide when the open command batch is committed (before `read` returns a buffer a pending pass writes, before
`write` or `fill` touches one a pending pass uses). So a missing `read`, a late host write or a wrong declaration
would still give the right numbers there. `idun.metal.mirrored.MirroredDevice` makes them visible on the Mac: it
wraps the Metal device and gives the host copies only.

| mistake | on the mirrored device |
|---|---|
| an array kept across a pass that writes its buffer | the array keeps the old values: results differ |
| a host write to a `write` array after the pass that used the buffer | `HostAccessError` at the buffer's next use, or at `close` |
| a kernel writes a buffer its signature declares read-only | `HostAccessError` after that pass (buffers up to 64 MiB are compared around every pass) |
| host access to data given to `place` | `HostAccessError` |

With the environment variable `IDUN_METAL_MIRRORED=1` every Metal backend opens a mirrored device;
`make test-metal-mirrored` runs every `metal` test that way. Results are the plain device's bit for bit; it is a
test device, slow on real data.

`idun.cuda.device.CudaDevice` is the device with memory of its own. A buffer lives on the GPU; the host sees it
through a page-locked mirror, made the first time `read` or `write` asks for the buffer:

| what the device knows of a buffer | what it saves |
|---|---|
| the leading bytes of the mirror that are current (none after a pass that writes the buffer, or a fill) | `read` and `write` download only the bytes the mirror lacks; a second `read` downloads nothing |
| the leading bytes `write` handed out since the last upload | a pass uploads those bytes of the buffers it binds, and nothing for a buffer the host did not ask to change |
| the buffer is still all zero (after `alloc` or a whole zero `fill`) | `read` clears the mirror instead of downloading; a zero `fill` is not sent |
| an upload from the mirror is queued and not yet waited for | `write` waits before it hands the mirror out again |

Uploads, fills and passes queue on one stream and run in the order they were encoded; the host waits at `flush`, at
a `read` or `write` that downloads, and at a `write` of a buffer whose upload is still queued. Data given to `place`
and `copy_in` goes up once, from the caller's memory, and has no mirror; `place` records `"uploaded"`, or `"repacked"`
for an array whose rows it first copies on the host to rows 16 bytes apart (that transient copy is freed once it is up;
Metal's `"copied"` is a second copy kept in the memory the host shares, which RealMLP's `bind` refuses). `uploaded_bytes` and `downloaded_bytes`
count what crossed.

## 3. C ABI of the device runtimes (`packages/idun/native/include/idun_metal.h`)

Conventions, binding for `idun_metal.h`. The CUDA runtime has no header of its own: it is Python `ctypes` over the CUDA
driver API and NVRTC, with the same surface in Python (section 3.1):

| rule | why |
|---|---|
| `extern "C"`; C types only; opaque handles (`idun_context*`, `idun_buffer*`, `idun_kernel*`) | callable from ctypes; the same header shape works for CUDA |
| return `int32_t` status: `IDUN_OK` = 0, negative `IDUN_ERR_*` codes; the message is written to a caller-provided `(char *err, size_t err_len)` | errors are values; the binding maps codes to exceptions (`runtime.error_for`) |
| no C++ or Objective-C exception crosses the boundary | undefined behaviour otherwise |
| `IDUN_ABI_VERSION`, checked by the binding at load | a stale `.dylib` fails loudly instead of misbehaving |
| the caller owns buffers it passes in; the library owns handles it returns, until the matching `_free`/`_destroy` | no hidden lifetimes |
| kernel sources are compiled with fixed options, and the prelude (`idun_source_prelude()`) is prepended by the library | determinism cannot be forgotten per kernel |

### v1

```c
int32_t     idun_abi_version(void);
const char *idun_source_prelude(void);
int32_t     idun_probe(idun_device_info *out, char *err, size_t err_len);
int32_t     idun_compile_check(const char *source, char *err, size_t err_len);
```

### v2–v4: the compute runtime

v3 added the `threadgroup_bytes` argument of `idun_dispatch`; v4 added `idun_device_info.supports_apple7` (the Metal
backend needs an Apple7 GPU or newer).

```c
idun_context *idun_context_create(char *err, size_t err_len);
void        idun_context_destroy(idun_context *);                       /* frees kernels; drops an open batch */
int32_t     idun_library_load(idun_context *, const char *source, char *err, size_t err_len);   /* prelude enforced */
idun_kernel  *idun_kernel_get(idun_context *, const char *name, char *err, size_t err_len);
uint32_t    idun_kernel_max_threads(const idun_kernel *);
uint32_t    idun_kernel_simd_width(const idun_kernel *);
idun_buffer  *idun_buffer_alloc(idun_context *, size_t bytes, void **contents, char *err, size_t err_len);  /* shared, zeroed */
idun_buffer  *idun_buffer_wrap(idun_context *, void *page_aligned, size_t bytes, char *err, size_t err_len); /* no copy */
void        idun_buffer_free(idun_buffer *);
int32_t     idun_batch_begin(idun_context *, char *err, size_t err_len);
int32_t     idun_dispatch(idun_context *, const idun_kernel *, idun_buffer *const *buffers, uint32_t n_buffers,
                        const void *params, size_t params_bytes, uint32_t threadgroup_bytes,  /* v3 */
                        const uint32_t groups[3], const uint32_t threads[3], char *err, size_t err_len);
                        /* params at index n_buffers; dynamic threadgroup memory at index 0; batches run in order */
int32_t     idun_batch_commit_wait(idun_context *, double *gpu_seconds, char *err, size_t err_len);
```

New codes: `IDUN_ERR_STATE` (−4, a call out of order) and `IDUN_ERR_GPU` (−5, a command buffer failed). Python wraps it as
`idun.metal.runtime.Context` (`load`, `kernel`, `alloc`, `wrap`, `batch`/`begin`/`dispatch`/`commit`, `close`).

### v5: batches committed without waiting

```c
int32_t     idun_batch_commit(idun_context *, char *err, size_t err_len);              /* commit; no wait */
int32_t     idun_batch_wait(idun_context *, uint32_t keep, double *gpu_seconds, char *err, size_t err_len);
                        /* wait, oldest first, until at most `keep` committed batches are unwaited */
```

Batches run in commit order on the context's one queue, each after the one before it (the buffers' hazard tracking), so
results do not change; the host encodes batch k + 1 while the GPU runs batch k. `idun_batch_commit_wait` now waits for
every committed batch, and `idun_context_destroy` waits for them before freeing. Python: `Context.commit_async`,
`Context.wait(keep)`, `Context.batch(wait=False)`; `MetalDevice.submit` commits that way and waits, at a `read` or
`write`, for the committed batches that write or use the buffer (`DeviceRealMLP.train_steps_rows` alternates two
row-index and loss buffers).

### 3.1 The CUDA runtime (`idun.cuda.runtime`)

No native library: `ctypes` calls NVIDIA's driver library (`libcuda.so.1`, installed with the GPU driver) and NVRTC,
the runtime compiler, so the pure wheel runs CUDA and nothing is built on the machine. Linux only.

| part | what it does |
|---|---|
| libraries | the driver by its name; NVRTC from the `nvidia-cuda-nvrtc*` pip packages of the environment, newest first, then by its name; the `libnvrtc-builtins` library beside it is loaded first (NVRTC opens it by name at its first compilation, and a pip package's folder is not on the loader's path). `IDUN_CUDA_LIB` and `IDUN_NVRTC_LIB` name a file instead. A library that cannot be loaded, or lacks a function the binding calls, is refused by name |
| errors | every call's status is checked; a failure raises `IdunError` naming the call, the driver's name for the code and its text |
| `probe()` | the device (`IDUN_CUDA_DEVICE`, default 0): name, memory, compute capability, the limits of a launch, the driver's version (NVRTC's: `nvrtc_version()`) |
| `compile_ptx(source, arch)`, `compile_cubin(source, arch)` | NVRTC with fixed options: `--fmad=false` (a multiply and an add stay two roundings), `--std=c++17`; PTX for a virtual architecture (`compute_XY`: the compile checks), or a binary for a real one (`sm_XY`). A context loads the binary for `binary_architecture(capability)`, the newest NVRTC knows of the device's major version and not above its minor: a driver of the same CUDA major version loads it whatever NVRTC's minor version, where it refuses the PTX of an NVRTC newer than itself (CUDA_ERROR_UNSUPPORTED_PTX_VERSION). A compile error raises `KernelCompileError` with the log; warnings are returned |
| `Context` | one context, one module, one stream. Allocation zero-fills (driver memory is not zeroed); copies, fills and launches queue on the stream; `sync` waits and returns the window's GPU seconds (an event pair) |
| launch | the kernel's buffers and its parameter structure by value, marshalled to `cuLaunchKernel`. Dynamic shared memory above the default 48 KiB is opted in for the kernel, and refused above the device's maximum. A grid beyond the device's limits (65,535 blocks along y and z) raises `ValueError` |
| guard mode | `IDUN_CUDA_GUARD=1` (or `guard=True`): every allocation lies between two 1 KiB bands of a fixed byte, the host waits after every launch and checks the bands, and a kernel that wrote outside a buffer is named. Slow; for tests |
| `close` | frees every buffer, page-locked block, the module, the stream and the context; a failed release is reported after the others ran |

The tests drive the runtime through a fake driver and compiler (`tests/unit/cuda/fake_driver.py`): the same `ctypes`
signatures, device memory as host bytes, launches run as Python functions, and every call recorded and failable. On
a Mac they check the marshalling, the order on the stream, the transfers and the releases; the `nvrtc` tests compile
the kernel sources with the real compiler, and the `cuda` tests run on a GPU.

## 4. Metal ↔ CUDA mapping (for writing portable kernels)

| Metal | CUDA |
|---|---|
| threadgroup | block |
| `threadgroup` memory (32 KiB here) | `__shared__` memory |
| SIMD-group (32 lanes) | warp (32 lanes) |
| `simd_sum`, `simd_shuffle`, `simd_ballot` | `__reduce_add_sync`, `__shfl_sync`, `__ballot_sync` |
| `atomic_fetch_add_explicit(..., memory_order_relaxed)` on `atomic_uint` / `atomic_int` | `atomicAdd` on `unsigned` / `int` |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads()` |
| `[[thread_position_in_grid]]` | `blockIdx.x * blockDim.x + threadIdx.x` |
| `kernel void f(device T *x, constant P &p, ...)` | `extern "C" __global__ void f(T *x, P p)`: the buffers in the table's order, `const` when only read, then the parameters by value |
| `newLibraryWithSource` | NVRTC `nvrtcCompileProgram` |
| command buffer, encoder | stream, kernel launches |
| `GPUStartTime` / `GPUEndTime` | an event pair around each wait (`cuEventRecord`, `cuEventElapsedTime`) |
| `MTLResourceStorageModeShared` | device memory and a page-locked host mirror, copied only when needed (`CudaDevice`) |
| `#pragma METAL fp contract(off)` | `nvcc -fmad=false` / NVRTC `--fmad=false` |
