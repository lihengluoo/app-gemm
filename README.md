# OpenBLAS integer GEMM subset

Self-contained `cblas_i8gemm` extracted from the local OpenBLAS
0.3.34.dev checkout. It computes `C = alpha * op(A) * op(B) + beta * C`
with int8 inputs A and B, int32 accumulation for C, alpha, and beta,
row/column layouts, transpose modes, and padded leading dimensions.
Accumulation is exact while `|alpha| * k * 127 * 127 + |beta| * |C|`
stays below 2^31; exceeding it is signed overflow.

## Build and benchmark

Requires Linux x86-64, a C11 GCC/Clang compiler, and GNU Make. The build
uses POSIX threads.

```sh
make -j          # build the library and the benchmark
make run         # build and run the i8gemm benchmark
```

The outputs are `build/libopenblas_gemm.a` and `build/gemm_bench`.
Applications include `include/cblas.h` and link the archive with
`-pthread`. The public header declares I8GEMM and the thread count
setter/getter implemented by this subset.

`bench.c`/`bench.h` (modeled on `applications/archive/gemm.c`) time
`BENCH_ITERATION_COUNT` warm calls over statically allocated
`BENCH_MATRIX_SIZE` matrices after one untimed warm-up, report elapsed
time and throughput in GOPS (`2*M*N*K` int8 operations per second), and
verify sampled results against an independent int64 reference dot
product. The sizes, iteration count, and thread count are compile-time
defines (`BENCH_MATRIX_SIZE`, `BENCH_K_DEPTH`, `BENCH_ITERATION_COUNT`,
`BENCH_THREAD_COUNT`); pass them as make variables to override, e.g.
`make run BENCH_MATRIX_SIZE=1024 BENCH_THREAD_COUNT=32` (a change
rebuilds the benchmark). Threads are set programmatically via
`openblas_set_num_threads`. `clock_gettime` is the only external call
the benchmark adds beyond the library's minimal contract.

## Kernel backends

`kernel_backend` selects the compute kernel at build time:

- `generic` (default): portable scalar 2x2 kernel; runs on any x86-64 host.
- `4509y`: AVX-512 VNNI kernel for Intel 4509Y class CPUs. Each int8 value
  is packed as a sign-extended int16 lane so `vpdpwssd` multiplies two k
  steps of eight rows against 32 columns per tile, with masked stores for
  column tails and a scalar row tail. Packed panels hold one 16-bit lane
  per element, so the driver spaces B panels by `2 * ceil(k / 2)` bytes
  (`pad_min_l`).

```sh
make -C applications/gemm run kernel_backend=4509y
```

The AVX-512 flags (`-mavx512f -mavx512bw -mavx512vl -mavx512vnni`) apply
only to the backend objects; the rest of the library stays baseline
x86-64. A constructor aborts with exit 86 if the host lacks the required
extensions, so building the 4509y backend on another machine is safe;
running its benchmark needs an AVX-512 VNNI host.

The benchmark verifies sampled results of every backend against an
independent int64 scalar reference.

## Extracted implementation

`lib/` contains the 17 OpenBLAS C files, preserving their upstream code and
license notices. `level3.c` is included by the regular driver, which is
compiled for NN/NT/TN/TT; the resulting archive has 18 objects.
`include/openblas/` contains the required private headers and fixed build
configuration. The build does not read `../OpenBLAS` or need CMake/Fortran.

The library compiles the shared generic kernels once per INT8 type layer:
int8 inputs and packed buffers (`IFLOAT`/`XFLOAT`), int32 accumulation
(`FLOAT`), with the DGEMM blocking parameters. The thread queue mode is
tagged `BLAS_DOUBLE` so the worker scratch keeps its size from that
shared blocking.

The configuration uses generic 2x2 C kernels (or the selected SIMD
backend), the existing simple Level 3 threading path, and one pthread per
job created and joined by each parallel call. Small-matrix kernels, GEMV
forwarding, dynamic CPU dispatch, LAPACK, and a persistent thread pool are
disabled. Threads are user selected through
`openblas_set_num_threads`, capped at `MAX_CPU_NUMBER` (144) in
`include/openblas/config.h`; there is no environment variable or CPU
probing, keeping the library's external surface at
`pthread_create`/`pthread_join`, `pthread_mutex_lock`/`unlock`,
`aligned_alloc`/`free`, `abort`, and `printf`.
Generic kernels prioritize a small extraction over tuned SIMD performance.

For a separate sanitizer build:

```sh
make run build_dir=build/asan \
    CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

`make clean` removes the selected `build_dir`. See `LICENSE` and
`LICENSE.gotoblas` for the OpenBLAS/GotoBLAS licenses.
