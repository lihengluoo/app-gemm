# OpenBLAS integer GEMM subset

Self-contained `cblas_i8gemm` extracted from the local OpenBLAS
0.3.34.dev checkout. It computes `C = alpha * op(A) * op(B) + beta * C`
with int8 inputs A and B, int32 accumulation for C, alpha, and beta,
row/column layouts, transpose modes, and padded leading dimensions.
Accumulation is exact while `|alpha| * k * 127 * 127 + |beta| * |C|`
stays below 2^31; exceeding it is signed overflow.

## Build and check

Requires Linux x86-64, a C11 GCC/Clang compiler, GNU Make, and a linker
supporting `--wrap`. The build uses POSIX threads and the C math library.

```sh
make -j
make test
# Build and run the same correctness executable:
make run
```

The outputs are `build/libopenblas_gemm.a` and `build/gemm_check`.
Applications include `include/cblas.h` and link the archive with
`-pthread -lm`. The public header declares I8GEMM and the thread count
setter/getter implemented by this subset.

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
make -C applications/gemm test kernel_backend=4509y
```

The AVX-512 flags (`-mavx512f -mavx512bw -mavx512vl -mavx512vnni`) apply
only to the backend objects; the rest of the library stays baseline
x86-64. A constructor aborts with exit 86 if the host lacks the required
extensions, and `make test` skips with exit 77 on such hosts, so building
the 4509y backend on another machine is safe.

The tests compare with an independent int64 scalar calculation for both
layouts, all four transpose combinations, and one/four threads. They check
odd and blocked dimensions, alpha/beta scaling, beta zero with junk-filled
C, zero dimensions/depth, unchanged inputs, and padding/guards. Linker
wrapping checks that the large cases dispatch four jobs through the
parallel driver.

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
disabled. Threads are user selected through `OPENBLAS_NUM_THREADS` or
`openblas_set_num_threads`, capped at `MAX_CPU_NUMBER` (144) in
`include/openblas/config.h`.
Generic kernels prioritize a small extraction over tuned SIMD performance.

For a separate sanitizer build:

```sh
make test build_dir=build/asan \
    CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

`make clean` removes the selected `build_dir`. See `LICENSE` and
`LICENSE.gotoblas` for the OpenBLAS/GotoBLAS licenses.
