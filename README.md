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

The tests compare with an independent int64 scalar calculation for both
layouts, all four transpose combinations, and one/four threads. They check
odd and blocked dimensions, alpha/beta scaling, beta zero with junk-filled
C, zero dimensions/depth, unchanged inputs, and padding/guards. Linker
wrapping checks that the large cases dispatch four jobs through the
OpenBLAS pool.

## Extracted implementation

`lib/` contains the 17 OpenBLAS C files, preserving their upstream code and
license notices. `level3.c` is included by the regular driver, which is
compiled for NN/NT/TN/TT; the resulting archive has 18 objects.
`include/openblas/` contains the required private headers and fixed build
configuration. The build does not read `../OpenBLAS` or need CMake/Fortran.

The library compiles the shared generic kernels once per INT8 type layer:
int8 inputs and packed buffers (`IFLOAT`/`XFLOAT`), int32 accumulation
(`FLOAT`), with the DGEMM blocking parameters. The thread queue mode is
tagged `BLAS_DOUBLE` so the server keeps sizing the worker scratch from
that shared blocking.

The configuration uses generic 2x2 C kernels, the existing simple Level 3
threading path, and a pthread pool capped at four threads. Small-matrix
kernels, GEMV forwarding, dynamic CPU dispatch, and LAPACK are disabled.
The thread cap is `MAX_CPU_NUMBER` in `include/openblas/config.h`.
Generic kernels prioritize a small extraction over tuned SIMD performance.

For a separate sanitizer build:

```sh
make test build_dir=build/asan \
    CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

`make clean` removes the selected `build_dir`. See `LICENSE` and
`LICENSE.gotoblas` for the OpenBLAS/GotoBLAS licenses.
