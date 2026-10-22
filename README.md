# OpenBLAS GEMM subset

Self-contained FP64 `cblas_dgemm` extracted from the local OpenBLAS
0.3.34.dev checkout. It computes `C = alpha * op(A) * op(B) + beta * C`
with row/column layouts, transpose modes, and padded leading dimensions.

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
`-pthread -lm`. The public header declares DGEMM and the thread count
setter/getter implemented by this subset.

The tests compare with an independent scalar calculation for both layouts,
all four transpose combinations, and one/four threads. They check odd and
blocked dimensions, alpha/beta scaling, beta zero with NaN-filled C, zero
dimensions/depth, unchanged inputs, and padding/guards. Linker wrapping
checks that the large cases dispatch four jobs through the OpenBLAS pool.

## Extracted implementation

`lib/` contains the 17 OpenBLAS C files, preserving their upstream code and
license notices. `level3.c` is included by the regular driver, which is
compiled for NN/NT/TN/TT; the resulting archive has 19 objects.
`include/openblas/` contains the required private headers and fixed build
configuration. The build does not read `../OpenBLAS` or need CMake/Fortran.

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
