# GEMM

Pure C11, pthread-based INT16 × INT16 → INT32 GEMM, organized around an
OpenBLAS-style blocked driver, packing, and processor-specific kernels.
The API supports runtime dimensions, strides, layouts, transposes,
coefficients, and row-partitioned workers. Arithmetic is modulo 2^32;
`beta=0` does not read old C. No C++ runtime is required.

## Layout

- `include/`: public API and internal declarations.
- `lib/`: driver, packing, threading, scalar and SIMD kernels.
- `tests/`: correctness checks and allocation/pthread failure injection.
- `gemm_benchmark.c`: the single compiler-analysis entry point, not a
  throughput timer. It calls the real library with runtime M, fixed N/K,
  and three requested workers, without libc argument parsing.

## Build and test

Requires CMake, a C11 Clang/GCC compiler, and POSIX threads on x86.

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang -Dgemm_backend=scalar
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/gemm_benchmark
```

`gemm_backend` selects the one `gemm_benchmark` executable's kernel:
`scalar` (4×32), `6780e_avx` (AVX-VNNI, 4×8), or `4509y_avx`
(AVX-512 VNNI, 4×32). Unsupported SIMD hosts return 77 from the benchmark
and CTest skips it. Native SIMD arithmetic must be checked on supported
hardware. Use `-DBUILD_TESTING=OFF` and build `--target gemm_benchmark`
when only the analysis program is needed.
