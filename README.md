# GEMM

Pure C11, pthread-based INT16 × INT16 → INT32 GEMM, organized around an
OpenBLAS-style blocked driver, packing, and processor-specific kernels.
The API supports runtime dimensions, strides, layouts, transposes,
coefficients, and row-partitioned workers. Arithmetic is modulo 2^32;
`beta=0` does not read old C. No C++ runtime is required.

## Layout

- `include/gemm.h`: all types, constants, and function declarations.
- `lib/`: driver, packing, threading, scalar and SIMD kernels.
- `tests/`: correctness checks and allocation/pthread failure injection.
- `gemm_benchmark.c`: the single compiler-analysis entry point, not a
  throughput timer. It calls the real library with M=N=K=32768 and 128
  workers by default, without libc argument parsing. Change the build-time
  settings with `gemm_size` (sets M, N and K together) and `gemm_threads`.

## Build and test

Requires GNU Make, a C11 Clang/GCC compiler, and POSIX threads on x86.

```sh
make -j
make test
./build/scalar/gemm_benchmark

# Only build the selected analysis executable, without tests:
make benchmark CC=clang gemm_backend=6780e_avx

# Build matching native and LLVM versions of a square workload:
make benchmark ir gemm_size=32768 gemm_threads=128 gemm_backend=4509y_avx

# AddressSanitizer and UBSan in a separate build directory:
make test CC=clang build_dir=build/asan \
  CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
```

`gemm_backend` selects the benchmark's kernel: `scalar` (4×32, default),
`6780e_avx` (AVX-VNNI, 4×8), or `4509y_avx` (AVX-512 VNNI, 4×32).
`gemm_size` must be positive; `gemm_threads` must be between 1 and 128.
Changing either setting rebuilds the benchmark automatically. The C library
API and the small correctness tests keep their runtime dimensions.
Each backend's library, objects, and executables live in `build/<backend>/`;
`build_dir` changes the build directory. `make` builds the selected benchmark
and tests for all three backends. `make test` runs those correctness checks
and Linux allocation/pthread failure checks. Run the benchmark explicitly:
its matrices need 8 GiB, plus 2 GiB for packed B and 32 MiB for worker A
panels with the default blocking options, excluding thread/runtime overhead.
Each worker owns 256 output rows. Unsupported SIMD hosts return 77 from the
benchmark before allocating matrices. The tests check argument validation
and CPU refusal on such hosts; native SIMD arithmetic must be checked on
supported hardware.
Native builds disable automatic vectorization but retain explicit SIMD
intrinsics. `make clean` removes the build directory, including IR captures.

## Generate LLVM IR

`build.sh` requires Bash, Make, `file`, and `wllvm` with `extract-bc`
beside it. Use Clang, `llvm-link`, and `opt` from the same LLVM installation
as the analyzer:

```sh
./build.sh --clang /path/to/llvm21/bin/clang \
  --wllvm /path/to/wllvm/bin/wllvm --backend scalar

# From a Delphi checkout, use its installed LLVM and wllvm:
make ir gemm_backend=scalar
# Optional toolchain overrides:
make ir llvm_root=/path/to/llvm21 wllvm=/path/to/wllvm/bin/wllvm
```

The shell script builds the selected benchmark through the Makefile with
`wllvm` in `build/ir/wllvm/<backend>/`, extracts the linked program's bitcode
to `build/ir/<backend>.bc`, and writes verified IR to `build/ir/<backend>.ll`.
The capture build is separate from native builds; no output goes to a
shared IR directory. `--backend` also accepts `6780e_avx` and `4509y_avx`;
`--output-dir` changes the capture directory (`ir_dir` for `make ir`).
Direct script calls accept `--size 32768 --threads 128`; `make ir` forwards
the same settings used by `make benchmark`.
The script uses `-O1 -g`, keeps IR names, disables automatic vectorization
and loop unrolling, and removes unused globals before verifying the linked
IR. Explicit SIMD intrinsics are retained. The default backend is `scalar`;
`--clang` defaults to `clang` on `PATH`, and `--wllvm` defaults to `WLLVM_CC`
or `wllvm` on `PATH`.
