#!/usr/bin/env bash
# Capture the selected GEMM benchmark through its Makefile build and wllvm.
set -euo pipefail

app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
clang=clang
wllvm="${WLLVM_CC:-wllvm}"
backend=scalar
gemm_size=4096
gemm_threads=128
output_dir="$app_dir/build/ir"

fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

print_usage() {
    printf 'Usage: %s [--clang PATH] [--wllvm PATH] [--backend NAME] [--output-dir DIR] [--size N] [--threads N]\n' "$0"
    printf 'Backends: scalar (default), 6780e_avx, 4509y_avx\n'
    printf 'Workload: M=N=K=4096, threads=128 by default\n'
    printf 'Defaults: clang and wllvm on PATH (or WLLVM_CC); output: %s/build/ir\n' "$app_dir"
}

find_executable() {
    local path
    path="$(command -v -- "$1")" || fail "Executable not found: $1"
    [[ -f "$path" && -x "$path" ]] || fail "Not an executable file: $path"
    printf '%s/%s\n' "$(cd -- "$(dirname -- "$path")" && pwd)" "$(basename -- "$path")"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clang|--wllvm|--backend|--output-dir|--size|--threads)
            [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires an argument"
            case "$1" in
                --clang) clang="$2" ;;
                --wllvm) wllvm="$2" ;;
                --backend) backend="$2" ;;
                --output-dir) output_dir="$2" ;;
                --size) gemm_size="$2" ;;
                --threads) gemm_threads="$2" ;;
            esac
            shift 2
            ;;
        -h|--help) print_usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

[[ "$gemm_size" =~ ^[1-9][0-9]*$ ]] || fail '--size must be a positive decimal integer'
[[ "$gemm_threads" =~ ^[1-9][0-9]*$ ]] || fail '--threads must be a positive decimal integer'

case "$backend" in
    scalar|6780e_avx|4509y_avx) ;;
    *) fail "Unknown GEMM backend: $backend" ;;
esac
clang="$(find_executable "$clang")"
wllvm="$(find_executable "$wllvm")"
llvm_dir="$(dirname -- "$clang")"
wllvm_dir="$(dirname -- "$wllvm")"
for tool in make file; do
    command -v "$tool" >/dev/null || fail "Required executable not found: $tool"
done
for tool in "$llvm_dir/llvm-link" "$llvm_dir/opt" "$wllvm_dir/extract-bc"; do
    [[ -x "$tool" ]] || fail "Required executable not found: $tool"
done

mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
export LLVM_COMPILER=clang LLVM_CC_NAME="${clang##*/}" LLVM_COMPILER_PATH="$llvm_dir"
# Use explicit capture flags, independently of an invoking make or configure.
unset WLLVM_CONFIGURE_ONLY MAKEFLAGS MFLAGS MAKEOVERRIDES
flags='-O1 -g -fno-discard-value-names -fno-vectorize -fno-slp-vectorize -fno-unroll-loops'
# Always rebuild to capture changed compiler options and embedded bitcode paths.
make -C "$app_dir" --always-make "build_dir=$output_dir/wllvm" \
    "CC=$wllvm" "CFLAGS=$flags" "gemm_backend=$backend" \
    "gemm_size=$gemm_size" "gemm_threads=$gemm_threads" benchmark
linked="$output_dir/$backend.bc"
ir="$output_dir/$backend.ll"
"$wllvm_dir/extract-bc" --linker "$llvm_dir/llvm-link" \
    --output "$linked" "$output_dir/wllvm/$backend/gemm_benchmark"
"$llvm_dir/opt" -passes=globaldce,verify -S "$linked" -o "$ir"
printf '%s\n' "$ir"
