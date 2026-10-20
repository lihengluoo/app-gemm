#include "gemm_benchmark.h"
#include <stdlib.h>

// Compiler-analysis benchmark using the selected processor's actual kernel.
// The default square matrices need 8 GiB; keep them outside the static image.
int main(void)
{
    if (is_gemm_cpu_supported()) {
        const size_t elements = (size_t)benchmark_size * benchmark_size;
        // Numeral-extent static matrices keep every address inside one
        // provable object for the compiler analysis; the benchmark keeps no
        // heap allocations of its own.
        static int16_t matrix_a[benchmark_size * benchmark_size];
        static int16_t matrix_b[benchmark_size * benchmark_size];
        static int32_t matrix_c[benchmark_size * benchmark_size];
        int16_t *a = matrix_a;
        int16_t *b = matrix_b;
        int32_t *c = matrix_c;
        gemm_status status = gemm_no_memory;
        if (a != NULL && b != NULL && c != NULL) {
            for (size_t index = 0; index < elements; ++index) {
                a[index] = (int16_t)(index % 31);
                b[index] = (int16_t)(index % 17);
            }
            gemm_args args = get_default_gemm_args();
            args.m = benchmark_size;
            args.n = benchmark_size;
            args.k = benchmark_size;
            args.a = a;
            args.b = b;
            args.c = c;
            args.lda = benchmark_size;
            args.ldb = benchmark_size;
            args.ldc = benchmark_size;
            gemm_options options = get_default_gemm_options();
            options.thread_count = benchmark_threads;
            status = compute_gemm_i16_i32(&args, &options);
        }
        return (int)status;
    } else {
        // Refuse unsupported SIMD hosts before allocating the matrices.
        return 77;
    }
}
