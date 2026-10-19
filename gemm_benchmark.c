#include "gemm.h"

// Compiler-analysis benchmark, not a throughput timer. Real backing arrays
// retain a runtime row count and the selected processor's actual kernel.
// argv is unused so libc argument parsing is not part of the analysis.
int main(int argc, char **argv)
{
    (void)argv;
    static int16_t a[9 * 17];
    static int16_t b[17 * 35];
    static int32_t c[9 * 35];
    const size_t m = (size_t)((unsigned)argc % 9) + 1;
    for (size_t index = 0; index < 9 * 17; ++index) {
        a[index] = (int16_t)(index % 31);
    }
    for (size_t index = 0; index < 17 * 35; ++index) {
        b[index] = (int16_t)(index % 17);
    }
    gemm_args args = get_default_gemm_args();
    args.m = m;
    args.n = 35;
    args.k = 17;
    args.a = a;
    args.b = b;
    args.c = c;
    args.lda = 17;
    args.ldb = 35;
    args.ldc = 35;
    gemm_options options = get_default_gemm_options();
    options.thread_count = 3;
    const gemm_status status = compute_gemm_i16_i32(&args, &options);
    // CTest marks unsupported SIMD hosts as skipped, not arithmetic passes.
    return status == gemm_unsupported_cpu ? 77 : (int)status;
}
