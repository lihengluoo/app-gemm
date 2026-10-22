/* i8gemm benchmark tunables; the Makefile passes these as -D flags, so
 * the #ifndef defaults only apply when bench.c is compiled standalone. */

#ifndef GEMM_BENCH_H
#define GEMM_BENCH_H

#ifndef BENCH_MATRIX_SIZE
#define BENCH_MATRIX_SIZE 512
#endif

#ifndef BENCH_K_DEPTH
#define BENCH_K_DEPTH BENCH_MATRIX_SIZE
#endif

#ifndef BENCH_ITERATION_COUNT
#define BENCH_ITERATION_COUNT 5
#endif

#ifndef BENCH_THREAD_COUNT
#define BENCH_THREAD_COUNT 4
#endif

#endif /* GEMM_BENCH_H */
