/* Minimal i8gemm performance benchmark, after the archived Strassen
 * bench main: fill constant-sized matrices, time the multiplication
 * with the monotonic clock, and check one sampled result against an
 * inline reference dot product. */

#define _POSIX_C_SOURCE 200809L

#include "bench.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "cblas.h"

#define row_count BENCH_MATRIX_SIZE
#define column_count BENCH_MATRIX_SIZE
#define depth_count BENCH_K_DEPTH

static int8_t matrix_a[row_count * depth_count];
static int8_t matrix_b[depth_count * column_count];
static int32_t matrix_c[row_count * column_count];

static uint64_t get_time_ms(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static void prepare_input(void) {
    for (int64_t row = 0; row < row_count; row++)
        for (int64_t k = 0; k < depth_count; k++)
            matrix_a[row * depth_count + k] = (int8_t)(((row * 7 + k * 3) % 11) - 5);
    for (int64_t k = 0; k < depth_count; k++)
        for (int64_t col = 0; col < column_count; col++)
            matrix_b[k * column_count + col] = (int8_t)(((k * 5 + col * 3) % 13) - 6);
}

static int verify_result(void) {
    const int64_t row = row_count - 1;
    const int64_t col = column_count - 1;
    int64_t expect = 0;

    for (int64_t k = 0; k < depth_count; k++)
        expect += matrix_a[row * depth_count + k] * matrix_b[k * column_count + col];

    if ((int64_t)matrix_c[row * column_count + col] != expect) {
        printf("FAIL sample C[%lld][%lld] = %d, expected %lld\n",
               (long long)row, (long long)col,
               matrix_c[row * column_count + col], (long long)expect);
        return 1;
    }
    printf("sample C[%lld][%lld] = %d\n",
           (long long)row, (long long)col, matrix_c[row * column_count + col]);
    return 0;
}

int main(void) {
    const int64_t ops_per_call = 2 * (int64_t)row_count * column_count * depth_count;
    uint64_t start_ms;

    prepare_input();

    openblas_set_num_threads(BENCH_THREAD_COUNT);
    printf("i8gemm benchmark: %d x %d x %d, %d threads, %d timed calls\n",
           row_count, column_count, depth_count,
           BENCH_THREAD_COUNT, BENCH_ITERATION_COUNT);

    /* Warm-up call, untimed. */
    cblas_i8gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 row_count, column_count, depth_count, 1,
                 matrix_a, depth_count, matrix_b, column_count,
                 0, matrix_c, column_count);

    start_ms = get_time_ms();
    for (int call = 0; call < BENCH_ITERATION_COUNT; call++)
        cblas_i8gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     row_count, column_count, depth_count, 1,
                     matrix_a, depth_count, matrix_b, column_count,
                     0, matrix_c, column_count);
    const uint64_t elapsed_ms = get_time_ms() - start_ms;

    /* One sampled result, checked against an inline reference dot. */
    if (verify_result() != 0) return 1;

    printf("elapsed: %llu ms for %d calls (%.1f ms per call)\n",
           (unsigned long long)elapsed_ms, BENCH_ITERATION_COUNT,
           (double)elapsed_ms / BENCH_ITERATION_COUNT);
    printf("throughput: %.3f GOPS\n",
           ((double)ops_per_call * BENCH_ITERATION_COUNT) /
               ((double)elapsed_ms / 1000.0) / 1e9);
    return 0;
}
