#include "gemm_test.h"
#include "gemm_internal.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void require_gemm(bool condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "%s: %s\n", get_gemm_cpu_name(), message);
        exit(1);
    }
}

size_t get_matrix_index(matrix_order order, size_t stride, size_t row,
                        size_t column)
{
    return order == gemm_row_major ? row * stride + column :
                                     column * stride + row;
}

void check_gemm_shape(size_t m, size_t n, size_t k, size_t threads,
                      matrix_order order, matrix_transpose transpose_a,
                      matrix_transpose transpose_b, int32_t alpha, int32_t beta)
{
    const bool row_major = order == gemm_row_major;
    const bool ta = transpose_a == gemm_transpose;
    const bool tb = transpose_b == gemm_transpose;
    const size_t a_rows = ta ? k : m;
    const size_t a_columns = ta ? m : k;
    const size_t b_rows = tb ? n : k;
    const size_t b_columns = tb ? k : n;
    const size_t lda = (row_major ? a_columns : a_rows) + 3;
    const size_t ldb = (row_major ? b_columns : b_rows) + 5;
    const size_t ldc = (row_major ? n : m) + 7;
    const size_t guard = 9;
    const size_t a_bytes =
        ((row_major ? a_rows : a_columns) * lda + 2 * guard) * sizeof(int16_t);
    const size_t b_bytes =
        ((row_major ? b_rows : b_columns) * ldb + 2 * guard) * sizeof(int16_t);
    const size_t c_bytes =
        ((row_major ? m : n) * ldc + 2 * guard) * sizeof(int32_t);
    int16_t *a = malloc(a_bytes);
    int16_t *b = malloc(b_bytes);
    int32_t *c = malloc(c_bytes);
    int16_t *a_before = malloc(a_bytes);
    int16_t *b_before = malloc(b_bytes);
    int32_t *expected = malloc(c_bytes);
    require_gemm(a != NULL && b != NULL && c != NULL && a_before != NULL &&
                     b_before != NULL && expected != NULL,
                 "test allocation failed");
    for (size_t index = 0; index < a_bytes / sizeof(*a); ++index) {
        a[index] = (int16_t)((int)(index * 31 % 255) - 127);
    }
    for (size_t index = 0; index < b_bytes / sizeof(*b); ++index) {
        b[index] = (int16_t)((int)(index * 17 % 255) - 127);
    }
    for (size_t index = 0; index < c_bytes / sizeof(*c); ++index) {
        c[index] = (int32_t)(index % 29) - 14;
    }
    memcpy(a_before, a, a_bytes);
    memcpy(b_before, b, b_bytes);
    memcpy(expected, c, c_bytes);
    for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
            int64_t sum = 0;
            for (size_t inner = 0; inner < k; ++inner) {
                const size_t ai = get_matrix_index(order, lda, ta ? inner : row,
                                                   ta ? row : inner);
                const size_t bi = get_matrix_index(
                    order, ldb, tb ? column : inner, tb ? inner : column);
                sum += (int64_t)a[guard + ai] * b[guard + bi];
            }
            const size_t ci = guard + get_matrix_index(order, ldc, row, column);
            const uint32_t bits =
                (uint32_t)((int64_t)alpha * sum + (int64_t)beta * expected[ci]);
            memcpy(&expected[ci], &bits, sizeof(bits));
        }
    }
    const gemm_args args = {order, transpose_a, transpose_b, m,         n,
                            k,     a + guard,   b + guard,   c + guard, lda,
                            ldb,   ldc,         alpha,       beta};
    gemm_options options = get_default_gemm_options();
    options.thread_count = threads;
    options.mc = 5;
    options.nc = 2 * get_gemm_column_block();
    options.kc = 4;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_success,
                 "compute failed");
    require_gemm(memcmp(c, expected, c_bytes) == 0,
                 "logical output, padding, or guard mismatch");
    require_gemm(memcmp(a, a_before, a_bytes) == 0 &&
                     memcmp(b, b_before, b_bytes) == 0,
                 "input modified");
    // Exercise another call, not a hidden cached output or mutable plan.
    options.mc = 7;
    options.kc = 6;
    options.stream_output = true;
    gemm_args repeated = args;
    repeated.beta = 0;
    require_gemm(compute_gemm_i16_i32(&repeated, &options) == gemm_success,
                 "repeat failed");
    if (beta == 0) {
        require_gemm(memcmp(c, expected, c_bytes) == 0,
                     "repeated beta-zero call changed its answer");
    }
    free(expected);
    free(b_before);
    free(a_before);
    free(c);
    free(b);
    free(a);
}

void check_gemm_arguments(void)
{
    gemm_args args = get_default_gemm_args();
    gemm_options options = get_default_gemm_options();
    require_gemm(compute_gemm_i16_i32(NULL, &options) == gemm_invalid_argument,
                 "null args accepted");
    require_gemm(compute_gemm_i16_i32(&args, NULL) == gemm_invalid_argument,
                 "null options accepted");
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_success,
                 "empty matrix rejected");
    options.thread_count = 0;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "zero threads accepted");
    options.thread_count = max_gemm_threads + 1;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "too many threads accepted");
    options.thread_count = 1;
    options.kc = 3;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "odd kc accepted");
    options.kc = 2;
    options.nc = 1;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "misaligned nc accepted");
    options.nc = 512;
    args.order = (matrix_order)9;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "bad layout accepted");
    args.order = gemm_row_major;
    int32_t c[4]; // beta zero must not read these uninitialized elements.
    args.m = 2;
    args.n = 2;
    args.ldc = 2;
    args.c = c;
    args.k = 17;
    args.alpha = 0;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_success,
                 "alpha-zero reads inputs");
    for (size_t index = 0; index < 4; ++index) {
        require_gemm(c[index] == 0, "beta-zero did not overwrite C");
    }
    c[0] = 7;
    args.k = 0;
    args.alpha = 1;
    args.beta = -3;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_success &&
                     c[0] == -21,
                 "k-zero scaling failed");
    args.ldc = 1;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "short ldc accepted");
    args.ldc = SIZE_MAX;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_size_overflow,
                 "C size overflow accepted");
    args.ldc = 2;
    args.k = 2;
    args.lda = 2;
    args.ldb = 2;
    int16_t b[4] = {0};
    args.a = (const int16_t *)c;
    args.b = b;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "overlapping C/A accepted");
    args.a = b;
    args.lda = 1;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_invalid_argument,
                 "short lda accepted");
    args.lda = SIZE_MAX;
    require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_size_overflow,
                 "A size overflow accepted");
}

void check_gemm_partitions(void)
{
    for (size_t rows = 0; rows < 40; ++rows) {
        for (size_t count = 1; count <= max_gemm_threads; ++count) {
            size_t cursor = 0;
            for (size_t index = 0; index < count; ++index) {
                const row_partition partition =
                    partition_rows(rows, count, index);
                require_gemm(partition.begin == cursor &&
                                 partition.end >= cursor &&
                                 partition.end <= rows,
                             "invalid row partition");
                cursor = partition.end;
            }
            require_gemm(cursor == rows, "row partition omitted rows");
        }
    }
    const row_partition last = partition_rows(SIZE_MAX, 7, 6);
    require_gemm(last.end == SIZE_MAX, "partition arithmetic overflow");
}

void check_gemm_wrap(void)
{
    int16_t a[4 * 4];
    int16_t b[4 * gemm_columns];
    int32_t c[4 * gemm_columns];
    uint32_t expected[4 * gemm_columns];
    const int32_t coefficients[][2] = {
        {1, 0}, {1, 1}, {3, -7}, {INT32_MIN, INT32_MAX}};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t inner = 0; inner < 4; ++inner) {
            a[row * 4 + inner] = inner < 2 ? INT16_MIN :
                                             (int16_t)(INT16_MAX - row);
        }
    }
    for (size_t inner = 0; inner < 4; ++inner) {
        for (size_t column = 0; column < gemm_columns; ++column) {
            b[inner * gemm_columns + column] =
                inner < 2 ? INT16_MIN : (int16_t)(INT16_MAX - column % 3);
        }
    }
    gemm_args args = get_default_gemm_args();
    args.k = args.lda = 4;
    args.ldb = args.ldc = gemm_columns;
    args.a = a;
    args.b = b;
    args.c = c;
    gemm_options options = get_default_gemm_options();
    options.kc = 2;
    // Include direct full-tile accumulation and bounded tail stores. The first
    // K pair overflows INT32; odd coefficients keep arithmetic errors visible.
    for (unsigned tail = 0; tail < 2; ++tail) {
        args.m = tail == 0 ? 4 : 1;
        args.n = tail == 0 ? gemm_columns : 1;
        for (size_t test = 0;
             test < sizeof(coefficients) / sizeof(coefficients[0]); ++test) {
            args.alpha = coefficients[test][0];
            args.beta = coefficients[test][1];
            for (size_t index = 0; index < 4 * gemm_columns; ++index) {
                c[index] = INT32_MIN;
                expected[index] = (uint32_t)c[index];
            }
            for (size_t row = 0; row < args.m; ++row) {
                for (size_t column = 0; column < args.n; ++column) {
                    uint32_t sum = 0;
                    for (size_t inner = 0; inner < args.k; ++inner) {
                        sum += (uint32_t)((int32_t)a[row * args.lda + inner] *
                                          b[inner * args.ldb + column]);
                    }
                    expected[row * args.ldc + column] =
                        (uint32_t)args.alpha * sum +
                        (uint32_t)args.beta * (uint32_t)INT32_MIN;
                }
            }
            require_gemm(compute_gemm_i16_i32(&args, &options) == gemm_success,
                         "full-range compute failed");
            require_gemm(memcmp(c, expected, sizeof(c)) == 0,
                         "full-range arithmetic differs from modulo reference");
        }
    }
}

void check_gemm_stream(void)
{
    _Alignas(64) int16_t a[8];
    _Alignas(64) int16_t b[2 * gemm_columns];
    int32_t *c = aligned_alloc(64, 4 * gemm_columns * sizeof(int32_t));
    require_gemm(c != NULL, "stream test allocation failed");
    for (size_t row = 0; row < 4; ++row) {
        a[2 * row] = (int16_t)(row + 1);
        a[2 * row + 1] = 2;
    }
    for (size_t column = 0; column < gemm_columns; ++column) {
        b[2 * column] = 3;
        b[2 * column + 1] = 4;
    }
    for (int32_t beta = 0; beta <= 1; ++beta) {
        const gemm_tile tile = {
            a, b, c, 2, gemm_columns, 1, 4, gemm_columns, 1, beta, true};
        compute_gemm_tile(&tile);
        _mm_sfence();
        for (size_t row = 0; row < 4; ++row) {
            for (size_t column = 0; column < gemm_columns; ++column) {
                const int32_t expected =
                    (int32_t)(3 * (row + 1) + 8) * (beta + 1);
                require_gemm(c[row * gemm_columns + column] == expected,
                             "streaming full tile failed");
            }
        }
    }
    free(c);
}

void *check_gemm_concurrent(void *args)
{
    (void)args;
    check_gemm_shape(13, 35, 19, 3, gemm_row_major, gemm_transpose,
                     gemm_no_transpose, -2, 3);
    return NULL;
}

int main(void)
{
    check_gemm_arguments();
    check_gemm_partitions();
    if (!is_gemm_cpu_supported()) {
        const int16_t a = 1;
        const int16_t b = 1;
        int32_t c = 0;
        gemm_args args = get_default_gemm_args();
        args.m = args.n = args.k = args.lda = args.ldb = args.ldc = 1;
        args.a = &a;
        args.b = &b;
        args.c = &c;
        gemm_options options = get_default_gemm_options();
        require_gemm(compute_gemm_i16_i32(&args, &options) ==
                         gemm_unsupported_cpu,
                     "unsupported CPU was not refused");
        printf("%s: API/partition checks passed; SIMD host unavailable\n",
               get_gemm_cpu_name());
        return 0;
    }
    const size_t shapes[][3] = {{0, 3, 5},     {3, 0, 5},    {3, 5, 0},
                                {1, 1, 1},     {2, 7, 3},    {3, 8, 9},
                                {4, 9, 7},     {5, 15, 3},   {7, 16, 5},
                                {9, 17, 9},    {11, 31, 7},  {13, 32, 11},
                                {17, 33, 13},  {33, 65, 17}, {9, 35, 513},
                                {1, 8, 133144}};
    const size_t thread_counts[] = {1, 3, 16};
    size_t checks = 0;
    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]);
         ++shape) {
        for (unsigned order = 0; order < 2; ++order) {
            for (unsigned ta = 0; ta < 2; ++ta) {
                for (unsigned tb = 0; tb < 2; ++tb) {
                    for (size_t thread = 0; thread < 3; ++thread) {
                        const int32_t alpha = checks % 3 == 0 ? -2 : 1;
                        const int32_t beta = (int32_t)(checks % 4) - 1;
                        check_gemm_shape(shapes[shape][0], shapes[shape][1],
                                         shapes[shape][2],
                                         thread_counts[thread],
                                         (matrix_order)order,
                                         (matrix_transpose)ta,
                                         (matrix_transpose)tb, alpha, beta);
                        ++checks;
                    }
                }
            }
        }
    }
    check_gemm_wrap();
    check_gemm_stream();
    pthread_t first;
    pthread_t second;
    require_gemm(pthread_create(&first, NULL, check_gemm_concurrent, NULL) == 0,
                 "concurrent first create failed");
    require_gemm(pthread_create(&second, NULL, check_gemm_concurrent, NULL) ==
                     0,
                 "concurrent second create failed");
    require_gemm(pthread_join(first, NULL) == 0 &&
                     pthread_join(second, NULL) == 0,
                 "concurrent join failed");
    printf("%s: %zu runtime shapes/layouts/transposes/coefficients passed\n",
           get_gemm_cpu_name(), checks);
    return 0;
}
