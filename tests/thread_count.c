#include "thread_count.h"

#include <stdint.h>
#include <stdio.h>

static int created_thread_count;
static int joined_thread_count;

int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                         void *(*start_routine)(void *), void *arg)
{
    const int result = __real_pthread_create(thread, attr, start_routine, arg);

    if (result == 0) created_thread_count++;
    return result;
}

int __wrap_pthread_join(pthread_t thread, void **result)
{
    const int status = __real_pthread_join(thread, result);

    if (status == 0) joined_thread_count++;
    return status;
}

static int check_gemm_thread_count(enum CBLAS_ORDER order, int row_count,
                                  int column_count, int configured_thread_count)
{
    int8_t matrix_a[test_row_count * test_depth_count];
    int8_t matrix_b[test_depth_count * test_column_count];
    int32_t matrix_c[test_row_count * test_column_count];
    const int row_major = order == CblasRowMajor;
    const int lda = row_major ? test_depth_count : row_count;
    const int ldb = row_major ? column_count : test_depth_count;
    const int ldc = row_major ? column_count : row_count;
    const int partition_count = row_major ? row_count : column_count;
    const int thread_count = configured_thread_count < partition_count
                                 ? configured_thread_count : partition_count;
    const int expected_worker_count = row_count > 0 && column_count > 0
                                          ? thread_count - 1 : 0;

    for (int index = 0; index < row_count * test_depth_count; index++)
        matrix_a[index] = (int8_t)(index % 13 - 6);
    for (int index = 0; index < test_depth_count * column_count; index++)
        matrix_b[index] = (int8_t)(index % 17 - 8);
    for (int index = 0; index < row_count * column_count; index++)
        matrix_c[index] = -1000000;

    created_thread_count = 0;
    joined_thread_count = 0;
    cblas_i8gemm(order, CblasNoTrans, CblasNoTrans,
                 row_count, column_count, test_depth_count, 1,
                 matrix_a, lda, matrix_b, ldb, 0, matrix_c, ldc);

    if (created_thread_count != expected_worker_count ||
        joined_thread_count != expected_worker_count) {
        printf("FAIL order=%d, %d x %d, configured=%d: created=%d, joined=%d, expected=%d\n",
               (int)order, row_count, column_count, configured_thread_count,
               created_thread_count, joined_thread_count, expected_worker_count);
        return 1;
    }
    if (openblas_get_num_threads() != configured_thread_count) {
        printf("FAIL configured thread count changed\n");
        return 1;
    }

    for (int row = 0; row < row_count; row++) {
        for (int col = 0; col < column_count; col++) {
            int64_t expect = 0;

            for (int k = 0; k < test_depth_count; k++) {
                const int a_index = row_major ? row * lda + k : row + k * lda;
                const int b_index = row_major ? k * ldb + col : k + col * ldb;

                expect += matrix_a[a_index] * matrix_b[b_index];
            }
            const int c_index = row_major ? row * ldc + col : row + col * ldc;
            if ((int64_t)matrix_c[c_index] != expect) {
                printf("FAIL order=%d, configured=%d: C[%d][%d]=%d, expected=%lld\n",
                       (int)order, configured_thread_count, row, col,
                       matrix_c[c_index], (long long)expect);
                return 1;
            }
        }
    }
    return 0;
}

int main(void)
{
    if (check_gemm_thread_count(CblasRowMajor, test_row_count, test_column_count,
                               default_thread_count) != 0)
        return 1;

    for (size_t index = 0;
         index < sizeof(configured_thread_counts) / sizeof(configured_thread_counts[0]);
         index++) {
        const int thread_count = configured_thread_counts[index];

        openblas_set_num_threads(thread_count);
        for (size_t order_index = 0;
             order_index < sizeof(matrix_orders) / sizeof(matrix_orders[0]);
             order_index++) {
            const enum CBLAS_ORDER order = matrix_orders[order_index];

            if (check_gemm_thread_count(order, test_row_count, test_column_count,
                                       thread_count) != 0 ||
                check_gemm_thread_count(order, 2, test_column_count,
                                       thread_count) != 0 ||
                check_gemm_thread_count(order, test_row_count, 2,
                                       thread_count) != 0 ||
                check_gemm_thread_count(order, test_row_count, 1,
                                       thread_count) != 0 ||
                check_gemm_thread_count(order, 0, test_column_count,
                                       thread_count) != 0 ||
                check_gemm_thread_count(order, test_row_count, 0,
                                       thread_count) != 0)
                return 1;
        }
    }

    printf("PASS configured GEMM thread counts and complete results\n");
    return 0;
}
