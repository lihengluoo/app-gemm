#ifndef GEMM_THREAD_COUNT_TEST_H
#define GEMM_THREAD_COUNT_TEST_H

#include <pthread.h>
#include "cblas.h"

enum {
    test_row_count = 7,
    test_column_count = 11,
    test_depth_count = 7,
    default_thread_count = 4
};

static const int configured_thread_counts[] = {1, 3, 5, 8};
static const enum CBLAS_ORDER matrix_orders[] = {CblasRowMajor, CblasColMajor};

int __real_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                         void *(*start_routine)(void *), void *arg);
int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                         void *(*start_routine)(void *), void *arg);
int __real_pthread_join(pthread_t thread, void **result);
int __wrap_pthread_join(pthread_t thread, void **result);
static int check_gemm_thread_count(enum CBLAS_ORDER order, int row_count,
                                  int column_count, int configured_thread_count);

#endif
