#pragma once

#include "cblas.h"
#include <stdbool.h>
#include <stddef.h>

enum { matrix_guard_count = 8, gemm_test_threads = 4 };
static const double matrix_guard = -4096.5;
static const int gemm_threads[] = {1, gemm_test_threads};

typedef enum matrix_role { matrix_a, matrix_b, matrix_c } matrix_role;

typedef struct matrix {
    double *ptr;
    double *entries;
    size_t rows;
    size_t columns;
    size_t stride;
    size_t count;
    bool row_major;
} matrix;

typedef struct gemm_case {
    const char *name;
    blasint m;
    blasint n;
    blasint k;
    double alpha;
    double beta;
    bool expect_parallel;
} gemm_case;

typedef struct gemm_thread_counts {
    size_t calls;
    long max_jobs;
} gemm_thread_counts;

static const gemm_case gemm_cases[] = {
    {"odd dimensions", 131, 99, 97, 2.0, -1.0, true},
    {"blocked, beta zero", 257, 129, 131, 0.5, 0.0, true},
    {"small dimensions", 3, 5, 7, 1.0, 1.0, false},
    {"alpha zero", 5, 3, 7, 0.0, -2.0, false},
    {"zero depth", 5, 3, 0, 2.0, -2.0, false},
    {"zero rows", 0, 5, 7, 1.0, 0.0, false},
    {"zero columns", 5, 0, 7, 1.0, 0.0, false},
};

/* Match the private queue pointer without exposing private OpenBLAS headers. */
struct blas_queue;
int __real_exec_blas(long count, struct blas_queue *queue);
int __wrap_exec_blas(long count, struct blas_queue *queue);

size_t get_matrix_index(const matrix *mat, size_t row, size_t column);
double get_matrix_number(matrix_role role, size_t row, size_t column);
bool allocate_matrix(matrix *mat, size_t rows, size_t columns, bool row_major);
void fill_matrix(matrix *mat, matrix_role role, bool use_nan);
bool check_matrix_padding(const matrix *mat);
bool check_matrix_input(const matrix *mat, matrix_role role);
bool check_gemm_result(const matrix *c, const gemm_case *args,
                       bool transpose_a, bool transpose_b);
bool run_gemm_case(const gemm_case *args, bool row_major, bool transpose_a,
                   bool transpose_b, int thread_count);
