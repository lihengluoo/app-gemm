#pragma once

#include "gemm.h"

void require_gemm(bool condition, const char *message);
size_t get_matrix_index(matrix_order order, size_t stride, size_t row,
                        size_t column);
void check_gemm_shape(size_t m, size_t n, size_t k, size_t threads,
                      matrix_order order, matrix_transpose transpose_a,
                      matrix_transpose transpose_b, int32_t alpha,
                      int32_t beta);
void check_gemm_arguments(void);
void check_gemm_partitions(void);
void check_gemm_wrap(void);
void check_gemm_stream(void);
void *check_gemm_concurrent(void *args);
