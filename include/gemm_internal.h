#pragma once

#include "gemm.h"
#include <pthread.h>

#ifndef gemm_cpu
#define gemm_cpu 0
#endif

#if gemm_cpu == 2
#define gemm_cpu_attrs \
    __attribute__((target("avx2,avx512f,avx512bw,avx512vnni")))
#elif gemm_cpu == 1
#define gemm_cpu_attrs __attribute__((target("avx2,avxvnni")))
#else
#define gemm_cpu_attrs
#endif

#if gemm_cpu == 1
enum { gemm_columns = 8 };
#else
enum { gemm_columns = 32 };
#endif
enum {
    gemm_rows_per_tile = 4,
    gemm_alignment = 64,
    gemm_stream_min_bytes = 96 * 1024 * 1024
};

typedef struct matrix_span {
    uintptr_t begin;
    uintptr_t end;
} matrix_span;

typedef struct gemm_inputs {
    const int16_t *a;
    const int16_t *b;
    size_t a_row_stride;
    size_t a_k_stride;
    size_t b_k_stride;
    size_t b_column_stride;
    size_t m;
    size_t n;
    size_t k;
    int32_t alpha;
    int32_t beta;
} gemm_inputs;

// No full writable C view is published through this shared input descriptor.
typedef struct gemm_plan {
    gemm_inputs inputs;
    gemm_options options;
    const int16_t *packed_b;
    size_t b_block_stride;
    size_t a_panel_elements;
    bool stream;
} gemm_plan;

// Initialized before any create, and never changed until every join finishes.
typedef struct gemm_rows {
    const gemm_plan *plan;
    int32_t *c;
    size_t row_count;
    size_t ldc;
    size_t a_row_begin;
} gemm_rows;

typedef struct gemm_tile {
    const int16_t *a;
    const int16_t *b;
    int32_t *c;
    size_t a_stride;
    size_t ldc;
    size_t pairs;
    size_t rows;
    size_t columns;
    int32_t alpha;
    int32_t beta;
    bool stream;
} gemm_tile;

size_t min_count(size_t left, size_t right);
bool multiply_count(size_t left, size_t right, size_t *dst);
bool round_count(size_t count, size_t multiple, size_t *dst);
void *allocate_gemm_buffer(size_t bytes);
gemm_status check_matrix_span(const void *ptr, matrix_order order, size_t rows,
                              size_t columns, size_t stride, size_t width,
                              matrix_span *span);
bool have_overlapping_matrices(const matrix_span *left,
                               const matrix_span *right);
gemm_status normalize_gemm(const gemm_args *args, const gemm_options *options,
                           gemm_inputs *inputs);
int16_t *pack_gemm_b(const gemm_inputs *inputs, size_t block_stride,
                     size_t elements);
void pack_gemm_a(const gemm_inputs *inputs, size_t row_begin, size_t rows,
                 size_t k_begin, size_t depth, size_t stride, int16_t *dst);
void finish_gemm_tile(const gemm_tile *tile, const uint32_t *sums);
void scale_gemm_rows(const gemm_rows *rows);
gemm_cpu_attrs void compute_gemm_tile(const gemm_tile *tile);
gemm_cpu_attrs gemm_status compute_gemm_rows(const gemm_rows *rows);
gemm_cpu_attrs void *run_gemm_worker(void *args);
gemm_status run_gemm_threads(const gemm_plan *plan, int32_t *c, size_t ldc);
