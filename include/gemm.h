#pragma once

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum matrix_order { gemm_row_major, gemm_column_major } matrix_order;
typedef enum matrix_transpose {
    gemm_no_transpose,
    gemm_transpose
} matrix_transpose;
typedef enum gemm_status {
    gemm_success,
    gemm_invalid_argument,
    gemm_size_overflow,
    gemm_no_memory,
    gemm_unsupported_cpu,
    gemm_thread_create_failed
} gemm_status;

enum { max_gemm_threads = 128 };

typedef struct gemm_args {
    matrix_order order;
    matrix_transpose transpose_a;
    matrix_transpose transpose_b;
    size_t m;
    size_t n;
    size_t k;
    const int16_t *a;
    const int16_t *b;
    int32_t *c;
    size_t lda;
    size_t ldb;
    size_t ldc;
    int32_t alpha;
    int32_t beta;
} gemm_args;

typedef struct gemm_options {
    size_t thread_count;
    size_t mc;
    size_t nc;
    size_t kc;
    bool stream_output;
} gemm_options;

typedef struct row_partition {
    size_t begin;
    size_t end;
} row_partition;

gemm_args get_default_gemm_args(void);
gemm_options get_default_gemm_options(void);

// Returns [begin,end); adjacent partitions meet, including when count > rows.
// Preconditions: count > 0 and index < count.
row_partition partition_rows(size_t rows, size_t count, size_t index);
bool is_gemm_cpu_supported(void);
const char *get_gemm_cpu_name(void);
size_t get_gemm_column_block(void);

// C = alpha * op(A) * op(B) + beta * C. Leading dimensions count elements.
// Matrices and argument descriptors must remain alive and unchanged for this
// synchronous call, except for C. C must not overlap A or B. The caller must
// exclude concurrent writes to the inputs/output. A/B may be null when alpha
// == 0 or k == 0. A zero m or n touches no matrix. beta == 0 never reads old C.
// Only logical C elements are modified.
// Arithmetic has defined two's-complement modulo-2^32 behavior, and equals
// mathematical integer GEMM whenever the result fits INT32. In particular the
// old [-127,127], bounded-K workload retains its exact results. Callers needing
// checked overflow must check a magnitude bound or use a wider reference.
// Invalid arguments/allocation failure before creation leave C unchanged;
// worker allocation/create failure can leave a partially computed C. A join
// failure is fatal: returning/freeing buffers with a possibly live worker is
// unsafe. No worker, allocation or mutable matrix state survives this call.
gemm_status compute_gemm_i16_i32(const gemm_args *args,
                                 const gemm_options *options);

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
// The plan is embedded by value so a single descriptor copy carries every
// worker input without following a loaded pointer.
typedef struct gemm_rows {
    gemm_plan plan;
    int32_t *c;
    size_t row_count;
    size_t ldc;
    size_t a_row_begin;
} gemm_rows;

// Tuning bounds that size the worker A panel at compile time. Rounding a
// shape-capped mc (multiple of gemm_rows_per_tile) and kc (even) never
// exceeds these products, so the panel alloca always holds the whole block.
enum { gemm_max_panel_rows = 256,
       gemm_max_panel_k = 512,
       gemm_max_panel_elements = gemm_max_panel_rows * gemm_max_panel_k };

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
