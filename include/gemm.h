#pragma once

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
