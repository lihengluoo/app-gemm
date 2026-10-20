#include "gemm.h"
#include <stdlib.h>
#include <string.h>

gemm_args get_default_gemm_args(void)
{
    return (gemm_args){.alpha = 1};
}

gemm_options get_default_gemm_options(void)
{
    return (gemm_options){.thread_count = 1, .mc = 256, .nc = 512, .kc = 512};
}

size_t min_count(size_t left, size_t right)
{
    return left < right ? left : right;
}

bool multiply_count(size_t left, size_t right, size_t *dst)
{
    if (right == 0 || left <= SIZE_MAX / right) {
        *dst = left * right;
        return true;
    }
    return false;
}

bool round_count(size_t count, size_t multiple, size_t *dst)
{
    const size_t remainder = count % multiple;
    const size_t extra = remainder == 0 ? 0 : multiple - remainder;
    if (count <= SIZE_MAX - extra) {
        *dst = count + extra;
        return true;
    }
    return false;
}

void *allocate_gemm_buffer(size_t bytes)
{
    size_t rounded;
    if (round_count(bytes, gemm_alignment, &rounded)) {
        return aligned_alloc(gemm_alignment, rounded);
    }
    return NULL;
}

row_partition partition_rows(size_t rows, size_t count, size_t index)
{
    const size_t quotient = rows / count;
    const size_t remainder = rows % count;
    const size_t begin = index * quotient + min_count(index, remainder);
    return (row_partition){begin, begin + quotient + (index < remainder)};
}

size_t get_gemm_column_block(void)
{
    return gemm_columns;
}

gemm_status check_matrix_span(const void *ptr, matrix_order order, size_t rows,
                              size_t columns, size_t stride, size_t width,
                              matrix_span *span)
{
    *span = (matrix_span){0, 0};
    if (rows == 0 || columns == 0) {
        return gemm_success;
    }
    const size_t major = order == gemm_row_major ? rows : columns;
    const size_t minor = order == gemm_row_major ? columns : rows;
    if (ptr == NULL || stride < minor) {
        return gemm_invalid_argument;
    }
    size_t prefix;
    size_t bytes;
    if (!multiply_count(major - 1, stride, &prefix) ||
        prefix > SIZE_MAX - minor ||
        !multiply_count(prefix + minor, width, &bytes) ||
        bytes > (size_t)PTRDIFF_MAX) {
        return gemm_size_overflow;
    }
    const uintptr_t begin = (uintptr_t)ptr;
    if (begin > UINTPTR_MAX - bytes) {
        return gemm_size_overflow;
    }
    *span = (matrix_span){begin, begin + bytes};
    return gemm_success;
}

bool have_overlapping_matrices(const matrix_span *left,
                               const matrix_span *right)
{
    return left->begin < right->end && right->begin < left->end;
}

gemm_status normalize_gemm(const gemm_args *args, const gemm_options *options,
                           gemm_inputs *inputs)
{
    if ((args->order != gemm_row_major && args->order != gemm_column_major) ||
        (args->transpose_a != gemm_no_transpose &&
         args->transpose_a != gemm_transpose) ||
        (args->transpose_b != gemm_no_transpose &&
         args->transpose_b != gemm_transpose) ||
        options->thread_count == 0 ||
        options->thread_count > max_gemm_threads || options->mc == 0 ||
        options->nc == 0 || options->kc == 0 ||
        options->nc % gemm_columns != 0 || options->kc % 2 != 0 ||
        options->mc > gemm_max_panel_rows || options->kc > gemm_max_panel_k) {
        return gemm_invalid_argument;
    }
    const bool transpose_a = args->transpose_a == gemm_transpose;
    const bool transpose_b = args->transpose_b == gemm_transpose;
    matrix_span c_span;
    gemm_status status = check_matrix_span(args->c, args->order, args->m,
                                           args->n, args->ldc, sizeof(int32_t),
                                           &c_span);
    if (status != gemm_success) {
        return status;
    }
    if (args->m != 0 && args->n != 0 && args->alpha != 0 && args->k != 0) {
        matrix_span a_span;
        matrix_span b_span;
        status = check_matrix_span(args->a, args->order,
                                   transpose_a ? args->k : args->m,
                                   transpose_a ? args->m : args->k, args->lda,
                                   sizeof(int16_t), &a_span);
        if (status != gemm_success) {
            return status;
        }
        status = check_matrix_span(args->b, args->order,
                                   transpose_b ? args->n : args->k,
                                   transpose_b ? args->k : args->n, args->ldb,
                                   sizeof(int16_t), &b_span);
        if (status != gemm_success) {
            return status;
        }
        if (have_overlapping_matrices(&a_span, &c_span) ||
            have_overlapping_matrices(&b_span, &c_span)) {
            return gemm_invalid_argument;
        }
    }
    const bool row_major = args->order == gemm_row_major;
    const size_t a_row = row_major != transpose_a ? args->lda : 1;
    const size_t a_k = row_major != transpose_a ? 1 : args->lda;
    const size_t b_k = row_major != transpose_b ? args->ldb : 1;
    const size_t b_column = row_major != transpose_b ? 1 : args->ldb;
    if (row_major) {
        *inputs = (gemm_inputs){args->a, args->b,     a_row,     a_k,
                                b_k,     b_column,    args->m,   args->n,
                                args->k, args->alpha, args->beta};
    } else {
        // C^T = op(B)^T * op(A)^T. Normalize without copying C, so both
        // layouts use the same contiguous-row ownership proof internally.
        *inputs = (gemm_inputs){args->b, args->a,     b_column,  b_k,
                                a_k,     a_row,       args->n,   args->m,
                                args->k, args->alpha, args->beta};
    }
    return gemm_success;
}

void finish_gemm_tile(const gemm_tile *tile, const uint32_t *sums)
{
    for (size_t row = 0; row < tile->rows; ++row) {
        for (size_t column = 0; column < tile->columns; ++column) {
            uint32_t bits =
                (uint32_t)tile->alpha * sums[row * gemm_columns + column];
            int32_t *dst = tile->c + row * tile->ldc + column;
            if (tile->beta != 0) {
                bits += (uint32_t)tile->beta * (uint32_t)*dst;
            }
            // Preserve the bit pattern without implementation-defined integer
            // narrowing or signed overflow. beta zero never reads old C.
            memcpy(dst, &bits, sizeof(bits));
        }
    }
}

void scale_gemm_rows(const gemm_rows *rows)
{
    const gemm_inputs *inputs = &rows->plan.inputs;
    const uint32_t beta = (uint32_t)inputs->beta;
    if (beta != 1) {
        for (size_t row = 0; row < rows->row_count; ++row) {
            for (size_t column = 0; column < inputs->n; ++column) {
                int32_t *dst = rows->c + row * rows->ldc + column;
                const uint32_t bits = beta == 0 ? 0 : beta * (uint32_t)*dst;
                memcpy(dst, &bits, sizeof(bits));
            }
        }
    }
}

gemm_status compute_gemm_i16_i32(const gemm_args *args,
                                 const gemm_options *options)
{
    if (args == NULL || options == NULL) {
        return gemm_invalid_argument;
    }
    gemm_inputs inputs;
    const gemm_status status = normalize_gemm(args, options, &inputs);
    if (status != gemm_success || inputs.m == 0 || inputs.n == 0) {
        return status;
    }
    if (inputs.alpha == 0 || inputs.k == 0) {
        const gemm_plan plan = {inputs, *options, NULL, 0, 0, false};
        const gemm_rows rows = {plan, args->c, inputs.m, args->ldc, 0};
        scale_gemm_rows(&rows);
        return gemm_success;
    }
    if (!is_gemm_cpu_supported()) {
        return gemm_unsupported_cpu;
    }
    const size_t pairs = inputs.k / 2 + inputs.k % 2;
    const size_t blocks =
        inputs.n / gemm_columns + (inputs.n % gemm_columns != 0);
    size_t b_stride;
    size_t b_elements;
    size_t b_bytes;
    size_t a_rows;
    size_t a_elements;
    size_t a_bytes;
    size_t a_stride;
    // Cap scratch to the shape, not a potentially much larger tuning bound.
    const size_t mc = min_count(options->mc, inputs.m);
    const size_t kc = min_count(options->kc, inputs.k);
    if (!multiply_count(pairs, 2 * gemm_columns, &b_stride) ||
        !multiply_count(blocks, b_stride, &b_elements) ||
        !multiply_count(b_elements, sizeof(int16_t), &b_bytes) ||
        !round_count(mc, gemm_rows_per_tile, &a_rows) ||
        !round_count(kc, 2, &a_stride) ||
        !multiply_count(a_rows, a_stride, &a_elements) ||
        !multiply_count(a_elements, sizeof(int16_t), &a_bytes) ||
        b_bytes > (size_t)PTRDIFF_MAX || a_bytes > (size_t)PTRDIFF_MAX) {
        return gemm_size_overflow;
    }
    int16_t *packed_b = pack_gemm_b(&inputs, b_stride, b_elements);
    if (packed_b == NULL) {
        return gemm_no_memory;
    }
    const bool stream = options->stream_output && args->beta == 0 &&
                        inputs.m >
                            gemm_stream_min_bytes / sizeof(int32_t) / inputs.n;
    const gemm_plan plan = {inputs,   *options,   packed_b,
                            b_stride, a_elements, stream};
    const gemm_status computed = run_gemm_threads(&plan, args->c, args->ldc);
    free(packed_b);
    return computed;
}
