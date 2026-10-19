#include "gemm.h"
#include <immintrin.h>
#include <stdlib.h>

gemm_status compute_gemm_rows(const gemm_rows *rows)
{
    if (rows->row_count == 0) {
        return gemm_success;
    }
    const gemm_plan *plan = rows->plan;
    int16_t *packed_a =
        allocate_gemm_buffer(plan->a_panel_elements * sizeof(int16_t));
    if (packed_a == NULL) {
        return gemm_no_memory;
    }
    for (size_t column_base = 0; column_base < plan->inputs.n;) {
        const size_t column_count =
            min_count(plan->options.nc, plan->inputs.n - column_base);
        for (size_t row_base = 0; row_base < rows->row_count;) {
            const size_t row_count =
                min_count(plan->options.mc, rows->row_count - row_base);
            for (size_t k_base = 0; k_base < plan->inputs.k;) {
                const size_t depth =
                    min_count(plan->options.kc, plan->inputs.k - k_base);
                const size_t stride = depth + depth % 2;
                pack_gemm_a(&plan->inputs, rows->a_row_begin + row_base,
                            row_count, k_base, depth, stride, packed_a);
                for (size_t column = 0; column < column_count;) {
                    const size_t columns =
                        min_count(gemm_columns, column_count - column);
                    const int16_t *b = plan->packed_b +
                                       ((column_base + column) / gemm_columns) *
                                           plan->b_block_stride +
                                       (k_base / 2) * 2 * gemm_columns;
                    for (size_t row = 0; row < row_count;) {
                        const size_t tile_rows =
                            min_count(gemm_rows_per_tile, row_count - row);
                        const gemm_tile tile = {
                            packed_a + row * stride,
                            b,
                            rows->c + (row_base + row) * rows->ldc +
                                column_base + column,
                            stride,
                            rows->ldc,
                            stride / 2,
                            tile_rows,
                            columns,
                            plan->inputs.alpha,
                            k_base == 0 ? plan->inputs.beta : 1,
                            plan->stream && depth == plan->inputs.k - k_base};
                        compute_gemm_tile(&tile);
                        row += tile_rows;
                    }
                    column += columns;
                }
                k_base += depth;
            }
            row_base += row_count;
        }
        column_base += column_count;
    }
    if (plan->stream) {
        // Drain streaming stores before pthread termination publishes completion.
        // This fence does not replace the parent's successful join.
        _mm_sfence();
    }
    free(packed_a);
    return gemm_success;
}

void *run_gemm_worker(void *args)
{
    const gemm_status status = compute_gemm_rows(args);
    return status == gemm_success ? NULL : (void *)(uintptr_t)status;
}

gemm_status run_gemm_threads(const gemm_plan *plan, int32_t *c, size_t ldc)
{
    pthread_t handles[max_gemm_threads];
    gemm_rows jobs[max_gemm_threads];
    const size_t count = min_count(plan->options.thread_count, plan->inputs.m);
    // Complete every descriptor before publishing any element to a worker.
    for (size_t thread = 0; thread < count; ++thread) {
        const row_partition partition =
            partition_rows(plan->inputs.m, count, thread);
        jobs[thread] = (gemm_rows){plan, c + partition.begin * ldc,
                                   partition.end - partition.begin, ldc,
                                   partition.begin};
    }
    size_t created = 0;
    for (; created < count; ++created) {
        if (pthread_create(&handles[created], NULL, run_gemm_worker,
                           &jobs[created]) != 0) {
            break;
        }
    }
    gemm_status status = created == count ? gemm_success :
                                            gemm_thread_create_failed;
    for (size_t thread = 0; thread < created; ++thread) {
        void *completed;
        if (pthread_join(handles[thread], &completed) == 0) {
            if (completed != NULL && status == gemm_success) {
                status = (gemm_status)(uintptr_t)completed;
            }
        } else {
            // Never return dangling descriptors or free B with a possibly live
            // worker after an unexpected pthread lifetime error.
            abort();
        }
    }
    return status;
}
