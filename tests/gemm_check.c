#include "gemm_check.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static gemm_thread_counts thread_counts;

int __wrap_exec_blas(long count, struct blas_queue *queue)
{
    ++thread_counts.calls;
    if (count > thread_counts.max_jobs) {
        thread_counts.max_jobs = count;
    }
    return __real_exec_blas(count, queue);
}

size_t get_matrix_index(const matrix *mat, size_t row, size_t column)
{
    return mat->row_major ? row * mat->stride + column :
                            column * mat->stride + row;
}

size_t get_matrix_int32_index(const matrix_int32 *mat, size_t row,
                              size_t column)
{
    return mat->row_major ? row * mat->stride + column :
                            column * mat->stride + row;
}

int get_matrix_number(matrix_role role, size_t row, size_t column)
{
    if (role == matrix_a) {
        return (int)((row * 3 + column * 5) % 17) - 8;
    } else if (role == matrix_b) {
        return (int)((row * 7 + column * 2) % 19) - 9;
    } else {
        return (int)((row * 9 + column * 4) % 13) - 6;
    }
}

bool allocate_matrix(matrix *mat, size_t rows, size_t columns, bool row_major)
{
    mat->rows = rows;
    mat->columns = columns;
    mat->row_major = row_major;
    mat->stride = (row_major ? columns : rows) + 3;
    mat->count = (row_major ? rows : columns) * mat->stride +
                 2 * matrix_guard_count;
    mat->ptr = malloc(mat->count * sizeof(*mat->ptr));
    if (mat->ptr != NULL) {
        mat->entries = mat->ptr + matrix_guard_count;
        for (size_t index = 0; index < mat->count; ++index) {
            mat->ptr[index] = matrix_input_guard;
        }
        return true;
    } else {
        return false;
    }
}

bool allocate_matrix_int32(matrix_int32 *mat, size_t rows, size_t columns,
                           bool row_major)
{
    mat->rows = rows;
    mat->columns = columns;
    mat->row_major = row_major;
    mat->stride = (row_major ? columns : rows) + 3;
    mat->count = (row_major ? rows : columns) * mat->stride +
                 2 * matrix_guard_count;
    mat->ptr = malloc(mat->count * sizeof(*mat->ptr));
    if (mat->ptr != NULL) {
        mat->entries = mat->ptr + matrix_guard_count;
        for (size_t index = 0; index < mat->count; ++index) {
            mat->ptr[index] = matrix_output_guard;
        }
        return true;
    } else {
        return false;
    }
}

void fill_matrix(matrix *mat, matrix_role role)
{
    for (size_t row = 0; row < mat->rows; ++row) {
        for (size_t column = 0; column < mat->columns; ++column) {
            mat->entries[get_matrix_index(mat, row, column)] =
                (int8_t)get_matrix_number(role, row, column);
        }
    }
}

void fill_matrix_int32(matrix_int32 *mat, bool use_junk)
{
    for (size_t row = 0; row < mat->rows; ++row) {
        for (size_t column = 0; column < mat->columns; ++column) {
            mat->entries[get_matrix_int32_index(mat, row, column)] =
                use_junk ? matrix_junk_fill :
                (int32_t)get_matrix_number(matrix_c, row, column);
        }
    }
}

bool check_matrix_padding(const matrix *mat)
{
    for (size_t index = 0; index < matrix_guard_count; ++index) {
        if (mat->ptr[index] != matrix_input_guard ||
            mat->ptr[mat->count - 1 - index] != matrix_input_guard) {
            fprintf(stderr, "matrix guard changed\n");
            return false;
        }
    }
    const size_t major = mat->row_major ? mat->rows : mat->columns;
    const size_t minor = mat->row_major ? mat->columns : mat->rows;
    for (size_t block = 0; block < major; ++block) {
        for (size_t index = minor; index < mat->stride; ++index) {
            if (mat->entries[block * mat->stride + index] !=
                matrix_input_guard) {
                fprintf(stderr, "matrix padding changed\n");
                return false;
            }
        }
    }
    return true;
}

bool check_matrix_int32_padding(const matrix_int32 *mat)
{
    for (size_t index = 0; index < matrix_guard_count; ++index) {
        if (mat->ptr[index] != matrix_output_guard ||
            mat->ptr[mat->count - 1 - index] != matrix_output_guard) {
            fprintf(stderr, "matrix guard changed\n");
            return false;
        }
    }
    const size_t major = mat->row_major ? mat->rows : mat->columns;
    const size_t minor = mat->row_major ? mat->columns : mat->rows;
    for (size_t block = 0; block < major; ++block) {
        for (size_t index = minor; index < mat->stride; ++index) {
            if (mat->entries[block * mat->stride + index] !=
                matrix_output_guard) {
                fprintf(stderr, "matrix padding changed\n");
                return false;
            }
        }
    }
    return true;
}

bool check_matrix_input(const matrix *mat, matrix_role role)
{
    for (size_t row = 0; row < mat->rows; ++row) {
        for (size_t column = 0; column < mat->columns; ++column) {
            if (mat->entries[get_matrix_index(mat, row, column)] !=
                (int8_t)get_matrix_number(role, row, column)) {
                fprintf(stderr, "input matrix changed\n");
                return false;
            }
        }
    }
    return check_matrix_padding(mat);
}

bool check_gemm_result(const matrix_int32 *c, const gemm_case *args,
                       bool transpose_a, bool transpose_b)
{
    for (size_t row = 0; row < c->rows; ++row) {
        for (size_t column = 0; column < c->columns; ++column) {
            int64_t expected = args->beta == 0 ? 0 :
                (int64_t)args->beta * get_matrix_number(matrix_c, row, column);
            if (args->alpha != 0) {
                int64_t sum = 0;
                for (size_t inner = 0; inner < (size_t)args->k; ++inner) {
                    sum += (int64_t)get_matrix_number(matrix_a,
                                transpose_a ? inner : row,
                                transpose_a ? row : inner) *
                           get_matrix_number(matrix_b,
                                transpose_b ? column : inner,
                                transpose_b ? inner : column);
                }
                expected += (int64_t)args->alpha * sum;
            }
            /* Small inputs keep the int64 reference and the int32
               arithmetic of the library exact. */
            const int32_t computed =
                c->entries[get_matrix_int32_index(c, row, column)];
            if (computed != (int32_t)expected) {
                fprintf(stderr,
                        "C[%zu,%zu]: expected %" PRId64 ", computed %" PRId32
                        "\n",
                        row, column, expected, computed);
                return false;
            }
        }
    }
    return check_matrix_int32_padding(c);
}

bool run_gemm_case(const gemm_case *args, bool row_major, bool transpose_a,
                   bool transpose_b, int thread_count)
{
    matrix a = {0}, b = {0};
    matrix_int32 c = {0};
    bool passed = allocate_matrix(&a, transpose_a ? args->k : args->m,
                                  transpose_a ? args->m : args->k, row_major) &&
                  allocate_matrix(&b, transpose_b ? args->n : args->k,
                                  transpose_b ? args->k : args->n, row_major) &&
                  allocate_matrix_int32(&c, args->m, args->n, row_major);
    if (passed) {
        fill_matrix(&a, matrix_a);
        fill_matrix(&b, matrix_b);
        /* beta zero must overwrite C even when its old entries are junk. */
        fill_matrix_int32(&c, args->beta == 0);
        thread_counts = (gemm_thread_counts){0};
        const bool use_inputs = args->alpha != 0 && args->k != 0;
        cblas_i8gemm(row_major ? CblasRowMajor : CblasColMajor,
                     transpose_a ? CblasTrans : CblasNoTrans,
                     transpose_b ? CblasTrans : CblasNoTrans,
                     args->m, args->n, args->k, args->alpha,
                     use_inputs ? a.entries : NULL, (blasint)a.stride,
                     use_inputs ? b.entries : NULL, (blasint)b.stride,
                     args->beta, c.entries, (blasint)c.stride);
        passed = check_gemm_result(&c, args, transpose_a, transpose_b) &&
                 check_matrix_input(&a, matrix_a) &&
                 check_matrix_input(&b, matrix_b);
        if (args->expect_parallel && thread_count > 1 &&
            (thread_counts.calls == 0 || thread_counts.max_jobs != thread_count)) {
            fprintf(stderr, "expected %d parallel jobs, dispatched %ld\n",
                    thread_count, thread_counts.max_jobs);
            passed = false;
        }
        if (thread_count == 1 && thread_counts.calls != 0) {
            fprintf(stderr, "single thread GEMM dispatched parallel jobs\n");
            passed = false;
        }
    } else {
        fprintf(stderr, "matrix allocation failed\n");
    }
    if (!passed) {
        fprintf(stderr, "%s: %s %c%c, threads=%d\n", args->name,
                row_major ? "row major" : "column major",
                transpose_a ? 'T' : 'N', transpose_b ? 'T' : 'N', thread_count);
    }
    free(a.ptr);
    free(b.ptr);
    free(c.ptr);
    return passed;
}

int main(void)
{
    size_t checks = 0;
    for (size_t thread_index = 0;
         thread_index < sizeof(gemm_threads) / sizeof(gemm_threads[0]); ++thread_index) {
        const int threads = gemm_threads[thread_index];
        openblas_set_num_threads(threads);
        if (openblas_get_num_threads() != threads) {
            fprintf(stderr, "could not set %d OpenBLAS threads\n", threads);
            return EXIT_FAILURE;
        }
        for (size_t index = 0; index < sizeof(gemm_cases) / sizeof(gemm_cases[0]); ++index) {
            for (int layout = 0; layout < 2; ++layout) {
                for (int transa = 0; transa < 2; ++transa) {
                    for (int transb = 0; transb < 2; ++transb) {
                        if (run_gemm_case(&gemm_cases[index], layout != 0,
                                          transa != 0, transb != 0, threads)) {
                            ++checks;
                        } else {
                            return EXIT_FAILURE;
                        }
                    }
                }
            }
        }
    }
    printf("PASS: %zu I8GEMM checks with 1 and %d threads; parallel dispatch verified.\n",
           checks, gemm_test_threads);
    return EXIT_SUCCESS;
}
