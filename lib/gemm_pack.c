#include "gemm.h"

int16_t *pack_gemm_b(const gemm_inputs *inputs, size_t block_stride,
                     size_t elements)
{
    int16_t *packed = allocate_gemm_buffer(elements * sizeof(int16_t));
    if (packed != NULL) {
        const size_t pairs = inputs->k / 2 + inputs->k % 2;
        const size_t blocks = elements / block_stride;
        for (size_t block = 0; block < blocks; ++block) {
            for (size_t pair = 0; pair < pairs; ++pair) {
                for (size_t column = 0; column < gemm_columns; ++column) {
                    for (size_t half = 0; half < 2; ++half) {
                        const size_t inner = 2 * pair + half;
                        const size_t index = block * gemm_columns + column;
                        packed[block * block_stride + pair * 2 * gemm_columns +
                               2 * column + half] =
                            inner < inputs->k && index < inputs->n ?
                                inputs->b[inner * inputs->b_k_stride +
                                          index * inputs->b_column_stride] :
                                0;
                    }
                }
            }
        }
    }
    return packed;
}

void pack_gemm_a(const gemm_inputs *inputs, size_t row_begin, size_t rows,
                 size_t k_begin, size_t depth, size_t stride, int16_t *dst)
{
    const size_t padded_rows = (rows + gemm_rows_per_tile - 1) /
                               gemm_rows_per_tile * gemm_rows_per_tile;
    for (size_t row = 0; row < padded_rows; ++row) {
        for (size_t inner = 0; inner < stride; ++inner) {
            dst[row * stride + inner] =
                row < rows && inner < depth ?
                    inputs->a[(row_begin + row) * inputs->a_row_stride +
                              (k_begin + inner) * inputs->a_k_stride] :
                    0;
        }
    }
}
