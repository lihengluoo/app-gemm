#include "gemm.h"
#include <immintrin.h>
#include <string.h>

bool is_gemm_cpu_supported(void)
{
    return true;
}

const char *get_gemm_cpu_name(void)
{
    return "scalar";
}

void compute_gemm_tile(const gemm_tile *tile)
{
    uint32_t acc[gemm_rows_per_tile * gemm_columns] = {0};
    for (size_t pair = 0; pair < tile->pairs; ++pair) {
        const int16_t *b = tile->b + pair * 2 * gemm_columns;
        for (size_t row = 0; row < tile->rows; ++row) {
            const int32_t a0 = tile->a[row * tile->a_stride + 2 * pair];
            const int32_t a1 = tile->a[row * tile->a_stride + 2 * pair + 1];
            for (size_t column = 0; column < gemm_columns; ++column) {
                // Each product fits INT32, but their signed sum need not.
                // Unsigned accumulation matches non-saturating VNNI exactly.
                acc[row * gemm_columns + column] +=
                    (uint32_t)(a0 * b[2 * column]) +
                    (uint32_t)(a1 * b[2 * column + 1]);
            }
        }
    }
    if (tile->alpha == 1 && (tile->beta == 0 || tile->beta == 1) &&
        tile->rows == gemm_rows_per_tile && tile->columns == gemm_columns) {
        for (size_t row = 0; row < gemm_rows_per_tile; ++row) {
            for (size_t column = 0; column < gemm_columns; ++column) {
                int32_t *dst = tile->c + row * tile->ldc + column;
                uint32_t bits = acc[row * gemm_columns + column];
                if (tile->beta == 1) {
                    bits += (uint32_t)*dst;
                }
                int32_t stored;
                memcpy(&stored, &bits, sizeof(stored));
                if (tile->stream) {
                    _mm_stream_si32(dst, stored);
                } else {
                    *dst = stored;
                }
            }
        }
    } else {
        finish_gemm_tile(tile, acc);
    }
}
