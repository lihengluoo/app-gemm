#include "gemm_internal.h"
#include <immintrin.h>
#include <string.h>

bool is_gemm_cpu_supported(void)
{
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vnni");
}

const char *get_gemm_cpu_name(void)
{
    return "4509y_avx";
}

void compute_gemm_tile(const gemm_tile *tile)
{
    // Four rows, two zmm vectors per row: eight independent accumulator chains.
    const bool direct = tile->rows == 4 && tile->columns == 32 &&
                        tile->alpha == 1 &&
                        (tile->beta == 0 || tile->beta == 1);
    __m512i acc0_lo = _mm512_setzero_si512();
    __m512i acc0_hi = _mm512_setzero_si512();
    __m512i acc1_lo = _mm512_setzero_si512();
    __m512i acc1_hi = _mm512_setzero_si512();
    __m512i acc2_lo = _mm512_setzero_si512();
    __m512i acc2_hi = _mm512_setzero_si512();
    __m512i acc3_lo = _mm512_setzero_si512();
    __m512i acc3_hi = _mm512_setzero_si512();
    if (direct && tile->beta == 1) {
        acc0_lo = _mm512_loadu_si512(tile->c);
        acc0_hi = _mm512_loadu_si512(tile->c + 16);
        acc1_lo = _mm512_loadu_si512(tile->c + tile->ldc);
        acc1_hi = _mm512_loadu_si512(tile->c + tile->ldc + 16);
        acc2_lo = _mm512_loadu_si512(tile->c + 2 * tile->ldc);
        acc2_hi = _mm512_loadu_si512(tile->c + 2 * tile->ldc + 16);
        acc3_lo = _mm512_loadu_si512(tile->c + 3 * tile->ldc);
        acc3_hi = _mm512_loadu_si512(tile->c + 3 * tile->ldc + 16);
    }
    for (size_t pair = 0; pair < tile->pairs; ++pair) {
        int32_t pair0;
        int32_t pair1;
        int32_t pair2;
        int32_t pair3;
        memcpy(&pair0, tile->a + 2 * pair, sizeof(pair0));
        memcpy(&pair1, tile->a + tile->a_stride + 2 * pair, sizeof(pair1));
        memcpy(&pair2, tile->a + 2 * tile->a_stride + 2 * pair, sizeof(pair2));
        memcpy(&pair3, tile->a + 3 * tile->a_stride + 2 * pair, sizeof(pair3));
        const __m512i a0 = _mm512_set1_epi32(pair0);
        const __m512i a1 = _mm512_set1_epi32(pair1);
        const __m512i a2 = _mm512_set1_epi32(pair2);
        const __m512i a3 = _mm512_set1_epi32(pair3);
        const __m512i b_lo = _mm512_load_si512(tile->b + pair * 64);
        const __m512i b_hi = _mm512_load_si512(tile->b + pair * 64 + 32);
        acc0_lo = _mm512_dpwssd_epi32(acc0_lo, a0, b_lo);
        acc0_hi = _mm512_dpwssd_epi32(acc0_hi, a0, b_hi);
        acc1_lo = _mm512_dpwssd_epi32(acc1_lo, a1, b_lo);
        acc1_hi = _mm512_dpwssd_epi32(acc1_hi, a1, b_hi);
        acc2_lo = _mm512_dpwssd_epi32(acc2_lo, a2, b_lo);
        acc2_hi = _mm512_dpwssd_epi32(acc2_hi, a2, b_hi);
        acc3_lo = _mm512_dpwssd_epi32(acc3_lo, a3, b_lo);
        acc3_hi = _mm512_dpwssd_epi32(acc3_hi, a3, b_hi);
    }
    if (direct) {
        // Only form high-half C pointers for complete 32-column tiles.
        const __m512i acc[8] = {acc0_lo, acc0_hi, acc1_lo, acc1_hi,
                                acc2_lo, acc2_hi, acc3_lo, acc3_hi};
        for (size_t row = 0; row < 4; ++row) {
            for (size_t half = 0; half < 2; ++half) {
                int32_t *dst = tile->c + row * tile->ldc + half * 16;
                if (tile->stream && ((uintptr_t)dst & 63) == 0) {
                    _mm512_stream_si512((__m512i *)dst, acc[2 * row + half]);
                } else {
                    _mm512_storeu_si512(dst, acc[2 * row + half]);
                }
            }
        }
    } else {
        _Alignas(64) uint32_t sums[4 * 32];
        _mm512_store_si512(sums, acc0_lo);
        _mm512_store_si512(sums + 16, acc0_hi);
        _mm512_store_si512(sums + 32, acc1_lo);
        _mm512_store_si512(sums + 48, acc1_hi);
        _mm512_store_si512(sums + 64, acc2_lo);
        _mm512_store_si512(sums + 80, acc2_hi);
        _mm512_store_si512(sums + 96, acc3_lo);
        _mm512_store_si512(sums + 112, acc3_hi);
        finish_gemm_tile(tile, sums);
    }
}
