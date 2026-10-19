#include "gemm_internal.h"
#include <immintrin.h>
#include <string.h>

bool is_gemm_cpu_supported(void)
{
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni");
}

const char *get_gemm_cpu_name(void)
{
    return "6780e_avx";
}

void compute_gemm_tile(const gemm_tile *tile)
{
    const bool direct = tile->rows == 4 && tile->columns == 8 &&
                        tile->alpha == 1 &&
                        (tile->beta == 0 || tile->beta == 1);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    if (direct && tile->beta == 1) {
        acc0 = _mm256_loadu_si256((const __m256i *)tile->c);
        acc1 = _mm256_loadu_si256((const __m256i *)(tile->c + tile->ldc));
        acc2 = _mm256_loadu_si256((const __m256i *)(tile->c + 2 * tile->ldc));
        acc3 = _mm256_loadu_si256((const __m256i *)(tile->c + 3 * tile->ldc));
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
        const __m256i b =
            _mm256_loadu_si256((const __m256i *)(tile->b + pair * 16));
        acc0 = _mm256_dpwssd_epi32(acc0, _mm256_set1_epi32(pair0), b);
        acc1 = _mm256_dpwssd_epi32(acc1, _mm256_set1_epi32(pair1), b);
        acc2 = _mm256_dpwssd_epi32(acc2, _mm256_set1_epi32(pair2), b);
        acc3 = _mm256_dpwssd_epi32(acc3, _mm256_set1_epi32(pair3), b);
    }
    if (direct) {
        const __m256i acc[4] = {acc0, acc1, acc2, acc3};
        for (size_t row = 0; row < 4; ++row) {
            __m256i *dst = (__m256i *)(tile->c + row * tile->ldc);
            if (tile->stream && ((uintptr_t)dst & 31) == 0) {
                _mm256_stream_si256(dst, acc[row]);
            } else {
                _mm256_storeu_si256(dst, acc[row]);
            }
        }
    } else {
        _Alignas(32) uint32_t sums[4 * 8];
        _mm256_store_si256((__m256i *)sums, acc0);
        _mm256_store_si256((__m256i *)(sums + 8), acc1);
        _mm256_store_si256((__m256i *)(sums + 16), acc2);
        _mm256_store_si256((__m256i *)(sums + 24), acc3);
        finish_gemm_tile(tile, sums);
    }
}
