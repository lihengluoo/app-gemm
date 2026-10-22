/*****************************************************************************
Copyright (c) 2026, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

#include "common.h"
#include <immintrin.h>
#include <string.h>

/* AVX-512 VNNI int8 kernel for the Intel 4509Y class (AVX-512 + VNNI).
   It follows the validated pairing scheme of the former 4509y backend:
   each int8 value is held as a sign-extended int16, so one vpdpwssd
   multiplies two k steps of one row against two k steps of sixteen
   columns.  Packed layouts, produced by lib/kernel/4509y/gemm_copy_4509y.c:

     A panel: row r    at ((const int16_t *)ba) + r    * k_pad
     B panel: column c at ((const int16_t *)bb) + c    * k_pad
     k_pad = (bk + 1) & ~1, odd-k tails zero-filled by the packers

   so the dword at (entity, 2 * pair) holds the values for k = 2*pair and
   2*pair + 1, exactly matching a broadcast A dword.  The driver keeps B
   panels at sb + pad_min_l * (jjs - js) with pad_min_l = 2 * k_pad, which
   is the column stride used here.  Accumulation wraps in int32 exactly
   like the generic kernel. */

#define TILE_ROWS 8
#define TILE_COLS 32

static void kernel_tile_8x32(const int16_t *a_row, const int16_t *b_col,
                             int32_t *c_ptr, BLASLONG ldc, BLASLONG nrem,
                             BLASLONG k_pad, BLASLONG pairs, int32_t alpha) {

  __m512i acc[TILE_ROWS][2];
  const __m512i lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                          8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i col_stride = _mm512_set1_epi32((int)(2 * k_pad));
  __m512i idx_lo = _mm512_mullo_epi32(lanes, col_stride);
  __m512i idx_hi = _mm512_add_epi32(idx_lo,
                                    _mm512_set1_epi32((int)(k_pad * TILE_COLS)));
  const __mmask16 m_lo = nrem >= 16 ? (__mmask16)0xFFFF
                                    : (__mmask16)((1u << nrem) - 1u);
  const __mmask16 m_hi = nrem > 16 ? (__mmask16)((1u << (nrem - 16)) - 1u) : 0;
  const char *base = (const char *)b_col;
  const __m512i zero = _mm512_setzero_si512();
  const __m512i pair_step = _mm512_set1_epi32(4);
  int r;

  for (r = 0; r < TILE_ROWS; r ++) {
    acc[r][0] = zero;
    acc[r][1] = zero;
  }

  for (BLASLONG p = 0; p < pairs; p ++) {

    __m512i b_lo = _mm512_mask_i32gather_epi32(zero, m_lo, idx_lo, base, 1);
    __m512i b_hi = _mm512_mask_i32gather_epi32(zero, m_hi, idx_hi, base, 1);

    for (r = 0; r < TILE_ROWS; r ++) {
      int32_t a_dword;
      memcpy(&a_dword, a_row + r * k_pad + 2 * p, sizeof(a_dword));
      __m512i a_v = _mm512_set1_epi32(a_dword);

      acc[r][0] = _mm512_dpwssd_epi32(acc[r][0], a_v, b_lo);
      acc[r][1] = _mm512_dpwssd_epi32(acc[r][1], a_v, b_hi);
    }

    idx_lo = _mm512_add_epi32(idx_lo, pair_step);
    idx_hi = _mm512_add_epi32(idx_hi, pair_step);
  }

  const __m512i v_alpha = _mm512_set1_epi32(alpha);

  for (r = 0; r < TILE_ROWS; r ++) {
    int32_t *c_row = c_ptr + r * ldc;

    __m512i s_lo = _mm512_mullo_epi32(acc[r][0], v_alpha);
    __m512i s_hi = _mm512_mullo_epi32(acc[r][1], v_alpha);
    __m512i old_lo = _mm512_maskz_loadu_epi32(m_lo, c_row);
    __m512i old_hi = _mm512_maskz_loadu_epi32(m_hi, c_row + 16);

    _mm512_mask_storeu_epi32(c_row, m_lo, _mm512_add_epi32(s_lo, old_lo));
    _mm512_mask_storeu_epi32(c_row + 16, m_hi, _mm512_add_epi32(s_hi, old_hi));
  }
}

int i8gemm_kernel(BLASLONG bm, BLASLONG bn, BLASLONG bk, int32_t alpha,
                  int8_t *ba, int8_t *bb, int32_t *c, BLASLONG ldc) {

  const int16_t *a16 = (const int16_t *)ba;
  const int16_t *b16 = (const int16_t *)bb;
  const BLASLONG k_pad = (bk + 1) & ~1L;
  const BLASLONG pairs = (bk + 1) >> 1;
  BLASLONG r, cb;

  for (r = 0; bm - r >= TILE_ROWS; r += TILE_ROWS) {
    for (cb = 0; cb < bn; cb += TILE_COLS) {
      BLASLONG nrem = bn - cb;
      if (nrem > TILE_COLS) nrem = TILE_COLS;

      kernel_tile_8x32(a16 + r * k_pad, b16, c + r * ldc + cb, ldc, nrem,
                       k_pad, pairs, alpha);
    }
  }

  /* Row tail of at most seven rows: exact scalar remainder. */
  for (; r < bm; r ++) {
    const int16_t *a_row = a16 + r * k_pad;
    int32_t *c_row = c + r * ldc;

    for (cb = 0; cb < bn; cb ++) {
      const int16_t *b_k = b16 + cb * k_pad;
      int32_t sum = 0;

      for (BLASLONG k = 0; k < bk; k ++) sum += a_row[k] * b_k[k];

      c_row[cb] += alpha * sum;
    }
  }

  return 0;
}
