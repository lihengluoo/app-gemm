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

/* Packing for the 4509y AVX-512 VNNI kernel.  Both entry points pack the
   same logical block: k = m steps by entity = n (rows of A, or columns of
   B) into the layout i8gemm_kernel expects:

     entity j at ((int16_t *)b) + j * k_pad, one int8 value per int16,
     k_pad = (m + 1) & ~1 with an odd-k tail zero-filled.

   The two variants differ only in source addressing, matching the level-3
   driver macros:
     i8gemm_oncopy reads element (i, j) at a[i + j * lda]
     i8gemm_otcopy reads element (i, j) at a[j + i * lda]

   A panel bytes are n * 2 * k_pad, and each packed B column occupies
   pad_min_l = 2 * k_pad bytes, the stride the driver uses between B
   panels.  */

static BLASLONG pack_k_pad_4509y(BLASLONG m) {
  return (m + 1) & ~1L;
}

int i8gemm_oncopy(BLASLONG m, BLASLONG n, int8_t *a, BLASLONG lda, int8_t *b) {

  int16_t *out = (int16_t *)b;
  const BLASLONG k_pad = pack_k_pad_4509y(m);

  for (BLASLONG j = 0; j < n; j ++) {
    const int8_t *src = a + j * lda;
    int16_t *dst = out + j * k_pad;
    BLASLONG i;

    for (i = 0; i < m; i ++) dst[i] = src[i];
    if (k_pad > m) dst[m] = 0;
  }

  return 0;
}

int i8gemm_otcopy(BLASLONG m, BLASLONG n, int8_t *a, BLASLONG lda, int8_t *b) {

  int16_t *out = (int16_t *)b;
  const BLASLONG k_pad = pack_k_pad_4509y(m);

  for (BLASLONG j = 0; j < n; j ++) {
    int16_t *dst = out + j * k_pad;
    BLASLONG i;

    for (i = 0; i < m; i ++) dst[i] = a[j + i * lda];
    if (k_pad > m) dst[m] = 0;
  }

  return 0;
}
