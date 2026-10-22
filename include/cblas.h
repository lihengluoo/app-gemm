#pragma once

#include <stdint.h>

/* The supported public API of this OpenBLAS extraction. */
typedef int blasint;

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
    CblasConjNoTrans = 114
};

#ifdef __cplusplus
extern "C" {
#endif

/* Integer C = alpha * op(A) * op(B) + beta * C with int8 inputs and
   int32 accumulation. */
void cblas_i8gemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transpose_a,
                  enum CBLAS_TRANSPOSE transpose_b, blasint m, blasint n,
                  blasint k, int32_t alpha, const int8_t *a, blasint lda,
                  const int8_t *b, blasint ldb, int32_t beta, int32_t *c,
                  blasint ldc);
void openblas_set_num_threads(int num_threads);
int openblas_get_num_threads(void);

#ifdef __cplusplus
}
#endif
