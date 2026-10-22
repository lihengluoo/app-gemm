#pragma once

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

void cblas_dgemm(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transpose_a,
                 enum CBLAS_TRANSPOSE transpose_b, blasint m, blasint n,
                 blasint k, double alpha, const double *a, blasint lda,
                 const double *b, blasint ldb, double beta, double *c,
                 blasint ldc);
void openblas_set_num_threads(int num_threads);
int openblas_get_num_threads(void);

#ifdef __cplusplus
}
#endif
