#ifndef COMMON_I8_H
#define COMMON_I8_H

/* Symbol table for the INT8 GEMM type: int8 inputs, int32 accumulation.
   Only the level-3 GEMM entries of the fixed non-DYNAMIC_ARCH build. */

#define	I8GEMM_ONCOPY		i8gemm_oncopy
#define	I8GEMM_OTCOPY		i8gemm_otcopy

/* The generic 2x2 kernel unrolls M and N equally, so the inner copies
   alias the outer ones, as DGEMM does when its unrolls match. */
#define	I8GEMM_INCOPY		i8gemm_oncopy
#define	I8GEMM_ITCOPY		i8gemm_otcopy

#define	I8GEMM_BETA		i8gemm_beta
#define	I8GEMM_KERNEL		i8gemm_kernel

#define	I8GEMM_NN		i8gemm_nn
#define	I8GEMM_CN		i8gemm_tn
#define	I8GEMM_TN		i8gemm_tn
#define	I8GEMM_NC		i8gemm_nt
#define	I8GEMM_NT		i8gemm_nt
#define	I8GEMM_CC		i8gemm_tt
#define	I8GEMM_CT		i8gemm_tt
#define	I8GEMM_TC		i8gemm_tt
#define	I8GEMM_TT		i8gemm_tt
#define	I8GEMM_NR		i8gemm_nn
#define	I8GEMM_TR		i8gemm_tn
#define	I8GEMM_CR		i8gemm_tn
#define	I8GEMM_RN		i8gemm_nn
#define	I8GEMM_RT		i8gemm_nt
#define	I8GEMM_RC		i8gemm_nt
#define	I8GEMM_RR		i8gemm_nn

/* Unused under USE_SIMPLE_THREADED_LEVEL3; kept for table parity. */
#define	I8GEMM_THREAD_NN	i8gemm_thread_nn
#define	I8GEMM_THREAD_CN	i8gemm_thread_tn
#define	I8GEMM_THREAD_TN	i8gemm_thread_tn
#define	I8GEMM_THREAD_NC	i8gemm_thread_nt
#define	I8GEMM_THREAD_NT	i8gemm_thread_nt
#define	I8GEMM_THREAD_CC	i8gemm_thread_tt
#define	I8GEMM_THREAD_CT	i8gemm_thread_tt
#define	I8GEMM_THREAD_TC	i8gemm_thread_tt
#define	I8GEMM_THREAD_TT	i8gemm_thread_tt
#define	I8GEMM_THREAD_NR	i8gemm_thread_nn
#define	I8GEMM_THREAD_TR	i8gemm_thread_tn
#define	I8GEMM_THREAD_CR	i8gemm_thread_tn
#define	I8GEMM_THREAD_RN	i8gemm_thread_nn
#define	I8GEMM_THREAD_RT	i8gemm_thread_nt
#define	I8GEMM_THREAD_RC	i8gemm_thread_nt
#define	I8GEMM_THREAD_RR	i8gemm_thread_nn

#endif
