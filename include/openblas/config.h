#pragma once

/* Fixed configuration for the extracted Linux x86-64 pthread DGEMM build. */
#if !defined(__linux__) || !defined(__x86_64__)
#error "This OpenBLAS subset requires Linux x86-64."
#endif

#define OS_LINUX	1
#define ARCH_X86_64	1
/* Clang defines __GNUC__, but must not enable old GCC atomic workarounds. */
#ifdef __clang__
#define C_CLANG 1
#else
#define C_GCC 1
#endif
#define __64BIT__	1
#define FUNDERSCORE	
#define BUNDERSCORE _
#define NEEDBUNDERSCORE 1
#define GENERIC
#define L1_DATA_SIZE 32768
#define L1_DATA_LINESIZE 128
#define L2_SIZE 512488
#define L2_LINESIZE 128
#define DTB_DEFAULT_ENTRIES 128
#define DTB_SIZE 4096
#define L2_ASSOCIATIVE 8
#define CORE_generic
#define CHAR_CORENAME "generic"
#define SLOCAL_BUFFER_SIZE	4096
#define DLOCAL_BUFFER_SIZE	4096
#define CLOCAL_BUFFER_SIZE	8192
#define ZLOCAL_BUFFER_SIZE	8192
#define GEMM_MULTITHREAD_THRESHOLD	4

#define HAVE_C11
#define SMP_SERVER
#define MAX_CPU_NUMBER 4
#define MAX_PARALLEL_NUMBER 1
#define USE_SIMPLE_THREADED_LEVEL3
#define NO_AFFINITY
#define NO_LAPACK
#define NO_LAPACKE
#define NO_WARMUP
#define BUILD_DOUBLE
#define MAX_STACK_ALLOC 2048
#define VERSION "0.3.34.dev"
