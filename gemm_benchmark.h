#pragma once

#include "gemm.h"

#ifndef benchmark_size
#define benchmark_size 4096
#endif
#ifndef benchmark_threads
#define benchmark_threads 128
#endif

_Static_assert(benchmark_size > 0, "gemm_size must be positive");
_Static_assert(benchmark_threads > 0 && benchmark_threads <= max_gemm_threads,
               "gemm_threads must be between 1 and 128");
_Static_assert((size_t)benchmark_size <= SIZE_MAX / benchmark_size /
                                             sizeof(int32_t),
               "gemm_size exceeds the addressable matrix size");
