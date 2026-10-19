#pragma once

#include "gemm.h"
#include <stdatomic.h>
#include <pthread.h>

extern _Atomic unsigned create_calls;
extern _Atomic unsigned join_calls;
extern _Atomic unsigned allocation_calls;
extern unsigned fail_create;
extern unsigned fail_allocation;
extern bool fail_join;

int __real_pthread_create(pthread_t *, const pthread_attr_t *,
                          void *(*)(void *), void *);
int __wrap_pthread_create(pthread_t *, const pthread_attr_t *,
                          void *(*)(void *), void *);
int __real_pthread_join(pthread_t, void **);
int __wrap_pthread_join(pthread_t, void **);
void *__real_aligned_alloc(size_t, size_t);
void *__wrap_aligned_alloc(size_t, size_t);
void require_gemm_fault(bool condition, const char *message);
gemm_status compute_gemm_with_fault(unsigned create_index,
                                    unsigned allocation_index);
