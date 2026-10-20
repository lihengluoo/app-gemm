#include "gemm_fault_test.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

_Atomic unsigned create_calls = 0;
_Atomic unsigned join_calls = 0;
_Atomic unsigned allocation_calls = 0;
unsigned fail_create = 0;
unsigned fail_allocation = 0;
bool fail_join = false;

int __wrap_pthread_create(pthread_t *handle, const pthread_attr_t *attrs,
                          void *(*entry)(void *), void *args)
{
    const unsigned index = ++create_calls;
    return index == fail_create ?
               EAGAIN :
               __real_pthread_create(handle, attrs, entry, args);
}

int __wrap_pthread_join(pthread_t handle, void **result)
{
    ++join_calls;
    return fail_join ? EINVAL : __real_pthread_join(handle, result);
}

void *__wrap_aligned_alloc(size_t alignment, size_t bytes)
{
    const unsigned index = ++allocation_calls;
    return index == fail_allocation ? NULL :
                                      __real_aligned_alloc(alignment, bytes);
}

void require_gemm_fault(bool condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "%s: %s\n", get_gemm_cpu_name(), message);
        exit(1);
    }
}

gemm_status compute_gemm_with_fault(unsigned create_index,
                                    unsigned allocation_index)
{
    fail_create = create_index;
    fail_allocation = allocation_index;
    create_calls = 0;
    join_calls = 0;
    allocation_calls = 0;
    int16_t a[8 * 7] = {0};
    int16_t b[7 * 17] = {0};
    int32_t c[8 * 17] = {0};
    gemm_args args = get_default_gemm_args();
    args.m = 8;
    args.n = 17;
    args.k = 7;
    args.lda = 7;
    args.ldb = args.ldc = 17;
    args.a = a;
    args.b = b;
    args.c = c;
    gemm_options options = get_default_gemm_options();
    options.thread_count = 4;
    return compute_gemm_i16_i32(&args, &options);
}

int main(void)
{
    if (!is_gemm_cpu_supported()) {
        return 0;
    }
    require_gemm_fault(compute_gemm_with_fault(0, 1) == gemm_no_memory &&
                           create_calls == 0,
                       "B allocation failure starts workers");
    // Worker A panels live on their stacks, so no per-worker allocation can
    // fail; verify the join discipline on the success path instead.
    require_gemm_fault(compute_gemm_with_fault(0, 0) == gemm_success &&
                           create_calls == 4 && join_calls == 4,
                       "successful run does not join every worker");
    for (unsigned index = 1; index <= 4; ++index) {
        require_gemm_fault(
            compute_gemm_with_fault(index, 0) == gemm_thread_create_failed &&
                join_calls == index - 1,
            "partial creation did not join its successful prefix");
    }
    // A join failure must not return and release C/descriptor/packed-B memory.
    const pid_t child = fork();
    require_gemm_fault(child >= 0, "fork failed");
    if (child == 0) {
        const struct rlimit no_core = {0, 0};
        setrlimit(RLIMIT_CORE, &no_core);
        fail_join = true;
        compute_gemm_with_fault(0, 0);
        _exit(2);
    }
    int status;
    require_gemm_fault(waitpid(child, &status, 0) == child &&
                           WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
                       "join failure returned with possibly live workers");
    printf("%s: allocation/create/join failure checks passed\n",
           get_gemm_cpu_name());
    return 0;
}
