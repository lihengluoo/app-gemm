/* CPU guard for kernel_backend=4509y builds.
 *
 * This translation unit is compiled without AVX-512 flags so
 * __builtin_cpu_supports stays a genuine runtime test instead of folding
 * to a constant.  The constructor aborts before any GEMM call when the
 * host cannot execute the AVX-512 VNNI kernel. */

#include <stdio.h>
#include <stdlib.h>

static void check_cpu_4509y(void) __attribute__((constructor));

static void check_cpu_4509y(void) {

  if (__builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512vnni")) return;

  fprintf(stderr,
          "i8gemm: the 4509y kernel needs avx512f, avx512bw and avx512vnni; "
          "rebuild with kernel_backend=generic for this host\n");
  exit(86);
}
