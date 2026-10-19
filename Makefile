CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -g

root_dir := $(abspath ../..)
llvm_root ?= $(root_dir)/dependencies/install/llvm21
clang := $(llvm_root)/bin/clang
wllvm ?= $(root_dir)/dependencies/install/wllvm/bin/wllvm
build_dir ?= build
ir_dir ?= $(build_dir)/ir
gemm_backend ?= scalar

backends := scalar 6780e_avx 4509y_avx
cpu_scalar := 0
cpu_6780e_avx := 1
cpu_4509y_avx := 2
kernel_scalar := lib/gemm_scalar.c
kernel_6780e_avx := lib/gemm_6780e.c
kernel_4509y_avx := lib/gemm_4509y.c
common_files := lib/gemm.c lib/gemm_pack.c lib/gemm_threads.c

ifeq ($(kernel_$(gemm_backend)),)
$(error Unknown GEMM backend: $(gemm_backend))
endif

include_flags := -D_POSIX_C_SOURCE=200809L -Iinclude
# GCC and Clang both accept this spelling; explicit SIMD intrinsics remain.
common_flags := -std=c11 -Wall -Wextra -Werror -Wpedantic \
                -Wstrict-prototypes -Wmissing-prototypes -fno-tree-vectorize
fault_flags := -Wl,--wrap=pthread_create -Wl,--wrap=pthread_join \
               -Wl,--wrap=aligned_alloc
check_programs := $(addsuffix /gemm_check,$(addprefix $(build_dir)/,$(backends)))
fault_programs :=
ifeq ($(shell uname -s),Linux)
fault_programs := $(addsuffix /gemm_fault_check,$(addprefix $(build_dir)/,$(backends)))
endif

.PHONY: all benchmark tests test ir clean

all: benchmark tests

benchmark: $(build_dir)/$(gemm_backend)/gemm_benchmark

tests: $(check_programs) $(fault_programs)

# Each backend has its own objects, library, and executables, so switching the
# selected benchmark cannot reuse objects compiled for a different tile size.
define backend_rules
$(build_dir)/$(1)/%.o: %.c include/gemm.h Makefile
	mkdir -p $$(@D)
	$$(CC) $$(CPPFLAGS) $$(include_flags) $$(CFLAGS) $$(common_flags) \
	    -Dgemm_cpu=$(cpu_$(1)) -pthread -MMD -MP -c $$< -o $$@

$(build_dir)/$(1)/libgemm.a: $(addprefix $(build_dir)/$(1)/,$(common_files:.c=.o) $(kernel_$(1):.c=.o))
	$$(AR) rcs $$@ $$^

$(build_dir)/$(1)/gemm_benchmark: $(build_dir)/$(1)/gemm_benchmark.o $(build_dir)/$(1)/libgemm.a
	$$(CC) $$(CFLAGS) $$(LDFLAGS) $$^ -pthread $$(LDLIBS) -o $$@

$(build_dir)/$(1)/gemm_check: $(build_dir)/$(1)/tests/gemm_check.o $(build_dir)/$(1)/libgemm.a
	$$(CC) $$(CFLAGS) $$(LDFLAGS) $$^ -pthread $$(LDLIBS) -o $$@

$(build_dir)/$(1)/gemm_fault_check: $(build_dir)/$(1)/tests/gemm_fault_check.o $(build_dir)/$(1)/libgemm.a
	$$(CC) $$(CFLAGS) $$(LDFLAGS) $$^ -pthread $$(fault_flags) $$(LDLIBS) -o $$@
endef
$(foreach backend,$(backends),$(eval $(call backend_rules,$(backend))))

-include $(wildcard $(build_dir)/*/*.d $(build_dir)/*/lib/*.d $(build_dir)/*/tests/*.d)

test: all
	@$(build_dir)/$(gemm_backend)/gemm_benchmark; status=$$?; \
	    if [ "$$status" -eq 0 ]; then :; \
	    elif [ "$$status" -eq 77 ]; then \
	        echo "$(gemm_backend): benchmark skipped; SIMD host unavailable"; \
	    else exit "$$status"; fi
	@set -e; for program in $(check_programs) $(fault_programs); do \
	    "$$program"; \
	done

ir:
	./build.sh --clang "$(clang)" --wllvm "$(wllvm)" \
	    --backend "$(gemm_backend)" --output-dir "$(ir_dir)"

clean:
	rm -rf "$(build_dir)"
