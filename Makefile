CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -g
build_dir ?= build

.DEFAULT_GOAL := all

library_cppflags := -Iinclude/openblas
library_cflags := -std=c11 -Wall -m64 -fPIC -pthread
test_cflags := -std=c11 -Wall -Wextra -Werror -Wpedantic \
               -Wstrict-prototypes -Wmissing-prototypes -pthread

runtime_names := blas_server divtable memory \
                 openblas_cancel openblas_env openblas_error_handle parameter xerbla
# Kernel backend: generic (portable scalar 2x2) or 4509y (AVX-512 VNNI,
# Intel 4509Y class).  The AVX-512 flags apply only to the backend objects;
# the rest of the library stays baseline x86-64.
kernel_backend ?= generic

ifeq ($(kernel_backend),4509y)
kernel_src := lib/kernel/4509y/gemm_kernel_4509y.c
kernel_flags := -DINT8 -mavx512f -mavx512bw -mavx512vl -mavx512vnni
driver_kernel_flags := -DI8GEMM_SIMD_PACK
kernel_copy_names := i8gemm_copy_4509y cpu_guard_4509y
else
kernel_src := lib/kernel/generic/gemmkernel_2x2.c
kernel_flags := -DINT8
driver_kernel_flags :=
kernel_copy_names := i8gemm_oncopy i8gemm_otcopy
endif

named_names := cblas_i8gemm i8gemm_nn i8gemm_nt i8gemm_tn i8gemm_tt gemm_thread_n \
               i8gemm_kernel i8gemm_beta $(kernel_copy_names)
library_objects := $(addprefix $(build_dir)/,$(addsuffix .o,$(runtime_names) $(named_names)))
library := $(build_dir)/libopenblas_gemm.a
check_program := $(build_dir)/gemm_check

# These names are required by the OpenBLAS source templates.
named_cppflags = -DASMNAME=$(1) -DASMFNAME=$(1)_ -DNAME=$(1)_ -DCNAME=$(1) \
                 -DCHAR_NAME='"$(1)_"' -DCHAR_CNAME='"$(1)"'

.PHONY: all test run clean

all: $(library) $(check_program)

$(library): $(library_objects)
	mkdir -p "$(@D)"
	$(AR) rcs $@ $^

$(build_dir)/%.o: lib/driver/others/%.c Makefile
	mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) $(library_cppflags) $(CFLAGS) $(library_cflags) \
	    -MMD -MP -c $< -o $@

define compile_named
$(build_dir)/$(1).o: $(2) $(4) Makefile
	mkdir -p "$$(@D)"
	$$(CC) $$(CPPFLAGS) $$(library_cppflags) $$(CFLAGS) $$(library_cflags) \
	    $(3) $(call named_cppflags,$(1)) -MMD -MP -c $$< -o $$@
endef

$(eval $(call compile_named,cblas_i8gemm,lib/interface/gemm.c,-DINT8 -DCBLAS))
$(eval $(call compile_named,i8gemm_nn,lib/driver/level3/gemm.c,-DINT8 -DNN $(driver_kernel_flags),lib/driver/level3/level3.c))
$(eval $(call compile_named,i8gemm_nt,lib/driver/level3/gemm.c,-DINT8 -DNT $(driver_kernel_flags),lib/driver/level3/level3.c))
$(eval $(call compile_named,i8gemm_tn,lib/driver/level3/gemm.c,-DINT8 -DTN $(driver_kernel_flags),lib/driver/level3/level3.c))
$(eval $(call compile_named,i8gemm_tt,lib/driver/level3/gemm.c,-DINT8 -DTT $(driver_kernel_flags),lib/driver/level3/level3.c))
$(eval $(call compile_named,gemm_thread_n,lib/driver/level3/gemm_thread_n.c))
$(eval $(call compile_named,i8gemm_kernel,$(kernel_src),$(kernel_flags)))
ifeq ($(kernel_backend),4509y)
$(eval $(call compile_named,i8gemm_copy_4509y,lib/kernel/4509y/gemm_copy_4509y.c,$(kernel_flags)))
$(eval $(call compile_named,cpu_guard_4509y,lib/kernel/4509y/cpu_guard_4509y.c,))
else
$(eval $(call compile_named,i8gemm_oncopy,lib/kernel/generic/gemm_ncopy_2.c,-DINT8))
$(eval $(call compile_named,i8gemm_otcopy,lib/kernel/generic/gemm_tcopy_2.c,-DINT8))
endif
$(eval $(call compile_named,i8gemm_beta,lib/kernel/generic/gemm_beta.c,-DINT8))

$(build_dir)/gemm_check.o: tests/gemm_check.c tests/gemm_check.h include/cblas.h Makefile
	mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) -Iinclude $(CFLAGS) $(test_cflags) -MMD -MP -c $< -o $@

$(check_program): $(build_dir)/gemm_check.o $(library)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -Wl,--wrap=exec_blas -pthread -lm $(LDLIBS) -o $@

test: $(check_program)
	@if [ "$(kernel_backend)" = "4509y" ] && ! grep -qw avx512_vnni /proc/cpuinfo; then \
	  echo "SKIP: this host cannot run kernel_backend=4509y (no AVX-512 VNNI)"; \
	  exit 77; \
	fi
	OPENBLAS_NUM_THREADS=4 $(check_program)

run: test

clean:
	rm -rf "$(build_dir)"

-include $(library_objects:.o=.d) $(build_dir)/gemm_check.d
