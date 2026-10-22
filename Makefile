CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -g
build_dir ?= build

.DEFAULT_GOAL := all

library_cppflags := -Iinclude/openblas
library_cflags := -std=c11 -Wall -m64 -fPIC -pthread
test_cflags := -std=c11 -Wall -Wextra -Werror -Wpedantic \
               -Wstrict-prototypes -Wmissing-prototypes -pthread

runtime_names := blas_server blas_server_callback divtable memory \
                 openblas_cancel openblas_env openblas_error_handle parameter xerbla
named_names := cblas_dgemm dgemm_nn dgemm_nt dgemm_tn dgemm_tt gemm_thread_n \
               dgemm_kernel dgemm_oncopy dgemm_otcopy dgemm_beta
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

$(eval $(call compile_named,cblas_dgemm,lib/interface/gemm.c,-DDOUBLE -DCBLAS))
$(eval $(call compile_named,dgemm_nn,lib/driver/level3/gemm.c,-DDOUBLE -DNN,lib/driver/level3/level3.c))
$(eval $(call compile_named,dgemm_nt,lib/driver/level3/gemm.c,-DDOUBLE -DNT,lib/driver/level3/level3.c))
$(eval $(call compile_named,dgemm_tn,lib/driver/level3/gemm.c,-DDOUBLE -DTN,lib/driver/level3/level3.c))
$(eval $(call compile_named,dgemm_tt,lib/driver/level3/gemm.c,-DDOUBLE -DTT,lib/driver/level3/level3.c))
$(eval $(call compile_named,gemm_thread_n,lib/driver/level3/gemm_thread_n.c))
$(eval $(call compile_named,dgemm_kernel,lib/kernel/generic/gemmkernel_2x2.c,-DDOUBLE))
$(eval $(call compile_named,dgemm_oncopy,lib/kernel/generic/gemm_ncopy_2.c,-DDOUBLE))
$(eval $(call compile_named,dgemm_otcopy,lib/kernel/generic/gemm_tcopy_2.c,-DDOUBLE))
$(eval $(call compile_named,dgemm_beta,lib/kernel/generic/gemm_beta.c,-DDOUBLE))

$(build_dir)/gemm_check.o: tests/gemm_check.c tests/gemm_check.h include/cblas.h Makefile
	mkdir -p "$(@D)"
	$(CC) $(CPPFLAGS) -Iinclude $(CFLAGS) $(test_cflags) -MMD -MP -c $< -o $@

$(check_program): $(build_dir)/gemm_check.o $(library)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -Wl,--wrap=exec_blas -pthread -lm $(LDLIBS) -o $@

test: $(check_program)
	OPENBLAS_NUM_THREADS=4 $(check_program)

run: test

clean:
	rm -rf "$(build_dir)"

-include $(library_objects:.o=.d) $(build_dir)/gemm_check.d
