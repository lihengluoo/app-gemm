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
named_names := cblas_i8gemm i8gemm_nn i8gemm_nt i8gemm_tn i8gemm_tt gemm_thread_n \
               i8gemm_kernel i8gemm_oncopy i8gemm_otcopy i8gemm_beta
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
$(eval $(call compile_named,i8gemm_nn,lib/driver/level3/gemm.c,-DINT8 -DNN,lib/driver/level3/level3.c))
$(eval $(call compile_named,i8gemm_nt,lib/driver/level3/gemm.c,-DINT8 -DNT,lib/driver/level3/level3.c))
$(eval $(call compile_named,i8gemm_tn,lib/driver/level3/gemm.c,-DINT8 -DTN,lib/driver/level3/level3.c))
$(eval $(call compile_named,i8gemm_tt,lib/driver/level3/gemm.c,-DINT8 -DTT,lib/driver/level3/level3.c))
$(eval $(call compile_named,gemm_thread_n,lib/driver/level3/gemm_thread_n.c))
$(eval $(call compile_named,i8gemm_kernel,lib/kernel/generic/gemmkernel_2x2.c,-DINT8))
$(eval $(call compile_named,i8gemm_oncopy,lib/kernel/generic/gemm_ncopy_2.c,-DINT8))
$(eval $(call compile_named,i8gemm_otcopy,lib/kernel/generic/gemm_tcopy_2.c,-DINT8))
$(eval $(call compile_named,i8gemm_beta,lib/kernel/generic/gemm_beta.c,-DINT8))

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
