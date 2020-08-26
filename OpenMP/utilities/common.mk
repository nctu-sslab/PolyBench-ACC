INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

SCRIPT      := run verify

ifdef POLY1D
$(warning "Poly-1D Enabled")
OFFLOAD=1
endif

ifdef OFFLOAD
$(warning "Offloading Enabled")
OMPOFFLOADFLAGS := -DOMP_OFFLOAD -fopenmp-targets=nvptx64
ifdef POLY1D
OMPOFFLOADFLAGS += -DPOLYBENCH_OFFLOAD1D -Wno-incompatible-pointer-types
else
OMPOFFLOADFLAGS += -DPOLYBENCH_DYNAMIC_ARRAYS
endif
ifdef OMP_MASK
$(warning "MASK Enabled")
OMPOFFLOADFLAGS += -DOMP_MASK -DOMP_DCAT
OMPOFFLOADFLAGS += -L $(LLVM_BUILD_PATH)/lib
OMPOFFLOADFLAGS += -lomptarget
endif
ifdef OMP_OFFSET
$(warning "OFFSET Enabled")
OMPOFFLOADFLAGS += -DOMP_OFFSET -DOMP_DCAT
OMPOFFLOADFLAGS += -L $(LLVM_BUILD_PATH)/lib
OMPOFFLOADFLAGS += -lomptarget
endif
ifdef OMP_UVM
$(warning "UVM Enabled")
OMPOFFLOADFLAGS += -DOMP_DCAT
OMPOFFLOADFLAGS += -L $(LLVM_BUILD_PATH)/lib
OMPOFFLOADFLAGS += -lomptarget
endif
CFLAGS      += $(OMPOFFLOADFLAGS)
CXXFLAGS    += $(OMPOFFLOADFLAGS)
LDLIBS      += $(OMPOFFLOADFLAGS)
endif


ifdef VERIFY
RUN_MINI=1
RUN_DUMP=1
endif

ifdef RUN_MINI
$(warning "Run-Mini Enabled")
FLAGS       := -DMINI_DATASET
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
else ifdef RUN_EXTRALARGE
$(warning "Run-Extra-Large Enabled")
FLAGS       := -DEXTRALARGE_DATASET
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
else ifdef RUN_LARGE
$(warning "Run-Large Enabled")
FLAGS       := -DLARGE_DATASET
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
endif

ifdef RUN_DUMP
$(warning "Array Dump Enabled")
FLAGS       := -DPOLYBENCH_DUMP_ARRAYS
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
endif

ifdef RUN_DYN
$(warning "Dynamic Array Enabled")
FLAGS       := -DPOLYBENCH_DYNAMIC_ARRAYS
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
endif

DEPS        := Makefile.dep
DEP_FLAG    := -MM

.PHONY: all exe clean veryclean

all : $(SCRIPT) exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(ACCFLAGS) $(CFLAGS) $(INCPATHS) $^ -o $@

clean :
	-rm -vf __hmpp* -vf $(EXE) *~ $(SCRIPT) $(DEPS) output.txt diff.txt

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)

run:
	@ln -s $(UTIL_DIR)/run run
	@echo "Install script run"

# TODO
verify:
	@ln -s $(UTIL_DIR)/verify verify
	@echo "Install script verify"
