INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

SCRIPT      := run

ifdef OFFLOAD
OMPOFFLOADFLAGS := -DOMP_OFFLOAD -fopenmp-targets=nvptx64 -DPOLYBENCH_DYNAMIC_ARRAYS
CFLAGS      += $(OMPOFFLOADFLAGS)
CXXFLAGS    += $(OMPOFFLOADFLAGS)
LDLIBS      += $(OMPOFFLOADFLAGS)
endif

ifdef RUN_MINI
$(warning "Run-Mini Enabled")
FLAGS       := -DMINI_DATASET
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
else ifdef RUN_LARGE
$(warning "Run-Large Enabled")
FLAGS       := -DEXTRALARGE_DATASET
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
endif

ifdef VERIFY
$(warning "Verify Enabled")
FLAGS       := -DPOLYBENCH_DUMP_ARRAYS
CFLAGS      := $(CFLAGS) $(FLAGS)
CXXFLAGS    := $(CXXFLAGS) $(FLAGS)
SCRIPT      += verify
endif

DEPS        := Makefile.dep
DEP_FLAG    := -MM

.PHONY: all exe clean veryclean

all : exe $(SCRIPT)

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
