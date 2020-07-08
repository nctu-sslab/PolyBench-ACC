INCPATHS = -I$(UTIL_DIR)

BENCHMARK = $(shell basename `pwd`)
EXE = $(BENCHMARK)_acc
SRC = $(BENCHMARK).c
HEADERS = $(BENCHMARK).h

SRC += $(UTIL_DIR)/polybench.c

ifdef OFFLOAD
OMPOFFLOADFLAGS = -DOMP_OFFLOAD -fopenmp-targets=nvptx64
CXXFLAGS += $(OMPOFFLOADFLAGS)
CFLAGS += $(OMPOFFLOADFLAGS)
LDLIBS += $(OMPOFFLOADFLAGS)
endif

ifdef DYN
FLAGS = -DPOLYBENCH_DYNAMIC_ARRAYS -DMINI_DATASET
# EXTRALARGE_DATASET LARGE_DATASET
CFLAGS += $(FLAGS)
CXXFLAGS += $(FLAGS)
endif

ifdef PRINT
FLAGS = -DPOLYBENCH_DUMP_ARRAYS
CFLAGS += $(FLAGS)
CXXFLAGS += $(FLAGS)
endif


DEPS        := Makefile.dep
DEP_FLAG    := -MM

.PHONY: all exe clean veryclean

all : exe install

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(ACCFLAGS) $(CFLAGS) $(INCPATHS) $^ -o $@

clean :
	-rm -vf __hmpp* -vf $(EXE) *~ run verify $(DEPS)

veryclean : clean
	-rm -vf $(DEPS)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)

install: run

run:
	ln -s $(UTIL_DIR)/run run
	#ln -s $(UTIL_DIR)/verify verify
