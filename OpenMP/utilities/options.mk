# CODE GENERATION OPTIONS
########################################

# Default OpenACC Target is OpenCL
TARGET_LANG = OPENCL

# Uncomment if you want CUDA
# TARGET_LANG = CUDA

# COMPILER OPTIONS -- ACCELERATOR
########################################

# Accelerator Compiler
#ACC = clang

# Accelerator Compiler flags
#ACCFLAGS = --codelet-required --openacc-target=$(TARGET_LANG)
ifdef RUN_NOOMP
$(warning NO OMP Enbaled)
else
ACCFLAGS := -fopenmp
endif
ACCFLAGS += -I$(LLVM_BUILD_INC)

# COMPILER OPTIONS -- HOST
########################################

# Compiler
CC = clang
#CC = gcc-8

# Compiler flags
CFLAGS = -O2 -lm
