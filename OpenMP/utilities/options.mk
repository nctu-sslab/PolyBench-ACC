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
ACCFLAGS =  -fopenmp -I(LLVM_BUILD_INC)

# COMPILER OPTIONS -- HOST
########################################

# Compiler
CC = clang

# Compiler flags
CFLAGS = -O2 -lm
