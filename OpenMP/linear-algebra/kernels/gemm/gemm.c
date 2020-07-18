/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 *
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE) i*j) / ni;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#ifndef OMP_OFFLOAD
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;
  #pragma scop
  #pragma omp parallel
  {
    /* C := alpha*A*B + beta*C */
    #pragma omp for private (j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
	{
	  C[i][j] *= beta;
	  for (k = 0; k < _PB_NK; ++k)
	    C[i][j] += alpha * A[i][k] * B[k][j];
	}
  }
  #pragma endscop
}
#elif defined POLYBENCH_OFFLOAD1D
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D_1D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D_1D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D_1D(B,NK,NJ,nk,nj))
{
  int i, j, k;
#define C_IDX(i,j) IDX2(C,i,j,ni,nj)
#define A_IDX(i,j) IDX2(A,i,j,ni,nk)
#define B_IDX(i,j) IDX2(B,i,j,nk,nj)
  #pragma scop
  {
    /* C := alpha*A*B + beta*C */
    #pragma omp target data map(to: A[:ni*nk], B[:nk*nj]) \
      map(tofrom: C[:ni*nj])
    #pragma omp target teams distribute parallel for private (k) collapse(2)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
	{
	  GET_IDX2(C,i,j) *= beta;
	  for (k = 0; k < _PB_NK; ++k)
	    GET_IDX2(C,i,j) += alpha * GET_IDX2(A,i,k) * GET_IDX2(B,k,j);
	}
  }
  #pragma endscop
}
#else
static
void kernel_gemm(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;
  #pragma scop
  {
    /* C := alpha*A*B + beta*C */
    #pragma omp target data map(to: A[:ni][:nk], B[:nk][:nj]) \
      map(tofrom: C[:ni][:nj])
    #pragma omp target teams distribute parallel for private (k) collapse(2)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
	{
	  C[i][j] *= beta;
	  for (k = 0; k < _PB_NK; ++k)
	    C[i][j] += alpha * A[i][k] * B[k][j];
	}
  }
  #pragma endscop
}
#endif


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gemm (ni, nj, nk,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
