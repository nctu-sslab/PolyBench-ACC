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
#include "3mm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE) i*j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#ifndef OMP_OFFLOAD
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;
  #pragma scop
  #pragma omp parallel private (j, k)
  {
    /* E := A*B */
    #pragma omp for
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
	{
          E[i][j] = 0;
	  for (k = 0; k < _PB_NK; ++k)
	    E[i][j] += A[i][k] * B[k][j];
        }
    /* F := C*D */
    #pragma omp for
    for (i = 0; i < _PB_NJ; i++)
      for (j = 0; j < _PB_NL; j++)
	{
	  F[i][j] = 0;
	  for (k = 0; k < _PB_NM; ++k)
	    F[i][j] += C[i][k] * D[k][j];
        }
    /* G := E*F */
    #pragma omp for
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NL; j++)
	{
	  G[i][j] = 0;
	  for (k = 0; k < _PB_NJ; ++k)
	    G[i][j] += E[i][k] * F[k][j];
	}
  }
}
#elif defined POLYBENCH_OFFLOAD1D
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D_1D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D_1D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D_1D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D_1D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D_1D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D_1D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D_1D(G,NI,NL,ni,nl))
{
  int i, j, k;
#define GET_IDX2(Name, i, j) Name##_IDX(i,j)
#define E_IDX(i,j) IDX2(E, i,j,ni,nj)
#define A_IDX(i,j) IDX2(A, i,j,ni,nk)
#define B_IDX(i,j) IDX2(B, i,j,nk,nj)
#define F_IDX(i,j) IDX2(F, i,j,nj,nl)
#define C_IDX(i,j) IDX2(C, i,j,nj,nm)
#define D_IDX(i,j) IDX2(D, i,j,nm,nl)
#define G_IDX(i,j) IDX2(G, i,j,ni,nl)
#pragma omp target data map(to: A[:NI*NK], B[:NK*NJ], C[:NJ*NM], \
        D[:NM*NL], E[:NI*NJ], F[:NJ*NL]) map(tofrom: G[:NI*NL])
  {
    /* E := A*B */
    #pragma omp target teams distribute parallel for private (j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
	{
          GET_IDX2(E,i,j) = 0;
	  for (k = 0; k < _PB_NK; ++k)
	    GET_IDX2(E,i,j) += GET_IDX2(A,i,k) * GET_IDX2(B,k,j);
        }
    /* F := C*D */
    #pragma omp target teams distribute parallel for private (j, k)
    for (i = 0; i < _PB_NJ; i++)
      for (j = 0; j < _PB_NL; j++)
	{
	  GET_IDX2(F,i,j) = 0;
	  for (k = 0; k < _PB_NM; ++k)
	    GET_IDX2(F,i,j) += GET_IDX2(C,i,k) * GET_IDX2(D,k,j);
        }
    /* G := E*F */
    #pragma omp target teams distribute parallel for private (j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NL; j++)
	{
	  GET_IDX2(G,i,j) = 0;
	  for (k = 0; k < _PB_NJ; ++k)
	    GET_IDX2(G,i,j) += GET_IDX2(E,i,k) * GET_IDX2(F,k,j);
	}
  }
}
#else
static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j, k;
#ifdef OMP_DCAT
#pragma omp target data map(to: A, B, C, D, E, F) map(tofrom: G)
#else
#pragma omp target data map(to: A[:NI][:NK], B[:NK][:NJ], C[:NJ][:NM], \
        D[:NM][:NL], E[:NI][:NJ], F[:NJ][:NL]) map(tofrom: G[:NI][:NL])
#endif
  {
    /* E := A*B */
    #pragma omp target teams distribute parallel for private (j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NJ; j++)
	{
          E[i][j] = 0;
	  for (k = 0; k < _PB_NK; ++k)
	    E[i][j] += A[i][k] * B[k][j];
        }
    /* F := C*D */
    #pragma omp target teams distribute parallel for private (j, k)
    for (i = 0; i < _PB_NJ; i++)
      for (j = 0; j < _PB_NL; j++)
	{
	  F[i][j] = 0;
	  for (k = 0; k < _PB_NM; ++k)
	    F[i][j] += C[i][k] * D[k][j];
        }
    /* G := E*F */
    #pragma omp target teams distribute parallel for private (j, k)
    for (i = 0; i < _PB_NI; i++)
      for (j = 0; j < _PB_NL; j++)
	{
	  G[i][j] = 0;
	  for (k = 0; k < _PB_NJ; ++k)
	    G[i][j] += E[i][k] * F[k][j];
	}
  }
}
#endif

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
  DC_END();

  /* Initialize array(s). */
  init_array (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D));


  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm (ni, nj, nk, nl, nm,
	      POLYBENCH_ARRAY(E),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(F),
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(D),
	      POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl,  POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}
