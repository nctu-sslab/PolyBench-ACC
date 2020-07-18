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
/* Default data type is int, default size is 50. */
#include "reg_detect.h"


/* Array initialization. */
static
void init_array(int maxgrid,
		DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
		DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
		DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid))
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++) {
      sum_tang[i][j] = (DATA_TYPE)((i+1)*(j+1));
      mean[i][j] = ((DATA_TYPE) i-j) / maxgrid;
      path[i][j] = ((DATA_TYPE) i*(j-1)) / maxgrid;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int maxgrid,
		 DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid))
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, path[i][j]);
      if ((i * maxgrid + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Source (modified): http://www.cs.uic.edu/~iluican/reg_detect.c */
#ifndef OMP_OFFLOAD
static
void kernel_reg_detect(int niter, int maxgrid, int length,
		       DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_3D(diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length),
		       DATA_TYPE POLYBENCH_3D(sum_diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length))
{
  int t, i, j, cnt;

  #pragma scop
  {
      for (t = 0; t < _PB_NITER; t++)
      {
        #pragma omp parallel for private (j,i, cnt) collapse(2) schedule(static)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            for (cnt = 0; cnt <= _PB_LENGTH - 1; cnt++)
              diff[j][i][cnt] = sum_tang[j][i];
          }
        #pragma omp parallel for private (i, cnt) collapse(2) schedule(static)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
        {
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            sum_diff[j][i][0] = diff[j][i][0];
            for (cnt = 1; cnt <= _PB_LENGTH - 1; cnt++)
              sum_diff[j][i][cnt] = sum_diff[j][i][cnt - 1] + diff[j][i][cnt];
            mean[j][i] = sum_diff[j][i][_PB_LENGTH - 1];
          }
        }
        #pragma omp parallel for
        for (i = 0; i <= _PB_MAXGRID - 1; i++)
          path[0][i] = mean[0][i];
        #pragma omp for private (i) collapse(2) schedule(static)
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
        for (j = 1; j <= _PB_MAXGRID - 1; j++)
          {
            if (i < j) {
              continue;
            }
            path[j][i] = path[j - 1][i - 1] + mean[j][i];
          }
      }
  }
  #pragma endscop
}
#elif POLYBENCH_OFFLOAD1D
static
void kernel_reg_detect(int niter, int maxgrid, int length,
		       DATA_TYPE POLYBENCH_2D_1D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D_1D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D_1D(path,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_3D_1D(diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length),
		       DATA_TYPE POLYBENCH_3D_1D(sum_diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length))
{
  int t, i, j, cnt;
#define sum_tang_IDX(i,j) IDX2(sum_tang,i,j,maxgrid,maxgrid)
#define mean_IDX(i,j) IDX2(mean,i,j,maxgrid,maxgrid)
#define path_IDX(i,j) IDX2(path,i,j,maxgrid,maxgrid)
#define diff_IDX(i,j,k) IDX3(diff,i,j,k,maxgrid,maxgrid,length)
#define sum_diff_IDX(i,j,k) IDX3(sum_diff,i,j,k,maxgrid,maxgrid,length)

  {
#pragma omp target data map(to:sum_tang[:maxgrid*maxgrid], \
        mean[:maxgrid*maxgrid], diff[:maxgrid*maxgrid*length], \
        sum_diff[:maxgrid*maxgrid*length]) \
        map(tofrom: path[:maxgrid*maxgrid])
      for (t = 0; t < _PB_NITER; t++)
      {
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            for (cnt = 0; cnt <= _PB_LENGTH - 1; cnt++)
              GET_IDX3(diff,j,i,cnt) = GET_IDX2(sum_tang,j,i);
          }
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
        {
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            GET_IDX3(sum_diff,j,i,0) = GET_IDX3(diff,j,i,0);
            for (cnt = 1; cnt <= _PB_LENGTH - 1; cnt++)
              GET_IDX3(sum_diff,j,i,cnt) = GET_IDX3(sum_diff,j,i,cnt - 1) + GET_IDX3(diff,j,i,cnt);
            GET_IDX2(mean,j,i) = GET_IDX3(sum_diff,j,i,_PB_LENGTH - 1);
          }
        }
        #pragma omp target teams distribute parallel for
        for (i = 0; i <= _PB_MAXGRID - 1; i++)
          GET_IDX2(path,0,i) = GET_IDX2(mean,0,i);
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
        for (j = 1; j <= _PB_MAXGRID - 1; j++)
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            GET_IDX2(path,j,i) = GET_IDX2(path,j - 1,i - 1) + GET_IDX2(mean,j,i);
          }
      }
  }
}
#else
static
void kernel_reg_detect(int niter, int maxgrid, int length,
		       DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_3D(diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length),
		       DATA_TYPE POLYBENCH_3D(sum_diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length))
{
  int t, i, j, cnt;

  {
#pragma omp target data map(to:sum_tang[:maxgrid][:maxgrid], \
        mean[:maxgrid][:maxgrid], diff[:maxgrid][:maxgrid][:length], \
        sum_diff[:maxgrid][:maxgrid][:length]) \
        map(tofrom: path[:maxgrid][:maxgrid])
      for (t = 0; t < _PB_NITER; t++)
      {
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            for (cnt = 0; cnt <= _PB_LENGTH - 1; cnt++)
              diff[j][i][cnt] = sum_tang[j][i];
          }
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
        {
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            sum_diff[j][i][0] = diff[j][i][0];
            for (cnt = 1; cnt <= _PB_LENGTH - 1; cnt++)
              sum_diff[j][i][cnt] = sum_diff[j][i][cnt - 1] + diff[j][i][cnt];
            mean[j][i] = sum_diff[j][i][_PB_LENGTH - 1];
          }
        }
        #pragma omp target teams distribute parallel for
        for (i = 0; i <= _PB_MAXGRID - 1; i++)
          path[0][i] = mean[0][i];
        #pragma omp target teams distribute parallel for collapse(2)
        for (j = 0; j <= _PB_MAXGRID - 1; j++)
        for (j = 1; j <= _PB_MAXGRID - 1; j++)
          for (i = 0; i <= _PB_MAXGRID - 1; i++)
          {
            if (i < j) {
              continue;
            }
            path[j][i] = path[j - 1][i - 1] + mean[j][i];
          }
      }
  }
}
#endif

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int niter = NITER;
  int maxgrid = MAXGRID;
  int length = LENGTH;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(sum_tang, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
  POLYBENCH_2D_ARRAY_DECL(mean, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
  POLYBENCH_3D_ARRAY_DECL(diff, DATA_TYPE, MAXGRID, MAXGRID, LENGTH, maxgrid, maxgrid, length);
  POLYBENCH_3D_ARRAY_DECL(sum_diff, DATA_TYPE, MAXGRID, MAXGRID, LENGTH, maxgrid, maxgrid, length);

  /* Initialize array(s). */
  init_array (maxgrid,
	      POLYBENCH_ARRAY(sum_tang),
	      POLYBENCH_ARRAY(mean),
	      POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_reg_detect (niter, maxgrid, length,
		     POLYBENCH_ARRAY(sum_tang),
		     POLYBENCH_ARRAY(mean),
		     POLYBENCH_ARRAY(path),
		     POLYBENCH_ARRAY(diff),
		     POLYBENCH_ARRAY(sum_diff));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(maxgrid, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(sum_tang);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(path);
  POLYBENCH_FREE_ARRAY(diff);
  POLYBENCH_FREE_ARRAY(sum_diff);

  return 0;
}
