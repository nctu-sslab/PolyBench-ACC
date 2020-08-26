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
/* Default data type is double, default size is 256x256x256. */
#include "fdtd-apml.h"


/* Array initialization. */
static
void init_array (int cz,
		 int cxm,
		 int cym,
		 DATA_TYPE *mui,
		 DATA_TYPE *ch,
		 DATA_TYPE POLYBENCH_2D(Ax,CZ+1,CYM+1,cz+1,cym+1),
		 DATA_TYPE POLYBENCH_2D(Ry,CZ+1,CYM+1,cz+1,cym+1),
		 DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_1D(czm,CZ+1,cz+1),
		 DATA_TYPE POLYBENCH_1D(czp,CZ+1,cz+1),
		 DATA_TYPE POLYBENCH_1D(cxmh,CXM+1,cxm+1),
		 DATA_TYPE POLYBENCH_1D(cxph,CXM+1,cxm+1),
		 DATA_TYPE POLYBENCH_1D(cymh,CYM+1,cym+1),
		 DATA_TYPE POLYBENCH_1D(cyph,CYM+1,cym+1))
{
  int i, j, k;
  *mui = 2341;
  *ch = 42;
  for (i = 0; i <= cz; i++)
    {
      czm[i] = ((DATA_TYPE) i + 1) / cxm;
      czp[i] = ((DATA_TYPE) i + 2) / cxm;
    }
  for (i = 0; i <= cxm; i++)
    {
      cxmh[i] = ((DATA_TYPE) i + 3) / cxm;
      cxph[i] = ((DATA_TYPE) i + 4) / cxm;
    }
  for (i = 0; i <= cym; i++)
    {
      cymh[i] = ((DATA_TYPE) i + 5) / cxm;
      cyph[i] = ((DATA_TYPE) i + 6) / cxm;
    }

  for (i = 0; i <= cz; i++)
    for (j = 0; j <= cym; j++)
      {
	Ry[i][j] = ((DATA_TYPE) i*(j+1) + 10) / cym;
	Ax[i][j] = ((DATA_TYPE) i*(j+2) + 11) / cym;
	for (k = 0; k <= cxm; k++)
	  {
	    Ex[i][j][k] = ((DATA_TYPE) i*(j+3) + k + 1) / cxm;
	    Ey[i][j][k] = ((DATA_TYPE) i*(j+4) + k + 2) / cym;
	    Hz[i][j][k] = ((DATA_TYPE) i*(j+5) + k + 3) / cz;
	  }
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int cz,
		 int cxm,
		 int cym,
		 DATA_TYPE POLYBENCH_3D(Bza,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		 DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1))
{
  int i, j, k;

  for (i = 0; i <= cz; i++)
    for (j = 0; j <= cym; j++)
      for (k = 0; k <= cxm; k++) {
	fprintf(stderr, DATA_PRINTF_MODIFIER, Bza[i][j][k]);
	fprintf(stderr, DATA_PRINTF_MODIFIER, Ex[i][j][k]);
	fprintf(stderr, DATA_PRINTF_MODIFIER, Ey[i][j][k]);
	fprintf(stderr, DATA_PRINTF_MODIFIER, Hz[i][j][k]);
	if ((i * cxm + j) % 20 == 0) fprintf(stderr, "\n");
      }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
#ifndef OMP_OFFLOAD
static
void kernel_fdtd_apml(int cz,
		      int cxm,
		      int cym,
		      DATA_TYPE mui,
		      DATA_TYPE ch,
		      DATA_TYPE POLYBENCH_2D(Ax,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D(Ry,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D(clf,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_2D(tmp,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Bza,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(czm,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(czp,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(cxmh,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cxph,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cymh,CYM+1,cym+1),
		      DATA_TYPE POLYBENCH_1D(cyph,CYM+1,cym+1))
{
  int iz, iy, ix;

  #pragma scop

  #pragma omp parallel
    {
      #pragma omp for private (iy, ix)
      for (iz = 0; iz < _PB_CZ; iz++)
        {
	  for (iy = 0; iy < _PB_CYM; iy++)
	    {
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  clf[iz][iy] = Ex[iz][iy][ix] - Ex[iz][iy+1][ix] + Ey[iz][iy][ix+1] - Ey[iz][iy][ix];
		  tmp[iz][iy] = (cymh[iy] / cyph[iy]) * Bza[iz][iy][ix] - (ch / cyph[iy]) * clf[iz][iy];
		  Hz[iz][iy][ix] = (cxmh[ix] /cxph[ix]) * Hz[iz][iy][ix]
		    + (mui * czp[iz] / cxph[ix]) * tmp[iz][iy]
		    - (mui * czm[iz] / cxph[ix]) * Bza[iz][iy][ix];
		  Bza[iz][iy][ix] = tmp[iz][iy];
		}
	      clf[iz][iy] = Ex[iz][iy][_PB_CXM] - Ex[iz][iy+1][_PB_CXM] + Ry[iz][iy] - Ey[iz][iy][_PB_CXM];
	      tmp[iz][iy] = (cymh[iy] / cyph[iy]) * Bza[iz][iy][_PB_CXM] - (ch / cyph[iy]) * clf[iz][iy];
	      Hz[iz][iy][_PB_CXM]=(cxmh[_PB_CXM] / cxph[_PB_CXM]) * Hz[iz][iy][_PB_CXM]
		+ (mui * czp[iz] / cxph[_PB_CXM]) * tmp[iz][iy]
		- (mui * czm[iz] / cxph[_PB_CXM]) * Bza[iz][iy][_PB_CXM];
	      Bza[iz][iy][_PB_CXM] = tmp[iz][iy];
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  clf[iz][iy] = Ex[iz][_PB_CYM][ix] - Ax[iz][ix] + Ey[iz][_PB_CYM][ix+1] - Ey[iz][_PB_CYM][ix];
		  tmp[iz][iy] = (cymh[_PB_CYM] / cyph[iy]) * Bza[iz][iy][ix] - (ch / cyph[iy]) * clf[iz][iy];
		  Hz[iz][_PB_CYM][ix] = (cxmh[ix] / cxph[ix]) * Hz[iz][_PB_CYM][ix]
		    + (mui * czp[iz] / cxph[ix]) * tmp[iz][iy]
		    - (mui * czm[iz] / cxph[ix]) * Bza[iz][_PB_CYM][ix];
		  Bza[iz][_PB_CYM][ix] = tmp[iz][iy];
		}
	      clf[iz][iy] = Ex[iz][_PB_CYM][_PB_CXM] - Ax[iz][_PB_CXM] + Ry[iz][_PB_CYM] - Ey[iz][_PB_CYM][_PB_CXM];
	      tmp[iz][iy] = (cymh[_PB_CYM] / cyph[_PB_CYM]) * Bza[iz][_PB_CYM][_PB_CXM] - (ch / cyph[_PB_CYM]) * clf[iz][iy];
	      Hz[iz][_PB_CYM][_PB_CXM] = (cxmh[_PB_CXM] / cxph[_PB_CXM]) * Hz[iz][_PB_CYM][_PB_CXM]
		+ (mui * czp[iz] / cxph[_PB_CXM]) * tmp[iz][iy]
		- (mui * czm[iz] / cxph[_PB_CXM]) * Bza[iz][_PB_CYM][_PB_CXM];
	      Bza[iz][_PB_CYM][_PB_CXM] = tmp[iz][iy];
	    }
	}
  }
  #pragma endscop
}
#elif defined POLYBENCH_OFFLOAD1D
static
void kernel_fdtd_apml(int cz,
		      int cxm,
		      int cym,
		      DATA_TYPE mui,
		      DATA_TYPE ch,
		      DATA_TYPE POLYBENCH_2D_1D(Ax,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D_1D(Ry,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D_1D(clf,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_2D_1D(tmp,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D_1D(Bza,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D_1D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D_1D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D_1D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(czm,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(czp,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(cxmh,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cxph,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cymh,CYM+1,cym+1),
		      DATA_TYPE POLYBENCH_1D(cyph,CYM+1,cym+1))
{
  int iz, iy, ix;
#define Ax_IDX(i,j) IDX2(Ax,i,j,cz+1,cym+1)
#define Ry_IDX(i,j) IDX2(Ry,i,j,cz+1,cym+1)
#define clf_IDX(i,j) IDX2(clf,i,j,cym+1,cxm+1)
#define tmp_IDX(i,j) IDX2(tmp,i,j,cym+1,cxm+1)
#define Bza_IDX(i,j,k) IDX3(Bza,i,j,k,cz+1,cym+1,cxm+1)
#define Ex_IDX(i,j,k) IDX3(Ex,i,j,k,cz+1,cym+1,cxm+1)
#define Ey_IDX(i,j,k) IDX3(Ey,i,j,k,cz+1,cym+1,cxm+1)
#define Hz_IDX(i,j,k) IDX3(Hz,i,j,k,cz+1,cym+1,cxm+1)

    #pragma omp target data map(to: Ax[:(CZ+1)*(CYM+1)], Ry[:(CZ+1)*(CYM+1)], \
            clf[:(CYM+1)*(CXM+1)], tmp[:(CYM+1)*(CXM+1)], czm[:CZ+1], czp[:CZ+1], \
            cxmh[:CXM+1], cxph[:CXM+1], cymh[:CYM+1], cyph[:CYM+1]) \
            map(tofrom: Bza[:(CZ+1)*(CYM+1)*(CXM+1)], Ex[:(CZ+1)*(CYM+1)*(CXM+1)], Ey[:(CZ+1)*(CYM+1)*(CXM+1)], Hz[:(CZ+1)*(CYM+1)*(CXM+1)])
    {
      #pragma omp target teams distribute parallel for private(iy, ix)
      for (iz = 0; iz < _PB_CZ; iz++)
        {
	  for (iy = 0; iy < _PB_CYM; iy++)
	    {
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  GET_IDX2(clf,iz,iy) = GET_IDX3(Ex,iz,iy,ix) - GET_IDX3(Ex,iz,iy+1,ix) + GET_IDX3(Ey,iz,iy,ix+1) - GET_IDX3(Ey,iz,iy,ix);
		  GET_IDX2(tmp,iz,iy) = (cymh[iy] / cyph[iy]) * GET_IDX3(Bza,iz,iy,ix) - (ch / cyph[iy]) * GET_IDX2(clf,iz,iy);
		  GET_IDX3(Hz,iz,iy,ix) = (cxmh[ix] /cxph[ix]) * GET_IDX3(Hz,iz,iy,ix)
		    + (mui * czp[iz] / cxph[ix]) * GET_IDX2(tmp,iz,iy)
		    - (mui * czm[iz] / cxph[ix]) * GET_IDX3(Bza,iz,iy,ix);
		  GET_IDX3(Bza,iz,iy,ix) = GET_IDX2(tmp,iz,iy);
		}
	      GET_IDX2(clf,iz,iy) = GET_IDX3(Ex,iz,iy,_PB_CXM) - GET_IDX3(Ex,iz,iy+1,_PB_CXM) + GET_IDX2(Ry,iz,iy) - GET_IDX3(Ey,iz,iy,_PB_CXM);
	      GET_IDX2(tmp,iz,iy) = (cymh[iy] / cyph[iy]) * GET_IDX3(Bza,iz,iy,_PB_CXM) - (ch / cyph[iy]) * GET_IDX2(clf,iz,iy);
	      GET_IDX3(Hz,iz,iy,_PB_CXM)=(cxmh[_PB_CXM] / cxph[_PB_CXM]) * GET_IDX3(Hz,iz,iy,_PB_CXM)
		+ (mui * czp[iz] / cxph[_PB_CXM]) * GET_IDX2(tmp,iz,iy)
		- (mui * czm[iz] / cxph[_PB_CXM]) * GET_IDX3(Bza,iz,iy,_PB_CXM);
	      GET_IDX3(Bza,iz,iy,_PB_CXM) = GET_IDX2(tmp,iz,iy);
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  GET_IDX2(clf,iz,iy) = GET_IDX3(Ex,iz,_PB_CYM,ix) - GET_IDX2(Ax,iz,ix) + GET_IDX3(Ey,iz,_PB_CYM,ix+1) - GET_IDX3(Ey,iz,_PB_CYM,ix);
		  GET_IDX2(tmp,iz,iy) = (cymh[_PB_CYM] / cyph[iy]) * GET_IDX3(Bza,iz,iy,ix) - (ch / cyph[iy]) * GET_IDX2(clf,iz,iy);
		  GET_IDX3(Hz,iz,_PB_CYM,ix) = (cxmh[ix] / cxph[ix]) * GET_IDX3(Hz,iz,_PB_CYM,ix)
		    + (mui * czp[iz] / cxph[ix]) * GET_IDX2(tmp,iz,iy)
		    - (mui * czm[iz] / cxph[ix]) * GET_IDX3(Bza,iz,_PB_CYM,ix);
		  GET_IDX3(Bza,iz,_PB_CYM,ix) = GET_IDX2(tmp,iz,iy);
		}
	      GET_IDX2(clf,iz,iy) = GET_IDX3(Ex,iz,_PB_CYM,_PB_CXM) - GET_IDX2(Ax,iz,_PB_CXM) + GET_IDX2(Ry,iz,_PB_CYM) - GET_IDX3(Ey,iz,_PB_CYM,_PB_CXM);
	      GET_IDX2(tmp,iz,iy) = (cymh[_PB_CYM] / cyph[_PB_CYM]) * GET_IDX3(Bza,iz,_PB_CYM,_PB_CXM) - (ch / cyph[_PB_CYM]) * GET_IDX2(clf,iz,iy);
	      GET_IDX3(Hz,iz,_PB_CYM,_PB_CXM) = (cxmh[_PB_CXM] / cxph[_PB_CXM]) * GET_IDX3(Hz,iz,_PB_CYM,_PB_CXM)
		+ (mui * czp[iz] / cxph[_PB_CXM]) * GET_IDX2(tmp,iz,iy)
		- (mui * czm[iz] / cxph[_PB_CXM]) * GET_IDX3(Bza,iz,_PB_CYM,_PB_CXM);
	      GET_IDX3(Bza,iz,_PB_CYM,_PB_CXM) = GET_IDX2(tmp,iz,iy);
	    }
	}
  }
}
#else
static
void kernel_fdtd_apml(int cz,
		      int cxm,
		      int cym,
		      DATA_TYPE mui,
		      DATA_TYPE ch,
		      DATA_TYPE POLYBENCH_2D(Ax,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D(Ry,CZ+1,CYM+1,cz+1,cym+1),
		      DATA_TYPE POLYBENCH_2D(clf,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_2D(tmp,CYM+1,CXM+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Bza,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Ex,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Ey,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_3D(Hz,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(czm,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(czp,CZ+1,cz+1),
		      DATA_TYPE POLYBENCH_1D(cxmh,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cxph,CXM+1,cxm+1),
		      DATA_TYPE POLYBENCH_1D(cymh,CYM+1,cym+1),
		      DATA_TYPE POLYBENCH_1D(cyph,CYM+1,cym+1))
{
  int iz, iy, ix;

#ifdef OMP_DCAT
    #pragma omp target data map(to: Ax, Ry, clf, tmp, czm[:CZ+1], czp[:CZ+1], \
            cxmh[:CXM+1], cxph[:CXM+1], cymh[:CYM+1], cyph[:CYM+1]) \
            map(tofrom: Bza, Ex,Ey, Hz)
#else
    #pragma omp target data map(to: Ax[:CZ+1][:CYM+1], Ry[:CZ+1][:CYM+1], \
            clf[:CYM+1][:CXM+1], tmp[:CYM+1][:CXM+1], czm[:CZ+1], czp[:CZ+1], \
            cxmh[:CXM+1], cxph[:CXM+1], cymh[:CYM+1], cyph[:CYM+1]) \
            map(tofrom: Bza[:CZ+1][:CYM+1][:CXM+1], Ex[:CZ+1][:CYM+1][:CXM+1],\
            Ey[:CZ+1][:CYM+1][:CXM+1], Hz[:CZ+1][:CYM+1][:CXM+1])
#endif
    {
      #pragma omp target teams distribute parallel for private(iy, ix)
      for (iz = 0; iz < _PB_CZ; iz++)
        {
	  for (iy = 0; iy < _PB_CYM; iy++)
	    {
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  clf[iz][iy] = Ex[iz][iy][ix] - Ex[iz][iy+1][ix] + Ey[iz][iy][ix+1] - Ey[iz][iy][ix];
		  tmp[iz][iy] = (cymh[iy] / cyph[iy]) * Bza[iz][iy][ix] - (ch / cyph[iy]) * clf[iz][iy];
		  Hz[iz][iy][ix] = (cxmh[ix] /cxph[ix]) * Hz[iz][iy][ix]
		    + (mui * czp[iz] / cxph[ix]) * tmp[iz][iy]
		    - (mui * czm[iz] / cxph[ix]) * Bza[iz][iy][ix];
		  Bza[iz][iy][ix] = tmp[iz][iy];
		}
	      clf[iz][iy] = Ex[iz][iy][_PB_CXM] - Ex[iz][iy+1][_PB_CXM] + Ry[iz][iy] - Ey[iz][iy][_PB_CXM];
	      tmp[iz][iy] = (cymh[iy] / cyph[iy]) * Bza[iz][iy][_PB_CXM] - (ch / cyph[iy]) * clf[iz][iy];
	      Hz[iz][iy][_PB_CXM]=(cxmh[_PB_CXM] / cxph[_PB_CXM]) * Hz[iz][iy][_PB_CXM]
		+ (mui * czp[iz] / cxph[_PB_CXM]) * tmp[iz][iy]
		- (mui * czm[iz] / cxph[_PB_CXM]) * Bza[iz][iy][_PB_CXM];
	      Bza[iz][iy][_PB_CXM] = tmp[iz][iy];
	      for (ix = 0; ix < _PB_CXM; ix++)
		{
		  clf[iz][iy] = Ex[iz][_PB_CYM][ix] - Ax[iz][ix] + Ey[iz][_PB_CYM][ix+1] - Ey[iz][_PB_CYM][ix];
		  tmp[iz][iy] = (cymh[_PB_CYM] / cyph[iy]) * Bza[iz][iy][ix] - (ch / cyph[iy]) * clf[iz][iy];
		  Hz[iz][_PB_CYM][ix] = (cxmh[ix] / cxph[ix]) * Hz[iz][_PB_CYM][ix]
		    + (mui * czp[iz] / cxph[ix]) * tmp[iz][iy]
		    - (mui * czm[iz] / cxph[ix]) * Bza[iz][_PB_CYM][ix];
		  Bza[iz][_PB_CYM][ix] = tmp[iz][iy];
		}
	      clf[iz][iy] = Ex[iz][_PB_CYM][_PB_CXM] - Ax[iz][_PB_CXM] + Ry[iz][_PB_CYM] - Ey[iz][_PB_CYM][_PB_CXM];
	      tmp[iz][iy] = (cymh[_PB_CYM] / cyph[_PB_CYM]) * Bza[iz][_PB_CYM][_PB_CXM] - (ch / cyph[_PB_CYM]) * clf[iz][iy];
	      Hz[iz][_PB_CYM][_PB_CXM] = (cxmh[_PB_CXM] / cxph[_PB_CXM]) * Hz[iz][_PB_CYM][_PB_CXM]
		+ (mui * czp[iz] / cxph[_PB_CXM]) * tmp[iz][iy]
		- (mui * czm[iz] / cxph[_PB_CXM]) * Bza[iz][_PB_CYM][_PB_CXM];
	      Bza[iz][_PB_CYM][_PB_CXM] = tmp[iz][iy];
	    }
	}
  }
}
#endif


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int cz = CZ;
  int cym = CYM;
  int cxm = CXM;

  /* Variable declaration/allocation. */
  DATA_TYPE mui;
  DATA_TYPE ch;
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(Ax,DATA_TYPE,CZ+1,CYM+1,cz+1,cym+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(Ry,DATA_TYPE,CZ+1,CYM+1,cz+1,cym+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(clf,DATA_TYPE,CYM+1,CXM+1,cym+1,cxm+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,CYM+1,CXM+1,cym+1,cxm+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_3D_ARRAY_DECL(Bza,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_3D_ARRAY_DECL(Ex,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_3D_ARRAY_DECL(Ey,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  DC_END();
  DC_BEGIN();
  POLYBENCH_3D_ARRAY_DECL(Hz,DATA_TYPE,CZ+1,CYM+1,CXM+1,cz+1,cym+1,cxm+1);
  DC_END();
  POLYBENCH_1D_ARRAY_DECL(czm,DATA_TYPE,CZ+1,cz+1);
  POLYBENCH_1D_ARRAY_DECL(czp,DATA_TYPE,CZ+1,cz+1);
  POLYBENCH_1D_ARRAY_DECL(cxmh,DATA_TYPE,CXM+1,cxm+1);
  POLYBENCH_1D_ARRAY_DECL(cxph,DATA_TYPE,CXM+1,cxm+1);
  POLYBENCH_1D_ARRAY_DECL(cymh,DATA_TYPE,CYM+1,cym+1);
  POLYBENCH_1D_ARRAY_DECL(cyph,DATA_TYPE,CYM+1,cym+1);

  /* Initialize array(s). */
  init_array (cz, cxm, cym, &mui, &ch,
  	      POLYBENCH_ARRAY(Ax),
  	      POLYBENCH_ARRAY(Ry),
  	      POLYBENCH_ARRAY(Ex),
  	      POLYBENCH_ARRAY(Ey),
  	      POLYBENCH_ARRAY(Hz),
  	      POLYBENCH_ARRAY(czm),
  	      POLYBENCH_ARRAY(czp),
  	      POLYBENCH_ARRAY(cxmh),
  	      POLYBENCH_ARRAY(cxph),
  	      POLYBENCH_ARRAY(cymh),
  	      POLYBENCH_ARRAY(cyph));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_fdtd_apml (cz, cxm, cym, mui, ch,
  		    POLYBENCH_ARRAY(Ax),
  		    POLYBENCH_ARRAY(Ry),
  		    POLYBENCH_ARRAY(clf),
  		    POLYBENCH_ARRAY(tmp),
  		    POLYBENCH_ARRAY(Bza),
  		    POLYBENCH_ARRAY(Ex),
  		    POLYBENCH_ARRAY(Ey),
  		    POLYBENCH_ARRAY(Hz),
  		    POLYBENCH_ARRAY(czm),
  		    POLYBENCH_ARRAY(czp),
  		    POLYBENCH_ARRAY(cxmh),
  		    POLYBENCH_ARRAY(cxph),
  		    POLYBENCH_ARRAY(cymh),
  		    POLYBENCH_ARRAY(cyph));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(cz, cxm, cym,
  				    POLYBENCH_ARRAY(Bza),
  				    POLYBENCH_ARRAY(Ex),
  				    POLYBENCH_ARRAY(Ey),
  				    POLYBENCH_ARRAY(Hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(Ax);
  POLYBENCH_FREE_ARRAY(Ry);
  POLYBENCH_FREE_ARRAY(clf);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(Bza);
  POLYBENCH_FREE_ARRAY(Ex);
  POLYBENCH_FREE_ARRAY(Ey);
  POLYBENCH_FREE_ARRAY(Hz);
  POLYBENCH_FREE_ARRAY(czm);
  POLYBENCH_FREE_ARRAY(czp);
  POLYBENCH_FREE_ARRAY(cxmh);
  POLYBENCH_FREE_ARRAY(cxph);
  POLYBENCH_FREE_ARRAY(cymh);
  POLYBENCH_FREE_ARRAY(cyph);

  return 0;
}

