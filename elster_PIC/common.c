/**
 * This file contains helper functions and functions needed by both solvers
 * and simulation. Debug functions and plotting functions are also included 
 * in this file.
 */


#include "declarations.h"

/******************************************/
/****   mask KSR function with clock() ***/
double user_seconds()
{
  double t;

  t = MPI_Wtime();
  if(t== -1)
    fprintf(stderr, "WARNING: timings might be incorrect, clock() returns -1\n");
  return t;
}

/* array_min
 * Author: Anne C. Elster
 * */
double array_min( double *A, int n)

{
  int i;
  double result;

  result = A[0];
  for (i = 1; i<n; i++)
    {
      result = A[i] < result ? A[i]: result;
    }
  return result;
}


/* array_max
 * Author: Anne C. Elster
 * */
double array_max( double *A, int n)
{
  int i;
  double result;

  result = A[0];
  for (i = 1; i<n; i++)
    {
      result = A[i] > result ? A[i]: result;
    }
  return result;
}


/************************************
 * plot_field
 * a plot-function to plot the field 
 * charges at a given frame
 * Author: Nils Magnus Larsgård
 ************************************/

void plot_field(double *Rho, int Nx, int Ny, int frame){
  FILE *gnuplot;
  char filename[12+strlen(PLOT_DIR)];
  sprintf(filename, PLOT_DIR);
  int i,j;
  if(frame<10)
    sprintf(&filename[strlen(PLOT_DIR)], "p%d_field00%d", world_rank, frame);
  else if(frame<100)
    sprintf(&filename[strlen(PLOT_DIR)], "p%d_field0%d", world_rank, frame);
  else 
    sprintf(&filename[strlen(PLOT_DIR)], "p%d_field%d", world_rank, frame);
  gnuplot = fopen(filename, "w");
  for (i = 0; i < Ny; ++i) {
    for (j = 0; j < Nx; ++j) {
      fprintf(gnuplot, "%d %d %le\n", 
              i + Ny*cart_coords[0], j+Nx*cart_coords[1], 
              Rho[i*Nx +j]);			
    }
  }
  fflush(gnuplot);
  fclose(gnuplot);
  return;
}


/************************************
 * plot_particles 
 * a plot-function to plot the 
 * position of all the particles 
 * at a given frame
 ************************************/
void plot_particles(double * x, double * y, int Np, int frame, int Nx, int Ny, double hx, double hy){
  FILE *gnuplot;
  char filename[12+strlen(PLOT_DIR)];
  sprintf(filename, PLOT_DIR);
  int i,j;
  if(frame<10)
    sprintf(&filename[strlen(PLOT_DIR)], "p_%d_plot00%d", world_rank, frame);
  else if(frame<100)
    sprintf(&filename[strlen(PLOT_DIR)], "p_%d_plot0%d", world_rank, frame);
  else 
    sprintf(&filename[strlen(PLOT_DIR)], "p_%d_plot%d", world_rank, frame);
  gnuplot = fopen(filename, "w");
  for (i = 0; i < Np; ++i) {
    if(x[i] == PARTICLE_EMPTY)
      continue;
    fprintf(gnuplot, "%7.5lf, %7.5lf\n", x[i]+ cart_coords[1]*Nx*hx, y[i] + cart_coords[0]*Ny*hy);			
  }
  fprintf(gnuplot, "0, 0\n");
  fflush(gnuplot);
  fclose(gnuplot);
}

/*
 * Malloc wrapper for safe mallocs
 * Author: Nils Magnus Larsgård
 * */

void * xmalloc (size_t size) {
  void * p;
  p = malloc (size); 

  if (p == NULL){
    fprintf (stderr, "Not enough memory space, exiting(-1)\n");
    fflush(stderr);
    exit(13);
  }
#ifdef DEBUG
  if(world_rank==0)
    fprintf(stdout, "allocated %d bytes\n", size);
#endif
  return p;
}

/* mpexit
 * debug function
 * Author:Nils Magnus Larsgård
 * */
void mpexit(){
  fprintf(stderr, "%d exiting by debug mpexit()\n", world_rank);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  exit(0);
}

#ifdef PIC_PETSC
/* petsc_exit
 * debug function
 * Author:Nils Magnus Larsgård
 * */
void petsc_exit(){
  fprintf(stderr, "%d exiting by petsc_exit()\n", world_rank);
  PetscFinalize();
  exit(0);
}
#endif

/* checkerr - not in use
 * 
 * */
int checkerr(int err_code){
  if(err_code){


    switch(err_code){
      case MPI_SUCCESS:
        break;
      case MPI_ERR_COMM:
        fprintf(stderr, "MPI_ERR_COMM\n");
        break;
      case MPI_ERR_COUNT:
        fprintf(stderr, "MPI_ERR_COUNT\n");
        break;
      case MPI_ERR_TYPE:
        fprintf(stderr, "MPI_ERR_TYPE\n");
        break;
      case MPI_ERR_TAG:
        fprintf(stderr, "MPI_ERR_TAG\n");
        break;
      case MPI_ERR_RANK:
        fprintf(stderr, "MPI_ERR_RANK\n");
        break;
      case MPI_ERR_INTERN:
        fprintf(stderr, "MPI_ERR_INTERN\n");
        break;
      default:    
        fprintf(stderr, "Error code %d FOUND, do something useful\n", err_code);

    }
  }
  fflush(stderr);
  return err_code;
}
