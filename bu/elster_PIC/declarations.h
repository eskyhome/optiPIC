#ifndef DECLARATIONS_H_
#define DECLARATIONS_H_

#include <omp.h>

#include <mpi.h>
//#include <petscsnes.h> 

#include <stdio.h>	   /* include standard C I/O routines */
#include <math.h>          /* include the C math library   */
#include <sys/time.h>
#include <sys/procesor.h>
#include <stdlib.h>
#include <string.h>
/* #include <complex.h>  do NOT include this file */ 
#include <fftw3-mpi.h>
//#include <fftw3.h>
#include "my_macros.h"

/* solvers enum */
enum SOLVER { FFT, FFTW, SOR1, SOR1_MPI, NSQUARE, PETSC_SOR };


//#define CLOCKS_PER_SEC 1000000  /* for timing purposes  */

/* constants from original version*/
#define	pi     3.141592653589793
#define two_pi 6.28318530717959
#define omega_p 178000.0	/* theoretical plasma frequency */
#define nxpix 500
#define nypix 200
#define nsqpix 300 	/* this no. squared must be < nxpix*nypix */

/* plotdir: */
#define PLOT_DIR "plotdata/"

/* SOR variables */
#define UPPER_DIRICHLET_BORDER 0.00000001
#define LOWER_DIRICHLET_BORDER 0
#define SOR_MAX_ERROR 0.0000000000001
#define SOR_MAX_ITERATIONS 10000
#define SOR_OMEGA 1.78

/****** MPI variables and constants *******/
#define NDIMS 2
int world_rank, // rank in MPI_COMM_WORLD
  cart_rank,  // rank in cartesian system
  cart_coords[2], //
  p_over, p_under, p_left, p_right, // neighbor processes
  nump;       // number of processes in WORLD
int wraparound[NDIMS];
int dim_sizes[NDIMS];

MPI_Comm CART_COMM;
MPI_Datatype mat_col, mat_row; // mat_col and mat_row for sending SOR-values
MPI_Datatype mpi_particle; // for migrating particles

#define PARTICLE_EMPTY -1337


/**************** PETSC stuff ***********/
/*Mat petsc_rho;
  Vec foo_solution, f;
  //PC  petsc_pc_sor;
  //KSP petsc_solver;
  //ISLocalToGlobalMapping mapping;
  SNES petsc_solver;
  int form_function(SNES snes, Vec x, Vec f, void *dummy);
  int form_jacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dummy);
*/


// temporary array for use in generic solvers
double *tempA;

/* simulation functions */
double user_seconds();

int ConstInit(int *Nx, int *Ny, int *Ng, int *Np, int *Px, int *Py,
              double *eps, double *mass, double *q, double *rho0, double *rho_k,
              double *drag, double *Lx, double *Ly, double *hx, double *hy,
              double *t, double *del_t, double *t_max);
/* ConstInit gloabally allocated the following arrays: 
   double Phi[], double Rho[], double Ex[], double Ey[],
   double Part_x[], double Part_y[], double Vx[], double Vy[]);
*/

int   DataInit(int Ng, int Np, double *vdrift, double Lx, double Ly,
               double t,
               double Phi[], double Part_x[], double Part_y[], double Vx[],
               double Vy[]);

int  Uniform_Grid_Init(int Np, double Part_x[], double Part_y[],
                       double Lx, double Ly, double t);

int  PART_RHO( int Np, int n, int m, double rho_k,
               double hx, double hy, double t,
               double Part_x[], double Part_y[], double Rho[]);

int  four1(double data[],int nn, int isign);
int  FFT_SOLVE(double Phi[], double Rho[], double eps, double Lx, double Ly,\
               int Nx, int Ny);

int  PERIODIC_FIELD_GRID(double Phi[], int Nx, int Ny, double hx, double hy,\
                         double Ex[], double Ey[]);
int  PUSH_V(double Ex[], double Ey[],\
            double Np, \
            double Part_x[], double Part_y[],\
            int Nx, int Ny,\
            double hx, double hy,\
            double drag, double q, double mass,\
            double del_t,\
            double Vx[], double Vy[]);
int  PUSH_LOC(int Np, double Part_x[], double Part_y[],\
              double Vx[], double Vy[],\
              double Lx, double Ly, double hx, double hy, double del_t, double t);
int  Trace_set_up(int Nx, int Ny);
int  Trace1(int Nx, int Ny, int Ng, int Np, int Px, int Py,
            double eps, double mass, double q, double rho0, double rho_k,
            double Lx, double Ly, double hx, double hy,
            double t, double del_t, double t_max,
            double Phi[], double Rho[], double Ex[], double Ey[],
            double Part_x[], double Part_y[], double Vx[], double Vy[]);

int  Trace_Oscil(int i, double loc_x1, double loc_y1,
                 double t, int Nx, int Ny, double Phi[]);
int  Trace_close();

double array_min( double *A, int n);
double array_max( double *A, int n);

/* SOLVER functions*/
int sor_mpi_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
                  int Nx, int Ny);
int petsc_sor_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
                    int Nx, int Ny);
int sor_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
              int Nx, int Ny);
int nsquare(double Phi[], double Rho[], double eps, double Lx, double Ly,\
               int Nx, int Ny);
int fftw_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
               int Nx, int Ny);
int FFT_SOLVE(double Phi[], double Rho[], double eps, double Lx, double Ly,\
              int Nx, int Ny);

/* helper functions, plotting*/
void plot_particles(double * x, double * y, int Np, int frame, int Nx, int Ny, double hx, double hy);
void plot_field(double *Rho, int Nx, int Ny, int frame);

/*init functions*/
int petsc_init(int Nx, int Ny, double *rho);
int mpi_pic_init();
int create_mpi_types(int Nx, int Ny);


/* debug and malloc wrapper */
void mpexit();
void petsc_exit();
void * xmalloc (size_t size);
int checkerr(int err_code);

#endif /*DECLARATIONS_H_*/
