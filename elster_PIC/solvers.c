#include "declarations.h"

int generic_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
                  int Nx, int Ny)
{
  int result, method = FFTW;

#ifdef PIC_SOR
  method=SOR1_MPI;
  method = NSQUARE;
#endif

  /*#ifdef PIC_PETSC
    method = PETSC_SOR;
    #endif*/

  fprintf(stderr, "[%d] ready to solve\n", cart_rank);
  fflush(stderr);

  switch(method){
    case FFT: // the original solver made by Anne C. Elster
      result = FFT_SOLVE( Phi, Rho, eps, Lx, Ly, Nx, Ny );
      break;
    case FFTW: // using the fftw-library with mpi and threads
      result = fftw_solve( Phi, Rho, eps, Lx, Ly, Nx, Ny );
      break;
    case SOR1: // simple sor-solver with openmp
      result = sor_solve( Phi, Rho, eps, Lx, Ly, Nx, Ny );
      break;
    case SOR1_MPI: // sor-solver with hybrid mpi/openmp
      result = sor_mpi_solve( Phi, Rho, eps, Lx, Ly, Nx, Ny );
      break;
    case NSQUARE: 
      result = nsquare( Phi, Rho, eps, Lx, Ly, Nx, Ny );
      break;
      /*		case PETSC_SOR:
			result = petsc_sor_solve( Phi, Rho, eps, Lx, Ly, Nx, Ny );
			break;*/
    default:
      fprintf(stdout, "Error, invalid solver given in solvers.c!\n\texiting now...\n");
      exit(-1);
  }

  fprintf(stderr, "[%d] solved\n", cart_rank);
  fflush(stderr);

  return result;
}

int nsquare(double Phi[], double Rho[], double eps, double Lx, double Ly,\
            int Nx, int Ny){
  int Np=Nx*Ny;
  int i, j, k;
  int index;
  double min_err, err, old;
#pragma omp parallel
 { 
#pragma omp for private(i,j, index)
   for(i=Nx+1;i<(Np-Nx-1);i++){
     min_err = 10;
     for(j=0;j<Np;j++){
       index = i;
       old = Rho[index];
       Rho[index]= (Rho[i-1]+ Rho[i+1] + Rho[i-Nx] + Rho[i+Nx])/4;
       err = fabs(Rho[index] -old);
       min_err =(err<min_err)? err: min_err;
     }
   }
#pragma omp for private(i)
   for(i=0; i<Np;i++){
     Rho[i]=0;
     Phi[i] = Rho[i];
   }
 }

  return 0;
}

#ifdef PIC_PETSC
/* petsc_sor_solve
 * 
 * 1. transfer rho to global matrix
 * 2. solve matrix
 * 3. transfer result to local matrix
 * 
 * */
int petsc_sor_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
                    int Nx, int Ny){
  int i,j,err=0;
  int NY, NX;
  MatGetSize(petsc_rho, &NY, &NX);
  int  idx[Nx], idy[Ny];
  for (i = 0; i < Nx; ++i) {
    idx[i] = i+ cart_coords[1]*Nx;
  }
  for (i = 0; i < Ny; ++i) {
    idy[i] = i + cart_coords[0]*Ny;
  }
  /*DEBUG*/
  /*for (j = 0; j < Ny; ++j) {
    for (i = 0; i < Nx; ++i) {
    Rho[i+j*Nx] = i+j*Nx;
    }
    }
  */
  //	1. transfer rho to global matrix	
  MatSetValues(petsc_rho, Ny, idy, Nx, idx, Rho, INSERT_VALUES);
  MatAssemblyBegin(petsc_rho, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(petsc_rho, MAT_FINAL_ASSEMBLY);

  // 2. solve matrix
  SNESSolve(petsc_solver, foo_solution, f);
  /*KSPSetOperators(petsc_solver, petsc_rho, petsc_rho, DIFFERENT_NONZERO_PATTERN);
    KSPSetUp(petsc_solver);
    KSPSolve(petsc_solver, foo_solution, foo_solution);*/
  //MatRelax(petsc_rho, foo_solution, SOR_OMEGA, SOR_FORWARD_SWEEP, 0, 1, 30, foo_solution);
	
  // 3. transfer result back to Rho
  //MatGetValues(petsc_rho, Ny, idy, Nx, idx, Rho);
  //VecGetValues(foo_solution, Nx*Ny, b, Rho);

  Mat submatrix;
  MatCreateSeqAIJ(PETSC_COMM_SELF, Ny, Nx, 0, PETSC_NULL, &submatrix);
  MatSetFromOptions(submatrix);
  IS is, isn;
  ISCreateBlock(PETSC_COMM_WORLD, Nx, Ny, idx, &is);
  ISSort(is);
  ISAllGather(is,&isn);	
  MatGetSubMatrix(petsc_rho, is, isn, PETSC_DECIDE, MAT_INITIAL_MATRIX, &submatrix);
	
  for (i = 0; i < Nx; ++i) 
    idx[i] = i;
	
  for (i = 0; i < Ny; ++i) 
    idy[i] = i;
	
  MatGetValues_SeqAIJ(submatrix, Ny, idy, Nx, idx, Rho);
	
  MPI_Barrier(PETSC_COMM_WORLD);
  //splot 'p0_field1337', 'p1_field1337'
  plot_field(Rho, Nx, Ny, 1337);
  petsc_exit();
  return 0;
}

// for use with the petsc snes solver
int form_function(SNES snes, Vec x, Vec f, void *dummy){
  //	PetsScalar *xx, *ff;
  //	VecGetArray(x, &xx);
  //	VecGetArray(f, &ff);
  return 0;
}
// for use with the petsc snes solver
int form_jacobian(SNES snes, Vec x, Mat *jac, Mat *B, MatStructure *flag, void *dummy){
  PetscScalar *xx;
  int ierr;
	
  VecGetArray(x, &xx);
	
  return 0;
}
#endif /*ifdef PIC_PETSC*/

int sor_mpi_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
                  int Nx, int Ny){
  double err=1, err_temp , err_largest = 1;
  double temp, old, old_upper, old_lower, old_left, old_right ;
  double hx, hy, h2;
  int Ng = (Nx +2)* (Ny+2), i , j, iter;
	
  int numreqs=0, maxreqs=8; // numreqs is a counter of sends(and requests), maxreqs is limit for requests
  MPI_Request reqs[maxreqs];
  MPI_Status  stats[maxreqs];
  int tag=0;

  for (i = 0; i < maxreqs; ++i) 
    reqs[i] =0; // dunno why, but it works

	
  if(tempA == NULL)
    tempA = (double * )xmalloc( (Nx+2)*(Ny+2) * sizeof(double));

  /* copy rho into tempA */

  for (i = 0; i < Ny; ++i) {
    for (j = 0; j < Nx; ++j) {
      tempA[(i+1) * (Nx+2) + j +1] = Rho[i*Nx +j];			
    }
  }
  for (i = 0; i < Nx+2; ++i) {
    tempA[i] =0;
    tempA[(Ny+1)*(Nx+2)+i] =0;
  }
	
  hy = Ly / Ny;
  hx = Lx / Nx;
  h2 = hx*hx + hy*hy;
  iter=0;
  fflush(stdout);
  int k;
  for(k=0; k< 128; k++){ // like meyer: fixed number of iterations to ease benchmarking
  //  while( err > SOR_MAX_ERROR && iter < SOR_MAX_ITERATIONS || iter < SOR_MAX_ITERATIONS/10){
    iter++;
#ifdef DEBUG
    fprintf(stderr, "[%d] starting iteration %d\n", cart_rank, iter);
    fflush(stderr);
#endif
    err_largest = 0;
    int row_index_start, row_index, index, ierr;
    for (row_index_start = 1; row_index_start < 3; row_index_start++) 
      {
        row_index = row_index_start;  
#pragma omp parallel private(i,j, index) //schedule (static, 1024)         
 {

   /*   int tid = omp_get_thread_num();
        bindprocessor(BINDTHREAD, tid , tid);*/
#pragma omp for 
        for (i =(cart_coords[0]==0)? 2 : 1 ; i < Ny+1; ++i) {
          for (j = row_index ; j < Nx+1; j+=2) {
            index = i*(Nx+2) +j;
            old = tempA[index];
					
            old_left  = tempA[ index -1 ]; 
            old_right = tempA[ index +1 ];
            old_upper = tempA[ index - (Nx+2) ];
            old_lower = tempA[ index + (Nx+2) ];

            temp = ( old_left + old_right + old_upper + old_lower)/4;
            temp = old + SOR_OMEGA*(temp - old);
            err_temp = fabs(old - temp);

            tempA[i*(Nx+2) + j] = temp;
				
            //if (err_temp> err_largest){
            // err_largest=  err_temp;
            //fprintf(stderr, "found me a new error :)\n");
            //}
          }	
          row_index = (row_index==2) ? 1 : 2 ;		
        }  // partial sor-iteration over here
        //#pragma omp flush(tempA)
        //fprintf(stderr, "[%d] did a red/black iteration(%d)\n", cart_rank, iter);
        //fflush(stdout);
        // mpi send-recv here
 }      
#ifdef DEBUG
        fprintf(stderr, "[%d] doing communication in iteration %d\n", cart_rank, iter);
        fflush(stderr);
#endif


        numreqs =-1;
        MPI_Barrier(CART_COMM);
        int htag1=1, htag2=2, vtag1=3, vtag2=4;
        //if(0) // horizontal borders
        { 
          if(cart_coords[0]!= (dim_sizes[0]-1)) // send/recv bottom borders if process not is bottom border itself
            {
#ifdef DEBUG
              fprintf(stderr,"[%d -> sendto %d] and [%d -> recvfrom %d]\n", cart_rank, p_under, cart_rank, p_under);
              fflush(stderr);
#endif
              index = (Ny)*(Nx+2)+row_index_start;
              ierr = MPI_Isend(&(tempA[index]), 1, mat_row, p_under, htag1, CART_COMM, &reqs[++numreqs]);
              checkerr(ierr);

              index = row_index_start + (Nx+2)*(Ny+1);
              ierr = MPI_Irecv(&(tempA[index]), 1, mat_row, p_under, htag2, CART_COMM, &reqs[++numreqs] );
              checkerr(ierr);
            }
          //		mpexit();
          if( cart_coords[0]!=0) // send/recv top borders if process not is bottom border itself 
            {
#ifdef DEBUG
              fprintf(stderr,"[%d -> sendto %d] and [%d -> recvfrom %d]\n", cart_rank, p_over, cart_rank, p_over);
              fflush(stderr);
#endif
              index = (Nx+2+row_index_start+1);
              ierr = MPI_Isend(&(tempA[ index ]), 1, mat_row, p_over, htag2, CART_COMM, &reqs[++numreqs]);
              checkerr(ierr);
              

              index = row_index_start;
              ierr = MPI_Irecv(&(tempA[ index ]), 1, mat_row, p_over, htag1, CART_COMM, &reqs[++numreqs]);
              checkerr(ierr);
            }
        }

        //if(0)// vertical borders
        {
#ifdef DEBUG
          fprintf(stderr,"[%d -> sendto %d] and [%d -> recvfrom %d]\n", cart_rank, p_under, cart_rank, p_right);
          fflush(stderr);
#endif
          // send/recv right borders 
          index = (Nx+2)*(( row_index_start + 1) ) - (2);
          MPI_Isend( &(tempA[ index ]) , 1, mat_col, p_right, vtag2, CART_COMM, &(reqs[++numreqs]));
          //printf("%d, %d sends right col from %d to right(%d)\n", cart_rank, iter, index, p_right);
          
          index = (Nx+2)*(( 4 - row_index_start  ) ) -1 ;
          MPI_Irecv( &(tempA[ index ]), 1, mat_col, p_right, vtag1, CART_COMM, &(reqs[++numreqs]));	 
          //printf("%d, %d receives right col to  %d\n", cart_rank, iter, index);
      
#ifdef DEBUG
          fprintf(stderr,"[%d -> sendto %d] and [%d -> recvfrom %d]\n", cart_rank, p_under, cart_rank, p_left);
          fflush(stderr);
#endif
          // send/recv left borders 
          index = (Nx+2)*(3 - row_index_start) + 1;
          MPI_Isend( &(tempA[index ]) , 1, mat_col, p_left, vtag1, CART_COMM, &(reqs[++numreqs]));
          //printf("%d sends left col from  %d\n", cart_rank, index);
          
          index = (Nx+2)*(row_index_start);
          MPI_Irecv( &(tempA[ index]), 1, mat_col, p_left, vtag2, CART_COMM, &(reqs[++numreqs]));	 
          //printf("%d, %d iter  receives left col to  %d from left(%d)\n", cart_rank, iter, index, p_left);
          }	
	/* wait for sends to complete*/
        if( numreqs > 0){
          numreqs++;
          MPI_Status statss[numreqs];
#ifdef DEBUG
          fprintf(stderr, "[%d] waiting for %d requests\n", cart_rank, numreqs);
#endif
          MPI_Waitall( numreqs , reqs, statss);
#ifdef DEBUG
          fprintf(stderr, "[%d] done waiting\n", cart_rank, numreqs);
#endif

        }
        //fflush(stdout);
        //if(iter==2)	mpexit();
        
      } // for row_index_start, red-black SOR is finished
    //fprintf(stderr, "%d: iter %d\n", world_rank, iter);
		
    
    err = err_largest;
    double tmperr = err;
    MPI_Allreduce(&tmperr, &err, 1, MPI_DOUBLE, MPI_MAX, CART_COMM );
  }
  /* copy tempA into phi */
	
  for (i = 0; i < Ny; ++i) {
    for (j = 0; j < Nx; ++j) {
      Phi[i * Nx + j] = tempA[(i+1)*(Nx+2) +j+1];			
    }
  }
  return 0;
}


/*************** SOR with OpenMP ********************/
int sor_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
              int Nx, int Ny){
  double err=1, err_temp , err_largest = 1;
  double temp, old, old_upper, old_lower, old_left, old_right ;
  double hx, hy, h2;
  int Ng = Nx * Ny, i , j, iter;
	
  if(tempA == NULL)
    if( (tempA = (double * )malloc( Ng * sizeof(double)) ) == NULL ){
      fprintf(stderr, " ERROR: Could not allocate enough memory, quitting.\n");
      exit(-1);
    }
	
  /* copy rho into tempA */
  for (i = 0; i < Ng; ++i) {
    tempA[i] = Rho[i];
  }
  hy = Ly / Ny;
  hx = Lx / Nx;
  h2 = hx*hx + hy*hy;
  iter=0;
  int k;
  for(k=0; k< 128; k++){// set to 128 iterations like in meyer's solver
    //  while( err > SOR_MAX_ERROR && iter < SOR_MAX_ITERATIONS ){
    iter++;
    err_largest = 0;
    int row_index_start=0;
    for (row_index_start = 1; row_index_start < 3; row_index_start++) 
      {
        //    		fprintf(stderr, "\nfirst start index: %d\n", row_index_start);
#pragma omp parallel for private(i,j, row_index_start) //schedule(static, 100) 
        for (i = 1; i < Ny-1; ++i) {
          row_index_start = (row_index_start==2) ? 1 : 2 ;
          //	    		fprintf(stderr, "start index: %d\t", row_index_start);
          for (j = row_index_start ; j < Nx; j+=2) {
				
            old = tempA[i*Nx +j];
					
            old_left  = (j != 0  )? tempA[ i * Nx + j - 1] : old; // tempA[ i * (Nx+1) ];
            old_right = (j != Nx )? tempA[ i * Nx + j + 1]: old ;//tempA[ i * Nx];
            old_upper = tempA[ (i-1) * Nx + j];
            old_lower = tempA[ (i+1) * Nx + j];
				
            temp = (/*old*h2/eps*/ + old_left + old_right + old_upper + old_lower)/4;
				
            temp = old + SOR_OMEGA*(temp - old);
				
            err_temp = fabs(old - temp);

            /*update matrix*/
            tempA[i*Nx + j] = temp;
				
            if (err_temp> err_largest){
              err_largest=  err_temp;
              //fprintf(stderr, "found me a new error :)\n");
            }
          }
        }
#pragma omp flush(tempA)
      } // for row_index_start
    err = err_largest;
  }
  /* copy tempA into phi */
  for (i = 0; i < Ng; ++i) {
    Phi[i] = tempA[i];
  }
  return 0;
}


fftw_plan dft_plan=NULL, rdft_plan=NULL; /* the fftw-plans, DFT and REVERSE */
fftw_complex *c_tempA=NULL; /* we use the same datastructure for input and result  */

int fftw_solve(double Phi[], double Rho[], double eps, double Lx, double Ly,\
               int Nx, int Ny){
#ifdef DEBUG
  fprintf(stderr, "[%d] using mpi_fftw with %d threads\n", cart_rank, 4 );
#endif

  int	Ng,
    index,
    isign,			/* FFT (1) or inv FFT (-1) */
    n,			/* number of "columns" in complex temp array */
    lda,  			/* leading dim. of real array */
    clda,  			/* leading dim. of complex array */
    i, j;			/* loop counters */

  double
    kx, ky, kxt, kyt,
    k2,			/* tem variable for k^2 = kx^2 + ky^2*/
    *Aptr,  *tempA;

  ptrdiff_t n0=Nx*nump, n1=Ny, alloc_local, local_n0=NULL, local_0_start=NULL; // fftw variables
  
  /***************** begin FFT_SOVLE *****************************/

  Ng = Nx * Ny;
  lda = Nx; /* assuming  row-storage */
  clda = 2*Nx; /* assuming  row-storage */
  n = 2*Nx;

  /* Allocate temporary array for storing complex column before calling FFT */
  /* The following mallocs could be moved to Data_In for efficiency
     --- i.e. malloced once for the program and then freed */

  isign = FFTW_FORWARD; /* regular FFT */

  tempA = (double*)xmalloc(sizeof (double)*2*Ng);

  /* init fftw_mpi and memory  */
  if(dft_plan == NULL || c_tempA == NULL){
  

    
    fftw_mpi_init();
    fftw_init_threads();
    
    fftw_plan_with_nthreads( omp_get_num_threads() );

    
    alloc_local = fftw_mpi_local_size_2d(n0, n1, CART_COMM, &local_n0,  &local_0_start);
    c_tempA = fftw_malloc(sizeof(fftw_complex) * alloc_local);

    //c_tempA = fftw_malloc(sizeof(fftw_complex)*Ng); // single cpu malloc
    fprintf(stderr, "[%d] mpi-allocated, local_n0:%d, local_0_start:%d\n",
            cart_rank, local_n0, local_0_start);
    fflush(stderr);
    

    /* mpi parallel plan */
    dft_plan = fftw_mpi_plan_dft_2d(n0, n1, c_tempA, c_tempA, 
                                    CART_COMM, isign, FFTW_MEASURE);

    if(cart_rank==0){
      FILE *wise;
      wise = fopen("wisdom", "w");
      fftw_export_wisdom_to_file(wise);
      fflush(wise);
    }
    //dft_plan = fftw_plan_dft_2d(Nx, Ny, c_tempA, c_tempA, isign, FFTW_MEASURE); // single cpu version
  }

  /* printf(" copy  matrix into complex array: \n\n"); */
  for (i=0; i < Ng; i++){
    c_tempA[i][0] = Rho[i];
    c_tempA[i][1] = 0.0;
  }

  /* printf(" copy  matrix into complex array: \n\n"); */
  for (i=0; i < Ng; i++)
    {
      tempA[2*i] = Rho[i];
      tempA[2*i +1] = 0.0;
    }

  /* actually execute the plan */
  fftw_execute_dft(dft_plan, c_tempA, c_tempA);
#ifdef DEBUG
  fprintf(stderr, "[%d] executed the plan\n", cart_rank);
  fflush(stderr);
#endif  
  for (i = 0; i < Ng; ++i) {
    tempA[2*i] = c_tempA[i][0];
    tempA[2*i+1] = c_tempA[i][1];
  }


  /* Now have FFT(Rho), i.e. Rho(kx, ky) -- then divide by k^2 = kx^2 + ky^2
     and scale by 1/eps to get Phi(kx, ky):
  */

  /* kx = 2 * pi * i / Lx,  ky = 2 * pi * j/ Ly */

  kxt = 2 * pi / Lx;
  kyt = 2 * pi / Ly;

#pragma omp parallel private(j,i,kx,ky,k2,index)
 {
#pragma omp for nowait
  for (j=0; j<Nx; j=j+2)
    {
      for (i = 0; i < Ny/2; i++)
        {
          if ((i==0) && (j==0))
            {
              Phi[0] = 0.0;

            }
          else
            {
              kx = kxt * (j/2);
              ky = kyt * (i);
              k2 = (kx*kx) + (ky*ky);
              index = (i*clda) + j;

              /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
              tempA[index] /= (k2 * eps);

              /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
              tempA[index+1] /= (k2 * eps);
            }	
        }
    }
#pragma omp for nowait
  for (j=0; j<Nx; j=j+2)
    {
      for (i = Ny/2; i < Ny; i++)
        {
          kx = kxt * (j/2);
          ky = kyt * (Ny - i);
          k2 = (kx*kx) + (ky*ky);
          index = (i*clda) + j;

          /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
          tempA[index] /= (k2 * eps);

          /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
          tempA[index+1] /= (k2 * eps);
        }
    }
#pragma omp for nowait
  for (j=Nx; j<2*Nx; j=j+2)
    {
      for (i = 0; i < Ny/2; i++)
        {
          kx = kxt * (Nx - j/2);
          ky = kyt * i;
          k2 = (kx*kx) + (ky*ky);
          index = (i*clda) + j;

          /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
          tempA[index] /= (k2 * eps);

          /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
          tempA[index+1] /= (k2 * eps);
        }
    }
#pragma omp for nowait
  for (j=Nx; j<2*Nx; j=j+2)
    {
      for (i = Ny/2; i < Ny; i++)
        {
          kx = kxt * (Nx - j/2);
          ky = kyt * (Ny - i);
          k2 = (kx*kx) + (ky*ky);
          index = (i*clda) + j;

          /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
          tempA[index] /= (k2 * eps);

          /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
          tempA[index+1] /= (k2 * eps);
        }
    }

 }
  /* call complex fft function doing inv. fft col-wise: */

  isign = FFTW_BACKWARD;
  for (i = 0; i < Ng; ++i) {
    c_tempA[i][0] = tempA[2*i];
    c_tempA[i][1] = tempA[2*i+1];
  }
  
  if(rdft_plan == NULL){
    rdft_plan = fftw_mpi_plan_dft_2d(n0, n1, c_tempA, c_tempA, CART_COMM, isign, FFTW_MEASURE); // mpi version
    //rdft_plan = fftw_plan_dft_2d(Nx, Ny, c_tempA, c_tempA, isign, FFTW_MEASURE); // single cpu version
  }
#ifdef DEBUG
  fprintf(stderr, "[%d] executing backwards\n", cart_rank);
  fflush(stderr);
#endif
  fftw_execute_dft(rdft_plan, c_tempA, c_tempA);

  for (i=0; i < Ng; i++)
    {
      Phi[i] = c_tempA[i][0];
    }

  free(tempA);


  /* printf("Final result (incl. mult. by 1/Nx * Ny)  = %i: \n \n",); */

  for (i=0; i<Ng; i++)
    {	
      Phi[i] = Phi[i]/(Ng);
    }
  return 0;
}



/************************************************************************/
/************************************************************************/
/****************     FFT_SOLVE  *****************************************/
/************************************************************************/

/*
 *      Author: Anne C. Elster                          July-Aug  1992
 *
 *      This fuction is a complex 2-D FFT implementation
 *      for periodic boundaries.
 *      The initial data are passed in as pointers.
 * 	The routine uses globally defined pi           */



/* WARNING:	This routine is VERY crude -- extra storage and
   extra bit-reversals are computed.
   It was written for the purpose fo testing only.
*/


int FFT_SOLVE(double Phi[], double Rho[], double eps, double Lx, double Ly,\
              int Nx, int Ny)

{


  int	Ng,
    index,
    isign,			/* FFT (1) or inv FFT (-1) */
    n,			/* number of "columns" in complex temp array */
    lda,  			/* leading dim. of real array */
    clda,  			/* leading dim. of complex array */
    i, j;			/* loop counters */

  double
    kx, ky, kxt, kyt,
    k2,			/* tem variable for k^2 = kx^2 + ky^2*/
    *Aptr, *tempcol, *tempA;

  /***************** begin FFT_SOVLE *****************************/

  Ng = Nx * Ny;
  lda = Nx; /* assuming  row-storage */
  clda = 2*Nx; /* assuming  row-storage */
  n = 2*Nx;

  /* Allocate temporary array for storing complex column before calling FFT */
  /* The following mallocs could be moved to Data_In for efficiency
     --- i.e. malloced once for the program and then freed */

  if((tempcol = (double*)malloc(sizeof (double)*2*Ny))==NULL)
    printf("FFT_SOLVE: temp allocation for column failed --- out of memory");

  if((tempA = (double*)malloc(sizeof (double)*2*Ng))==NULL)
    printf("FFT_SOLVE: temp alloc. for array failed --- out of memory");



  /* printf(" copy  matrix into complex array: \n\n"); */
  for (i=0; i < Ng; i++)
    {
      tempA[2*i] = Rho[i];
      tempA[2*i +1] = 0.0;
    }


  isign = 1; /* regular FFT */
  n = 2*Nx;

  /* call complex  fft fuction row-wise : */

  Aptr = &tempA[0] - 1;
  for (i=0; i<Ny; i++)
    {
      four1(Aptr,Nx,isign);
      Aptr += n;
    }


  /* call complex ftt fuction column-wise : */

  /* printf("use temp storage of column to avoid stride ...\n\n"); */
	
  clda = 2*Nx;
  Aptr = &tempcol[0] - 1;
  for (j=0; j<clda; j=j+2)
    {
      for (i = 0; i < Ny; i++)
        {
          index = (i*clda) + j;
          tempcol[2*i] = tempA[index];	    /* tempA(i,j) */
          tempcol[(2*i)+1] = tempA[index+1];  /* tempA(i,j+1) */
        }
      four1(Aptr,Ny,isign);
      for (i = 0; i < Ny; i++)
        {
          index = (i*clda) + j;
          tempA[index] = tempcol[2*i];
          tempA[index+1] = tempcol[(2*i)+1];
        }
    }


  /* Now have FFT(Rho), i.e. Rho(kx, ky) -- then divide by k^2 = kx^2 + ky^2
     and scale by 1/eps to get Phi(kx, ky):
  */

  /* kx = 2 * pi * i / Lx,  ky = 2 * pi * j/ Ly */

  kxt = 2 * pi / Lx;
  kyt = 2 * pi / Ly;


  for (j=0; j<Nx; j=j+2)
    {
      for (i = 0; i < Ny/2; i++)
        {
          if ((i==0) && (j==0))
            {
              Phi[0] = 0.0;

            }
          else
            {
              kx = kxt * (j/2);
              ky = kyt * (i);
              k2 = (kx*kx) + (ky*ky);
              index = (i*clda) + j;

              /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
              tempA[index] /= (k2 * eps);

              /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
              tempA[index+1] /= (k2 * eps);
            }	
        }
    }

  for (j=0; j<Nx; j=j+2)
    {
      for (i = Ny/2; i < Ny; i++)
        {
          kx = kxt * (j/2);
          ky = kyt * (Ny - i);
          k2 = (kx*kx) + (ky*ky);
          index = (i*clda) + j;

          /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
          tempA[index] /= (k2 * eps);

          /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
          tempA[index+1] /= (k2 * eps);
        }
    }

  for (j=Nx; j<2*Nx; j=j+2)
    {
      for (i = 0; i < Ny/2; i++)
        {
          kx = kxt * (Nx - j/2);
          ky = kyt * i;
          k2 = (kx*kx) + (ky*ky);
          index = (i*clda) + j;

          /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
          tempA[index] /= (k2 * eps);

          /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
          tempA[index+1] /= (k2 * eps);
        }
    }

  for (j=Nx; j<2*Nx; j=j+2)
    {
      for (i = Ny/2; i < Ny; i++)
        {
          kx = kxt * (Nx - j/2);
          ky = kyt * (Ny - i);
          k2 = (kx*kx) + (ky*ky);
          index = (i*clda) + j;

          /* ReA(i,j) = ReA(i,j)/(k^2) * 1/eps */
          tempA[index] /= (k2 * eps);

          /* ImA(i,j) = ImA(i,j)/(k^2) * 1/eps */
          tempA[index+1] /= (k2 * eps);
        }
    }

  /* call complex fft function doing inv. fft col-wise: */

  isign = -1;
  Aptr = &tempcol[0] - 1;

  for (j=0; j<n; j=j+2)
    {
      for (i = 0; i < Ny; i++)
        {
          index = (i*clda) + j;
          tempcol[2*i] = tempA[index];	   /* tempA(i,j) */
          tempcol[2*i+1] = tempA[index+1];   /* tempA(i,j+1) */
        }
      four1(Aptr,Ny,isign);
      for (i = 0; i < Ny; i++)
        {
          index = (i*clda) + j;
          tempA[index] = tempcol[2*i];   /* tempA(i,j) */
          tempA[index+1] = tempcol[2*i+1]; /* tempA(i,j+1) */
        }
    }

  free(tempcol);  /* release temporary storage */


  /* call complex inv. fft fuction row-wise : */

  Aptr = &tempA[0] - 1;
  for (i=0; i<Ny; i++)
    {
      four1(Aptr,Nx,isign);
      Aptr += n;
    }



  /* printf("transfering result back to real array ....\n \n "); */

  for (i=0; i < Ng; i++)
    {
      Phi[i] = tempA[2*i];
    }

  free(tempA);


  /* printf("Final result (incl. mult. by 1/Nx * Ny)  = %i: \n \n",); */

  for (i=0; i<Ng; i++)
    {	
      Phi[i] = Phi[i]/(Ng);
    }

  return 0;
}
/* ------------------- end FFT_ SOLVE ---------------------*/

/************************************************************************/
/************************************************************************/
/*********************   four1  *****************************************/
/************************************************************************/

/* routine for complex fft by Anne C. Elster July 1992 */
/* Based on program from "Numerical Recipes"           */


int four1(double data[],int nn, int isign)

{
  int n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  double tempr,tempi;

  n=nn << 1;
  j=1;
  for (i=1;i<n;i+=2) 
    {
      if (j > i)
        {
          SWAP(data[j],data[i]);
          SWAP(data[j+1],data[i+1]);
        }
      m=n >> 1;
      while (m >= 2 && j > m)
        {
          j -= m;
          m >>= 1;
        }
      j += m;
    }
  mmax=2;
  while (n > mmax)
    {
      istep=2*mmax;
      theta=6.28318530717959/(isign*mmax);
      wtemp=sin(0.5*theta);
      wpr = -2.0*wtemp*wtemp;
      wpi=sin(theta);
      wr=1.0;
      wi=0.0;
      for (m=1;m<mmax;m+=2)
        {
          for (i=m;i<=n;i+=istep)
            {
              j=i+mmax;
              tempr=wr*data[j]-wi*data[j+1];
              tempi=wr*data[j+1]+wi*data[j];
              data[j]=data[i]-tempr;
              data[j+1]=data[i+1]-tempi;
              data[i] += tempr;
              data[i+1] += tempi;
            }
          wr=(wtemp=wr)*wpr-wi*wpi+wr;
          wi=wi*wpr+wtemp*wpi+wi;
        }
      mmax=istep;
    }

  return 0;
}

/* ------------------------ end fft routines from Recipies ----------*/

/******************  end FFT routines  **********************************/

