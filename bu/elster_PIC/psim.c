#include "declarations.h"

/***************** Set up Pthread variables: ****************/
int 	team_id, 	/*Identifier for team of threads */
  numthreads; 	/* no. of Pthreads used */

/************ Local functions used: **************/




/******** variable determine format  of "double" for printf: */

/* Dynamically allocated 1-d arrays: */

double
*Phi,		/* pointer to array Phi (malloced based on input) */
  *Rho,		/* space charge density at each node */
  *Ex,		/* array for E-field in x direction */
  *Ey,		/* array for E-field in y direction */
  *Part_x,	/* Array containing x-locations of particles */
  *Part_y,	/* Array containing y-locations of particles */
  *Vx,		/* Arrays of velocities of each particle */
  *Vy;


int 	count=0, err;		/* trace-flag */

double	/* temps for position tracking (oscil) of part. no.1: */
temp1xa = 0.0,	
  temp1xb = 0.0,
  temp1xc = 0.0,
  temp1ya = 0.0,	
  temp1yb = 0.0,
  temp1yc = 0.0,

  /* temps for position tracking (oscil) of part. no.2: */
  temp2xa = 0.0,
  temp2xb = 0.0,
  temp2xc = 0.0;

FILE 	/* pointers to input and output files, respectively */
*gnuplot,
  *ifp,	/* pointer to input file for gen parameters */	
  *ofpr,  /* pointer to distr.out -- trace of max */
  *ofpt,  /* pointer to general trace file (parameters. Rho, etc) */
  *ofpw,  /* pointer to file for WARNING messages */

  /* pointers to particle trace files: */
  *ofp,
  *ofp1, *ofpy1, *ofp2, *ofp3, *ofp4, *ofp5, *ofp6, *ofp7, *ofp8, *ofp9;

char palette[768], image[nxpix*nypix], outfilename[80];

/************************  MAIN ****************************************/

int main(int argc, char **argv) {
#ifdef PIC_PETSC
  char desc[] = "Petsc version of SOR solver for PIC";
  err = PetscInitialize(&argc,&argv,(char *)0,desc); checkerr(err);
#else
  MPI_Init(&argc, &argv);	
#endif

  mpi_pic_init();

  int	 i,j, frame = 0;       	/* loop counters */

#ifdef DEBUG
  fprintf(stderr, "[%d] Position (%d, %d), up: %d, down:%d, right:%d, left:%d\n",
          cart_rank,cart_coords[0], cart_coords[1], p_over, p_under, p_right, p_left );

#endif	
  /************ simulation variables: *******************************/
  int
    global_Nx, global_Ny, /* global field grid size */
    Nx, Ny,       	/*  no. grid points in X AND y direction */
    Ng,		/* total number of field grid points */
    Np,		/* total number of particles in simulation */
    Px, Py;


  double
    eps,		/* dielectric constant (typically 1.0 in our sim.)*/
    mass,		/* mass of a simulation particle (typically 1.0) */
    q,		/* charge of a simluation particle (typically -1.0) */
    rho0,		/* rho0 = mean charge density = q * n0, (typ. -0.01)
                           n0 = mean number density */
    rho_k,		/* scaling factor for part_rho = rho_p/(hx * hy) */
    drag = 0.0,	/* drag coefficient (mult. by velocity in ACCEL_PART)*/
    Lx, Ly,		/* system size in x and y, respectively */
    hx, hy,	    	/* spacing  in x and y, respectively */
    t,		/* time variable (time spent upto now) */
    del_t,		/* time step for updating of field (fixed) */
    t_max;	 	/* max. time limit for simulation */

  double	vdrift;		/* Drift velocity */

  /* Timing variables  ( ksr: double; Sun: long) */
  /* Ksr: user_seconds() ; Sun: clock() */

  double simstart, simstop;
  simstart = MPI_Wtime();
  double
    time1, time2, tmpt1, tmpt2, tinit, tpart_rho, tpush_v,
    tsolve, tpush_loc, tfield_grid, tmpta, tmptb;

  double	 phi_min, phi_max;
  /*************************** begin main ********************************/

  if(world_rank ==0)
    fprintf(stdout, " \n\n 2-d simulation w/ periodic boundaries starting .. \n\n");
  /*
    printf("Plasma  oscilation tests turned off.\n");
  */
  /***** set-up trace/output files: ************************/
  if(world_rank ==0){
    ofpw = fopen("warnings.sim","w");
    ofpt = fopen("test.trace","w");
    /*Trace_set_up(Nx, Ny);*/
    
    /* *********** initialize timing variables, etc ***************** */
    tinit=0.0;
    tpart_rho = 0.0;
    tpush_v = 0.0;
    tsolve = 0.0;
    tpush_loc = 0.0;
    tfield_grid = 0.0;
    tmpta = 0.0;
    tmptb = 0.0;
	
    /* Start timing -- Performance measurement calls:*/

    tmpt1 = user_seconds(); 
 
    /*
      pmon_delta(&pmon_buff);
    */

    ConstInit(&global_Nx, &global_Ny, &Ng, &Np, &Px, &Py,
              &eps, &mass, &q, &rho0, &rho_k, &drag,
              &Lx, &Ly, &hx, &hy, &t, &del_t, &t_max);
    /* Note: ConstInit also globally allocated the following arrays: 
       Phi, Rho, Ex, Ey, Part_x, Part_y, Vx, Vy
    */


    DataInit( Ng, Np, &vdrift,  Lx, Ly, t, Phi, Part_x, Part_y, Vx, Vy);
  		
    /* the following commented lines are for plotting with the hdf format */
    //black_palette(palette);
    //palette_entry (palette,254,0,0,255);
    //scatterplot 
    //         (image,nsqpix,nsqpix,Part_x,Part_y,Np,1,0.0,Lx,0.0,Ly,254);
    //put ("initpos.hdf",image,nsqpix,nsqpix,palette); 
  }	
  /* Broadcast all data read from file */
  MPI_Bcast(&global_Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);	
  MPI_Bcast(&global_Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);	
  MPI_Bcast(&Ng, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Np, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Px, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Py, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
  MPI_Bcast(&Lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(&hx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&hy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&t_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&del_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&drag, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rho_k, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rho0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&q, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
  // create datatypes only first when we now matrix size
  Nx = global_Nx /dim_sizes[1];
  Ny = global_Ny /dim_sizes[0];
#ifdef DEBUG
  fprintf(stderr, "[%d]\tglobal_nx: %d, global_ny: %d, nx: %d, ny: %d \n",cart_rank, global_Nx, global_Ny, Nx, Ny);
#endif
  create_mpi_types( Nx,  Ny);	

  if( world_rank != 0 ){

    //Phi, Rho, Ex, Ey, Part_x, Part_y, Vx, Vy
    Phi = xmalloc( Nx *  Ny * sizeof(double));
    Rho = xmalloc( Nx *  Ny * sizeof(double));
    Ex = xmalloc( Nx *  Ny * sizeof(double));
    Ey = xmalloc( Nx *  Ny * sizeof(double));
    Part_x = xmalloc(Np * sizeof(double));
    Part_y = xmalloc(Np * sizeof(double));
    Vx = xmalloc(Np * sizeof(double));
    Vy = xmalloc(Np * sizeof(double));
  		
    DataInit( Ng, Np, &vdrift,  Lx, Ly, t, Phi, Part_x, Part_y, Vx, Vy);

  } // if world_rank != 0 ends
	
#ifdef PIC_PETSC
  fprintf(stderr, "calling petsc_init\n"); fflush(stdout);
  petsc_init( Nx,  Ny, Rho);
  fprintf(stderr, "called petsc_init ok\n"); fflush(stdout);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
	
  if(world_rank==0)
    {
      time1 = user_seconds();
      printf(" DataInit. took : % 7.5lf seconds \n", ((double) (time1-tmpt1)));
    }

  /********** Set up Pthread environment ***************************************/

  /* Create a team consisting of one pthread per processor in the pset */

  /*
    numthreads = psm_numprocs();
  */
  /* get presto team id: */
  /*
    team_id = pr_create_team (numthreads);

    printf(" pthread team created ... num threads = %i\n", numthreads);
  */

  /* Use all team members to execute functions in parallel.
   * Arguments are passed to Foo(A[], int n] as follows: 
   * pr_pcall( team_id, Foo, copyargs(Foo, (int) n) );      

   * relinquish teams by calling:
   * pr_destroy_team (team_id);  
   */


  /*************  start numerical computations: *****************************/

  /* Set the charge density array to zero: */
 

  for (i=0; i<Ng; i++)
    {
      Rho[i] = 0.0;
    }
  
#ifdef PIC_SOR   
  if(cart_coords[0]==0)    
    for (i = 0; i <  Nx; ++i) 
      Rho[i] = UPPER_DIRICHLET_BORDER;
#endif

  /* Figure in what the particles contribute to Rho: */
  /* In parallel version:
   *   Use all teams to execute PART_RHO in parallel.
   *   The arguments are passed to PART_RHO using pr_call. */

  /*
    pr_pcall( team_id, PART_RHO, copyargs((double) rho_k, (int) Np, (int) Nx, (int) Ny, (double) hx, (double) hy, Rho) );
 
  */
 
  PART_RHO( Np,  Nx,  Ny, rho_k, hx, hy, t, Part_x, Part_y, Rho);

  /*
    phi_min = array_min(Rho, Ng);
    phi_max = array_max(Rho, Ng);
    if(world_rank == 0)
    fprintf(ofpt,"\tAfter first call to part_rho:\n\trho_max = %le rho_min =%le \n", phi_max, phi_min);
  */
  //grayscale(palette);
  //color_plot(image, nsqpix, nsqpix, Rho, Nx, Ny, -1.602e-12, phi_max, 0);
  //put ("out.rho0",image,nsqpix,nsqpix,palette); 



  /* Solve the field equation given charge densities, etc: */

#ifdef DEBUG
  fprintf(stderr, "[%d] entering generic solve\n", cart_rank);
  fflush(stderr);
#endif

  generic_solve(Phi, Rho, eps, Lx, Ly, Nx, Ny);

#ifdef DEBUG
  fprintf(stderr, "[%d] did generic solve\n", cart_rank);
  fflush(stderr);
#endif

  phi_min = array_min(Phi, Ng);
  phi_max = array_max(Phi, Ng);
  /*
    sprintf(outfilename,"outphi.%05i",frame);
    grayscale(palette);
    color_plot(image,nsqpix,nsqpix,Phi, Nx, Ny, phi_min, phi_max);
    put (outfilename,image,nsqpix,nsqpix,palette); 
  */

  if(world_rank==0){
    fprintf(ofpt,"After first call to generic_solve: \n");

    fprintf(ofpt,"phi_max = %le phi_min =%le \n", phi_max, phi_min);
  }

  /* Calculate the field at each node -- store in array Ex and Ey for
   * x- and y-direction, respectively: */
  PERIODIC_FIELD_GRID(Phi, Nx, Ny, hx, hy, Ex, Ey);

  phi_min = array_min(Ex, Ng);
  phi_max = array_max(Ex, Ng);

  //grayscale(palette);
  //color_plot(image,nsqpix,nsqpix,Ex, Nx, Ny, phi_min, phi_max,0);
  //put ("outExO",image,nsqpix,nsqpix,palette); 

  phi_min = array_min(Ey, Ng);
  phi_max = array_max(Ey, Ng);

  
  /*
    grayscale(palette);
    color_plot(image,nsqpix,nsqpix,Ey, Nx, Ny, phi_min, phi_max,0);
    put ("outEyO",image,nsqpix,nsqpix,palette); */
  if(world_rank==0){
    fprintf(ofpt,"After first call to PERIODIC_FIELD_GRID: \n");

    fprintf(ofpt,"Ex_max = %le Ex_min =%le \n", phi_max, phi_min);
    fprintf(ofpt,"Ey_max = %le Ey_min =%le \n", phi_max, phi_min);
  }

  /* Apply field to particle and
   * Pull back velocities a half time-step--Leap-frog method */

  t = -del_t/2;

  /* In parallel version: Use all teams to execute
   * PUSH_V(Ex ,Ey, Np, Part_x,Part_y, Nx, Ny, hx, hy, drag, q, mass,
   *        t, Vx, Vy);  in parallel.
   * External parameters are passed as arguments using pr_call. */

  /*
    pr_pcall( team_id, PUSH_V, copyargs(Ex, Ey, (double) Np, Part_x, Part_y, (int) Nx, (int) Ny, (double) hx, (double) hy, (double) drag, (double) q, (double) mass, (double) t, Vx, Vy) );
 
  */
  PUSH_V(Ex ,Ey, Np, Part_x, Part_y, Nx, Ny, hx, hy, drag,
         q, mass, t, Vx, Vy);

  /*Performance measurement calls.*/
  if(world_rank ==0){
    tmpt1 = user_seconds();
    printf(" DataInit. incl. pull-back of V took : % 7.5lf seconds \n", ((double) (tmpt1-time1)));
  }
  
  /*************  trace set-up calculations: ************************/

  /*
    if(world_rank==0){
    fprintf(ofpt,"Trace after Trace_set_up : \n\n");
    
    Trace1( Nx, Ny, Ng, Np, Px, Py,
    eps, mass, q, rho0, rho_k, Lx, Ly, hx, hy, t, del_t, t_max,
    Phi, Rho, Ex, Ey, Part_x, Part_y, Vx, Vy);
    
    }
  */
#ifdef DEBUG
  fprintf(stderr, "[%d] ready for mainloop\n", cart_rank);
  fflush(stderr);
#endif


  /* *********************** main simulation loop: ************** */
  t = 0.0;
  while (t <= t_max)
    {
      MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG
      fprintf(stderr, "[%d] frame %d\n", cart_rank, frame);
      fflush(stderr);
#endif

      plot_particles(Part_x, Part_y, Np, frame, Nx, Ny, hx, hy);

#ifdef DEBUG
      plot_field(Phi, Nx, Ny, frame);
      fprintf(stderr, "[%d] plotted field no. %d \n", cart_rank, frame);
      fflush(stderr);
#endif
      //printf("timestep %lf\n", t);
      /* Calculate new velocities and positions for each particle: */

      tmpt1 = user_seconds(); 	/*Performance measurement call*/

      /* printf("Apply field to particles and  \n \n"); */
      /* printf("Calculate velocities  ... \n \n"); */

      /* In parallel version: Use all teams to execute
       * PUSH_V(Ex ,Ey, Np, Part_x,Part_y, Nx, Ny, hx, hy, drag, q, mass,
       *        t, Vx, Vy);  in parallel.
       * External parameters are passed as arguments using pr_call. */
 
      /*
        pr_pcall( team_id, PUSH_V, copyargs(Ex, Ey, (double) Np, Part_x, Part_y, (int) Nx, (int) Ny, (double) hx, (double) hy, (double) drag, (double) q, (double) mass, (double) t, Vx, Vy) );
      */
      /*      fprintf(stdout, "%d", frame);
              fflush(stdout);
              MPI_Barrier(CART_COMM);
      */

#ifdef DEBUG
      fprintf(stderr, "[%d] entering push_v\n", cart_rank);
      fflush(stderr);
#endif 
      PUSH_V(Ex ,Ey, Np, Part_x, Part_y, Nx, Ny, hx, hy, drag, q, mass, t, Vx, Vy);
      /*	  printf("%d pushed_v on frame %d\n", cart_rank, frame);
                  fflush(stdout);
                  MPI_Barrier(CART_COMM);*/
      tmpta = user_seconds();		/*Performance measurement call*/
      tpush_v += tmpta - tmpt1;

#ifdef DEBUG
      fprintf(stderr, "[%d] entering push_loc\n", cart_rank);
      fflush(stderr);
#endif 
      PUSH_LOC(Np, Part_x, Part_y, Vx, Vy, Lx, Ly, hx, hy, del_t, t);
      //	     printf("%d pushed loc on frame %d\n", cart_rank, frame);
  	  
      tmpt2 = user_seconds();          /*Performance measurement call*/
      tpush_loc += tmpt2 - tmpta;

      /* Reset the charge desity array to zero: */

#ifndef PIC_SOR
      for (i=0; i < Ng; i++){
        Rho[i] = 0.0;
      }
#else
      if(cart_coords[0]==0)
        for (i = 0; i < Nx; ++i) {
          Rho[i] = UPPER_DIRICHLET_BORDER;
        }
      if(cart_coords[0]== (dim_sizes[0]-1))
        for (i = 0; i < Nx; ++i) {
          Rho[i+(Ny-1)*Nx] = LOWER_DIRICHLET_BORDER;
        }
#endif
      
      /* printf(" Figure in what the particles contribute to Rho: \n); */
       
#ifdef DEBUG
      fprintf(stderr, "[%d] entering part_rho\n", cart_rank);
      fflush(stderr);
      MPI_Barrier(MPI_COMM_WORLD);
#endif 
      PART_RHO( Np, Nx, Ny, rho_k, hx, hy, t, Part_x, Part_y, Rho);
      /*	  printf("%d parted rho on frame %d\n", cart_rank, frame);
                  fflush(stdout);
      */

      tmpt1 = user_seconds();          /*Performance measurement call*/
      tpart_rho += tmpt1 - tmpt2;

      /* printf(" Solve the field equation given charge densities, etc.\n"); */

      generic_solve(Phi, Rho, eps, Lx, Ly, Nx, Ny);
      /*printf("%d solved  on frame %d\n", cart_rank, frame);
        fflush(stdout);*/

#ifdef PIC_SOR
      memcpy(Rho, Phi, Ng*sizeof(double));
#endif

      tmpt2 = user_seconds();          /*Performance measurement call*/
      tsolve += tmpt2 - tmpt1;

      /* Calculate the field at each node -- store in array Ex and Ey for
       *
       * x- and y-direction, respectively: */
      //       MPI_Barrier(CART_COMM);
#ifdef DEBUG
      fprintf(stderr, "[%d] entering periodic_field_grid\n", cart_rank);
      fflush(stderr);
#endif 
      PERIODIC_FIELD_GRID(Phi, Nx, Ny, hx, hy, Ex, Ey);
      /*	  printf("%d perioadic field grided on frame %d\n", cart_rank, frame);
                  fflush(stdout);*/

      tmpt1 = user_seconds();          /*Performance measurement call*/
      tfield_grid += tmpt1 - tmpt2;

      t = t + del_t;
	  
      frame++;

      //if((frame%10)==0) plot_field(Phi, Nx, Ny, frame);
      if(frame == 20) {
        fprintf(stdout,"[%d] ended because frame reached 60\n", cart_rank);
        break;
      } // debug purposes only
     
    }  /* end while t */
  
  simstop = MPI_Wtime();
  /****************** end main simulation loop ****************************/


  /* END timimng: */
  if(world_rank ==0  ){
    time2 = user_seconds();        /*Performance measurement call*/


    printf("   del_t = %le , t_max = %le \n\n",del_t, t_max);
    printf("   No. of time-steps: %i \n" , (int) (t_max/del_t));

    printf(" Part_rho took : % 7.5lf seconds \n", ((double) (tpart_rho)) );
    printf(" Push_v took : % 7.5lf seconds \n", ((double) (tpush_v)) );
    printf(" Push_loc took : % 7.5lf seconds \n", ((double) (tpush_loc)) );
    printf(" Solve took : % 7.5lf seconds \n", ((double) (tsolve)) );
    printf(" Field_grid took : % 7.5lf seconds \n", ((double) (tfield_grid)) );
    printf(" Simulation  took : % 7.5lf seconds \n", ((double) (time2-time1)) );

    printf("Simulation is complete; output files closed. \n \n");

    time2 = user_seconds();        /*Performance measurement call*/
    printf(" Total simulation  took : %f seconds \n", ((simstop-simstart)) );
	
    fprintf(ofpt, "\n\n Final trace: \n\n");

    Trace1( Nx, Ny, Ng, Np, Px, Py,
            eps, mass, q, rho0, rho_k, Lx, Ly, hx, hy, t, del_t, t_max,
            Phi, Rho, Ex, Ey, Part_x, Part_y, Vx, Vy);

    /***************** close trace files: *************************/
    Trace_close();	
  }
  mpexit(); //close and exit(0)
  return 0; /* return successfully */

} /* end main */

/* -------------------- END MAIN ------------------------------------- */


/********  INITIALIZE MPI VARIABLES and communication grid ****/
/* author: Nils Magnus Larsgård */
int mpi_pic_init(){
  /* MPI init*/
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nump);	
	
  int i,j;//counters
	

  /* MPI Cartesian create */
  int reorder=1; // let mpi reorder processes 
  for (i = 0; i < NDIMS; ++i) {
    dim_sizes[i] = 0 ; // set to zero 
    wraparound[i] = 1 ;
  }

#ifdef PIC_FFTW // for the FFTW, we have a flat topology because of the fftw library
  dim_sizes[0] = 1;
  dim_sizes[1] = nump;
#elif PIC_SOR // for our home-made sor, we let mpi decide topology
  MPI_Dims_create( nump, NDIMS, dim_sizes); 
#endif
  MPI_Cart_create( MPI_COMM_WORLD, NDIMS, dim_sizes, wraparound , reorder, &CART_COMM );
  MPI_Comm_rank(   CART_COMM, &cart_rank);
  MPI_Cart_coords( CART_COMM, cart_rank, NDIMS, cart_coords);

  // find cartesian neighbours
  int tmp_coord[2];
	
  tmp_coord[0] = cart_coords[0];
  tmp_coord[1] = cart_coords[1]-1;	
  MPI_Cart_rank( CART_COMM, tmp_coord , &p_left );

  tmp_coord[1] = cart_coords[1]+1;	
  MPI_Cart_rank( CART_COMM, tmp_coord , &p_right );

  tmp_coord[0] = cart_coords[0]-1;	
  tmp_coord[1] = cart_coords[1];	
  MPI_Cart_rank( CART_COMM, tmp_coord , &p_over );
	
  tmp_coord[0] = cart_coords[0]+1;	
  MPI_Cart_rank( CART_COMM, tmp_coord , &p_under );
	
  return 0;
}

/* create MPI datatypes for sending borders of matices */
/* author: Nils Magnus Larsgård*/
int create_mpi_types(int Nx, int Ny){
  MPI_Type_vector(Ny/2, 1, 2*(Nx+2), MPI_DOUBLE, &mat_col ); // single column for a SOR-segment
  MPI_Type_commit(&mat_col);
	
  MPI_Type_vector(Nx/2, 1, 2 , MPI_DOUBLE, &mat_row ); // single row for a SOR-segment
  MPI_Type_commit(&mat_row);
  	
  /* 
   * 4 doubles for mpi_particle: 
   * part_x, part_y, v_x, v_y
   * */
  MPI_Type_vector(1, 4, 1, MPI_DOUBLE, &mpi_particle); 
  MPI_Type_commit(&mpi_particle);
}

/************************************************************************/
/************************************************************************/
/****************     FUNCTION PART_RHO   *******************************/
/************************************************************************/

/*
 *      Author: Anne C. Elster				March 1992
 *						 revised Nov. 1993 
 *					(moved particle loop inside)
 *						pthreads added Dec '93
 *
 *	This fuction assigns the charges the particles contribute
 *	by updating the charge density grid (Rho).
 *
 *	Grid size and the grid spacings given.
 *
 *	The data is passed thru pointers. */

/* 	Uses global Pthread parameters numpthread and team_id */


int  PART_RHO( int Np, int n, int m, double rho_k,
               double hx, double hy, double t,
               double Part_x[], double Part_y[], double Rho[])

     /* Note: Row-major assumed, so expect m = Ny, n = Nx */

{


  /* The _private qualifier is optional here, since "private
   * is the default qualification for automatic variables. */
  //fprintf(stderr, "rho_k er %lf\n", rho_k);
  double 
    invhx = 1/hx ,invhy = 1/hy; /* avoid divisions in loop */	


  double	
    x0, y0,	 	/* temp. position of particle */
    a, b,   	/* dist from particle to cell side (see fig) */
    tmpi, tmpj;	/* temp. variables to make truncation work */

  int	l,		/* particle loop variable */
    me,		/* local pthread ID */
    numthreads,	/* no. of parallel threads available */
    my_work,	/* no. of local particles operated on */
    my_begin_ptr,	/* ptr to beginning of local work */
    my_end_ptr,	/* ptr to end of local work */
    index,  	/* global array index */
    i,j;		/* node entry of lower corner of current cell */

  
  for ( l = 0; l < Np; l++) {
    if(Part_x[l] == PARTICLE_EMPTY || Part_y[l] == PARTICLE_EMPTY ){
      continue;
    }
    x0 = Part_x[l];
    y0 = Part_y[l];
    
    /* Calculate node entry of lower corner of current cell:
       
    (i+1,j)	---------------(i+1,j+1)
    |			  |
    |			  |
    |			  |
    |			  |
    |       x		  |
    |     (x0,y0)	  |
    |			  |
    |			  |
    |			  |
    (i,j) ----------------- (i,j+1)
    
    */
    
    tmpj = x0 * invhx;
    tmpj = floor(tmpj);
    j = (int) (tmpj);
#ifdef DEBUG
    fprintf(stderr, "[%d] j: %d \t x0:%f \n", cart_rank, j, x0);
#endif
    tmpi = y0 * invhy;
    tmpi = floor(tmpi);
    i = (int) (tmpi);
#ifdef DEBUG
    fprintf(stderr, "[%d] i: %d \t y0:%f\n", cart_rank, i, y0);
#endif
    /* Calculate charge contributed to each node:
       
    
    (i+1,j)	---------------(i+1,j+1)
    |	   		  |
    |	   		  |
    |      		          |
    |       		  |
    |	   		  |
    |<-a-->x (x0,y0)	  |  hy
    |      ^		  |
    |      |		  |
    |      b		  |
    |      |		  |
    |      v		  |
    (i,j) ----------------- (i,j+1)
    hx	
    
    */
    a = x0 - (j*hx);
    b = y0 - (i*hy);
    
    /* rho_k = rho_p/(hx*hy) */
    /* Note: row-major starge assumed; expect m = Ny, n = Nx */
    
    index = (i * n)  + j;
    
    /* NOTE: Parallel version needs to use _gspwt or locks to ensure consistent
       results when updating Rho since the updates are READ_WRITES! */
    
    if ( ((i+1) < m) && ((j+1) < n) )
      { /* interior */
        Rho[index] += (hx - a) * (hy - b) * rho_k; /* (i,j) */
        
        Rho[index + 1] += a * (hy - b) * rho_k;    /* (i, j+1) */
        Rho[(i+1)*n + j] += (hx - a) *  b * rho_k; /* (i+1, j) */
        Rho[(i+1)*n + (j+1)] += a * b * rho_k;     /* (i+1,j+1) */
        
      }
    
#ifndef PIC_FFTW
#ifndef PIC_SOR  /* SOR is not periodic */
    else         
      if ( ((j+1) == n)  && ((i+1) < m) )
        { /* top */
          Rho[index] += (hx - a) * (hy - b) * rho_k;	/* (i,j) */
          Rho[i*n] += a * (hy - b) * rho_k; 	        /* (i, 0) */
          Rho[(i+1)*n + j] += (hx - a) *  b * rho_k; 	/* (i+1, j) */
          Rho[(i+1)*n ] += a * b * rho_k;       	/* (i+1, 0) */
            
        }
        else 
          if  ( ((i+1) == m) && ((j+1) < n) )
            {  /* right */
              Rho[index] += (hx - a) * (hy - b) * rho_k;	/* (i,j) */
              Rho[index + 1] += a * (hy - b) * rho_k;    	/* (i, j+1) */
              Rho[j] += (hx - a) *  b * rho_k;           	/* (0, j) */
              Rho[j+1] += a * b * rho_k;             	/* (0, j+1) */
            }
          else 
            if ( ((i+1) == m) && ((j+1) == n) )
              { /* upper corner */
                Rho[index] += (hx - a) * (hy - b) * rho_k;	/* (i,j) */
                Rho[i*n] += a * (hy - b) * rho_k;         	/* (i, 0) */
                Rho[j] += (hx - a) *  b * rho_k;           	/* (0, j) */
                Rho[0] += a * b * rho_k;                	/* (0, 0) */
              }
#endif
#endif
      /* 
         else if(world_rank==0){	
         //fprintf(ofpw,"\n PART_RHO: WARNING -- No cases hit (i,j) = %i, %i :-( \n", i, j);
         //fprintf(ofpw,"PART_RHO:  t = %e (x0,y0) = (%e, %e) \n", t, x0, y0);
         //fprintf(ofpw,"PART_RHO: (tmpj,tempi) = (%e , %e ) \n\n", tmpj, tmpi);
         //fprintf(stderr, "didnt update rho with particle(%d,%d)\n", i,j);
         }
      */	

    } /* end for each particle */


  /* *************** END PART_RHO   ************************************** */
  return 0;
}



/************************************************************************/
/************************************************************************/
/****************     FUNCTION PERIODIC_FIELD_GRID   ********************/
/************************************************************************/

/*
 *      Author: Anne C. Elster			    March - August 1992
 *
 *	This fuction calculates the electric field at
 * 	each node, given the potential at the nodes, and the
 * 	grid spacings.
 *
 *	This routine assumes PERIODIC boundaries
 *
 *	The data is passed thru pointers. */


int PERIODIC_FIELD_GRID(double Phi[], int Nx, int Ny, double hx, double hy,\
                        double Ex[], double Ey[])

{

  int	
    row, col, index;	/* loop counters */

  double
    inv_2hx = 1/(2*hx),
    inv_2hy = 1/(2*hy);


  /* NOTE: Assuming row-major storage: */

  /* Calculate interior points: */
#pragma omp parallel for private(row, index)
  for (row = 1; row < Ny-1; row++)
    {
      for (col = 1; col < Nx-1; col++)
        {	/* Ex(i,j) = (Phi(i,j-1) - Phi(i,j+1))/(2*hx)
                   Ey(i,j) = (Phi(i-1,j) - Phi(i+1,j))/(2*hy) */

          index = row*Nx + col;
          Ex[index] = (Phi[index-1] - Phi[index+1]) * inv_2hx;
          Ey[index] = (Phi[index-Nx] - Phi[index+Nx]) * inv_2hy;
        }
    }
#ifdef PIC_SOR
  return 0;
#endif    

  /* General border cases: */


  /* left border: (col = 0) */
#pragma omp parallel for private(row, index)
  for (row = 1; row < Ny-1; row++)
    {

      /* Ex(i,0) = (Phi(i,Nx-1) - Phi(i,1))/(2*hx)
         Ey(i,0) = (Phi(i-1,0) - Phi(i+1,0))/(2*hy) */

      index = row*Nx;
      Ex[index] = (Phi[index + Ny-1] - Phi[index+1]) * inv_2hx;
      Ey[index] = (Phi[index-Nx] - Phi[index+Nx]) * inv_2hy;
    }


  /* right border: (col = Nx-1) */
#pragma omp parallel for private(row, index)
  for (row = 1; row < Ny-1; row++)
    {

      /* Ex(i,Nx-1) = (Phi(i,Nx-2) - Phi(i,0))/(2*hx)
         Ey(i,Nx-1) = (Phi(i-1,Nx-1) - Phi(i+1,Nx-1))/(2*hy) */

      index = (row*Nx) + Nx-1;
      Ex[index] = (Phi[index-1] - Phi[row*Nx]) * inv_2hx;
      Ey[index] = (Phi[index-Nx] - Phi[index+Nx]) * inv_2hy;
    }


  /* bottom (south): (row = 0) */
#pragma omp parallel for private(col, index)
  for (col = 1; col < Nx-1; col++)
    {	

      /* Ex(0,j) = (Phi(0,j-1) - Phi(0,j+1))/(2*hx)
         Ey(0,j) = (Phi(Ny-1,j) - Phi(1,j))/(2*hy) */

      index = col;
      Ex[index] = (Phi[index-1] - Phi[index+1]) * inv_2hx;
      Ey[index] = (Phi[(Ny-1)*Nx + index] - Phi[index+Nx]) *inv_2hy;
	
    }




  /* top (north): (row = Ny-1) */
#pragma omp parallel for private(col, index)
  for (col = 1; col < Nx-1; col++)
    {	

      /* Ex(Ny-1,j) = (Phi(Ny-1,j-1) - Phi(Ny-1,j+1))/(2*hx)
         Ey(Ny-1,j) = (Phi(Ny-2,j) - Phi(0,j))/(2*hy) */

      index = (Ny-1)*Nx + col;
      Ex[index] = (Phi[index-1] - Phi[index+1]) * inv_2hx;
      Ey[index] = (Phi[index-Nx] - Phi[col]) *inv_2hy;
    }

  /* Corner cases: */
  /* Generic index = row*Nx + col */

  /*      Lower left (SW)  (0,0) */

  /* Ex(0,0) = (Phi(0,Nx-1) - Phi(0,1))/(2*hx)
     Ey(0,0) = (Phi(Ny-1,0) - Phi(1,0))/(2*hy) */

  Ex[0] = (Phi[Nx-1] - Phi[1]) * inv_2hx;
  Ey[0] = (Phi[(Ny-1)*Nx] - Phi[Nx]) * inv_2hy;


  /*      Upper left (NW) (Ny-1,0) */


  /* Ex(Ny-1,0) = (Phi(Ny-1,Nx-1) - Phi(Ny-1,1))/(2*hx)
     Ey(Ny-1,0) = (Phi(Ny-2,0) - Phi(0,0))/(2*hy) */

  Ex[(Ny-1)*Nx] = (Phi[(Ny-1)*Nx + Nx-1] - Phi[(Ny-1)*Nx + 1]) * inv_2hx;
  Ey[(Ny-1)*Nx] = (Phi[(Ny-2)*Nx] - Phi[0]) * inv_2hy;


  /*      Lower right (SE)  (0, Nx-1) */

  /* Ex(0,Nx-1) = (Phi(0,Nx-2) - Phi(0,0))/(2*hx)
     Ey(0,Nx-1) = (Phi(Ny-1,Nx-1) - Phi(1,Nx-1))/(2*hy) */

  Ex[Nx-1] = (Phi[Nx-2] - Phi[0]) * inv_2hx;
  Ey[Nx-1] = (Phi[(Ny-1)*Nx + Nx-1] - Phi[Nx + Nx-1]) * inv_2hy;

  /*      Upper right (NE)  (Ny-1,Nx-1) */

  /* Ex(Ny-1,Nx-1) = (Phi(Ny-1,Nx-2) - Phi(Ny-1,0))/(2*hx)
     Ey(Ny-1,Nx-1) = (Phi(Ny-2,Nx-1) - Phi(0,Nx-1))/(2*hy) */

  index = row*Nx + col;
  Ex[(Ny-1)*Nx + Nx-1] = (Phi[(Ny-1)*Nx + Nx-2] - Phi[(Ny-1)*Nx])*inv_2hx;
  Ey[(Ny-1)*Nx + Nx-1] = (Phi[(Ny-2)*Nx + Nx-1] - Phi[Nx-1]) * inv_2hy;



  /* *************************** END PERIODIC_FIELD_GRID ********************* */
  return 0;
}
/************************************************************************/
/************************************************************************/
/****************     FUNCTION PUSH_V   *********************************/
/************************************************************************/

/*
 *      Author: Anne C. Elster				Mar. 1992
 *						revised Nov. 1992
 *   						revised Nov. 1993
 *					(combined FIELD_PART and V_PART)
 *					parallelized using SPC Dec '93
 *
 *	This fuction calculates the electric field at each
 *	particle location (x0,y0), given the field grid,
 *	its size, and the grid spacings.
 *
 *	This fuction then uses the ACCELERATION MODEL to calculate the new
 *	speed (vx1,vy1) of a particle at location (x0,y0) with
 *	initial  speed (vx0, vy0), via the LEAP-FROG method.
 *	Given are the force F = q E, where E is the field (Ex, Ey)
 *	at that particle, particle mass m, time-step (dt), and the grid.
 *
 *	The Leap-Frog method models the particle's movement by having
 *	the velocity update lag a ha;f time-step behind the update
 *	of the particle's positioni (position (x,y) "leaping over"
 *	the velocity, then the other way around:
 *
 *	v^{n + 1/2} =  v^{n - 1/2} + (F((x^n,y^n))/ m) * del_t
 *
 *	x^{n + 1} = x^{n} + v^{n + 1/2} * del_t (see fuction X_PART)
 *
 *	The particle locations are updated by the accompanying routine
 *	PUSH_LOC.
 *	

 * 	WARNING: ROW-MAJOR data-storage assumed!
 *
 *	The data is passed thru pointers. */


int PUSH_V(double Ex[], double Ey[],\
	   double Np, \
           double Part_x[], double Part_y[],\
           int Nx, int Ny,\
           double hx, double hy,\
           double drag, double q, double mass,\
           double del_t,\
           double Vx[], double Vy[])

     /* Vx[], Vy[]:  Out varibles -- velocities */	

     /* Nx, Ny:	grid size */

{
  /* The _private qualifier is optional here, since "private
   * is the default qualification for automatic variables. */

  double 	a, b,	         /* dist from particle to cell side (see fig) */
    x0, y0,		 /* particle locations */
    tmpi, tmpj,	/* temp varables to make truncation work */
    vx, vy, 	 /* particle velocitties */
    Epart_x, Epart_y, /* field at particle */
    Fx, Fy; 	  /* force exerted on a particle */

  int
    me,		/* local pthread ID */
    numthreads,	/* no. of parallel threads available */
    my_work,	/* no. of local particles operated on */
    my_begin_ptr,	/* ptr to beginning of local work */
    my_end_ptr,	/* ptr to end of local work */

    index,  /* pointer to array location */
    l, 	/* particle loop variable */
    i,j;	/* node entry of lower corner of current cell */

  /* Precompute inverses to avoid divisions in particle loop */

  double 
    invhx = 1/hx ,invhy = 1/hy,	
    inv_hx_hy = 1/(hx*hy), 
    inv_mass  = 1/mass;

  /***********  begin parallelized PUSH_V: *****************************/

  /* Parallel code: */
  /* get calling pthread's presto team member ID */
  /*
    me  = pr_mid();
  */
 
  /* Partition the iteration space such that each team member 
   * works on one partition */
  /*
    numthreads = pr_tsize();
  */
 
  /* sort of ceil(Np/numthreads), i.e. no. of local particles */
  /*
    my_work = (Np + numthreads -1) / numthreads;
  */
 
  /* claculate starting local array pointer */
  /*
    my_begin_ptr = me * my_work;
  */
 
  /* calculating ending array pointer for my_work */
  /*
    my_end_ptr = my_begin_ptr + my_work -1;
  */
 
  /* if Np is not divisible by numthreads, adjust upper pointer */
  /*
    if (my_end_ptr > Np-1) my_end_ptr = Np-1;
 
    fprintf(ofpt,"PUSH_V: Ptr limits for thread = %i : %i - %i \n",me, my_begin_ptr, my_end_ptr);
 
  */
  /* Do my share of the work: */
 
  /* Parallel version: 
     for (l = my_begin_ptr; l <= my_end_ptr; l++)
  */

  /* Serial version: */
  for ( l = 0; l < Np; l++) 
    {
      if(Part_x[l] == PARTICLE_EMPTY)
      	continue;
      /***************   begin standard subroutine push_v ***********************/

      x0 = Part_x[l];
      y0 = Part_y[l];
      vx = Vx[l];
      vy = Vy[l];



      /* Calculate node entry of lower corner of current cell:

      (i+1,j)	---------------(i+1,j+1)
      |			  |
      |			  |
      |			  |
      |			  |
      |       x		  |
      |     (x0,y0)	  |
      |			  |
      |			  |
      |			  |
      (i,j) ----------------- (i,j+1)
		
	
      NOTE: i = row-index (y-axis), j = column-index (x-axis)
      */

      /*
        i = (int) (floor(y0/hy));
        j = (int) (floor(x0/hx));
      */

      tmpj = x0 * invhx;
      tmpj = floor(tmpj);
      j = (int) (tmpj);

      tmpi = y0 * invhy;
      tmpi = floor(tmpi);
      i = (int) (tmpi);


      /* Calculate field at particle location using bi-linear interpolation:


      (i+1,j)	---------------(i+1,j+1)
      |	   		  |
      |	   		  |
      |      		  |
      |       		  |
      |	   		  |
      |<-a ->x (x0,y0)	  |  hy
      |      ^		  |
      |      |		  |
      |      b		  |
      |      |		  |
      |      v		  |
      (i,j) ----------------- (i,j+1)
      hx	
	
      NOTE: i = row-index (y-axis), j = column-index (x-axis)
      */
      a = x0 - (j*hx);
      b = y0 - (i*hy);

      index = (i * Nx)  + j; 

      /* NOTE: In the parallel version we do not need to lock Ex, Ey, Vx or Vy
       * since Ex and Ey are  only accessed though READs in this case, and
       * there is only  one unique WRITE for each particle. */

      if (  ((i+1) < Ny) && ((j+1) < Nx) )
        {
          /* Epart_x = Ex(i,j) + Ex(i,j+1) + Ex(i+1,j) + Ex(i+1,j+1) */
	
          Epart_x = (Ex[index] * (hx - a) * (hy - b) 
                     + Ex[index+1] * a * (hy - b)
                     + Ex[(i+1)*Nx + j] * (hx - a) *  b
                     + Ex[(i+1)*Nx + (j+1)] * a * b) * inv_hx_hy;

          Epart_y = (Ey[index] * (hx - a) * (hy - b)
                     + Ey[index+1] * a * (hy - b)
                     + Ey[(i+1)*Nx + j] * (hx - a) *  b
                     + Ey[(i+1)*Nx + (j+1)] * a * b) * inv_hx_hy;
        }

      if ( ((j+1) == Nx) && ((i+1) < Ny) )
        {	
          Epart_x = (Ex[index] * (hx - a) * (hy - b)
                     + Ex[i*Nx] * a * (hy - b)
                     + Ex[(i+1)*Nx + j] * (hx - a) *  b
                     + Ex[(i+1)*Nx] * a * b) * inv_hx_hy;

          Epart_y = (Ey[index] * (hx - a) * (hy - b)
                     + Ey[i*Nx] * a * (hy - b)
                     + Ey[(i+1)*Nx + j] * (hx - a) *  b
                     + Ey[(i+1)*Nx] * a * b) * inv_hx_hy;
        }

      if ( ((i+1) == Ny)  && ((j+1) < Nx) )
        {	
          Epart_x = (Ex[index] * (hx - a) * (hy - b)
                     + Ex[index+1] * a * (hy - b)
                     + Ex[j] * (hx - a) *  b
                     + Ex[(j+1)] * a * b) * inv_hx_hy;

          Epart_y = (Ey[index] * (hx - a) * (hy - b)
                     + Ey[index+1] * a * (hy - b)
                     + Ey[j] * (hx - a) *  b
                     + Ey[(j+1)] * a * b) * inv_hx_hy;
        }
	
      if ( ((i+1) == Ny) && ((j+1) == Nx) )
        {	
	
          Epart_x = (Ex[index] * (hx - a) * (hy - b)
                     + Ex[i*Nx] * a * (hy - b)
                     + Ex[j] * (hx - a) *  b
                     + Ex[0] * a * b) * inv_hx_hy;

          Epart_y = (Ey[index] * (hx - a) * (hy - b)
                     + Ey[i*Nx] * a * (hy - b)
                     + Ey[j] * (hx - a) *  b
                     + Ey[0] * a * b) * inv_hx_hy;

        }
      /****** end applying field to particle *************************/

      /* Calculate force on particle */

      Fx = q * Epart_x;
      Fy = q * Epart_y;

      /* Update velocity to be returned: */

      Vx[l]  = vx + ((Fx * inv_mass) * del_t) - (drag * vx * del_t);
      Vy[l]  = vy + ((Fy * inv_mass) * del_t) - (drag * vy * del_t);

    } /* end for each particle */
	
  /* **************** END PUSH_V  ************************************* */
  return 0;
}



/************************************************************************/
/************************************************************************/
/****************     FUNCTION PUSH_LOC   *******************************/
/************************************************************************/

/*
 *      Author: Anne C. Elster				April 1992
 *						revised Nov. 1993
 *
 *	This fuction uses the ACCELERATION MODEL to calculate the new
 *	locations the  particles give their velocities via the LEAP-FROG
 *	method.
 *
 *	The Leap-Frog method models the particle's movement by having
 *	the velocity updates lag a half time-step behind the updates
 *	of the particle positions (position (x,y) "leaping over"
 *	the velocity, then the other way around):
 *
 *	v^{n + 1/2} =  v^{n - 1/2} + (F((x^n,y^n))/ m) * del_t  
 *
 *	x^{n + 1} = x^{n} + v^{n + 1/2} * del_t

 *	See function PUSH_V for the velocity calculations.
 *	
 *
 *	When the particle hits a boarder, the particle stops there,
 *	and bflag gets set.
 *
 *	All data is passed thru pointers. */

/*      May use hx, hy, and the diff. btw the old and new location to
 *	decide whether to reduce time-step  -- not yet implemented! */


int PUSH_LOC(int Np, double Part_x[], double Part_y[],\
             double Vx[], double Vy[],\
             double Lx, double Ly, double hx, double hy, double del_t, double t)


{
  double	loc_x1, loc_y1; /* updated position */


  int 	i, 		/* particle loop variable */
    bflag;		/* flag if particle hit border */

  /***********  begin PUSH_LOC **************************************/

  for ( i = 0; i < Np; i++)
    {
      if(Part_x[i] == PARTICLE_EMPTY)
      	continue;
      /* printf("Calculate new particle location ... \n \n"); */


      loc_x1 = Part_x[i] + Vx[i] * del_t;
      loc_y1 = Part_y[i] + Vy[i] * del_t;
 
      /* Check whether particle moved more that one system length: */

      if (fabs(loc_x1 - Part_x[i]) > Lx)
        {	
          fprintf(ofpw,"WARNING: Large time-step! Particle moved > Lx in one update, t = %le !\n", t);
        }
      if (fabs(loc_y1 - Part_y[i]) > Ly)
        {
          fprintf(ofpw,"WARNING: Large time-step! Particle moved > Ly in one update, t = %le !\n", t);
        }


      /* Check whether reached boarder; if so, implement periodic particle move */
      bflag = 0;
#ifdef PIC_SOR

      if(nump==1){ // only truncate location if 1 processor, 
        // see end of method for migration of particles
        loc_x1 = (loc_x1 >= Lx ) ? Lx : loc_x1;
        loc_x1 = (loc_x1 <= 0.0) ? 0.0 : loc_x1;
        loc_y1 = (loc_y1 >= Ly ) ? Ly : loc_y1;
        loc_y1 = (loc_y1 <= 0.0 ) ? 0.0 : loc_y1;	
      } 
#else
      if (loc_x1 >= Lx)
        {
          bflag = 10;
          loc_x1 = loc_x1 - Lx;
        }
      else
        {
          if (loc_x1 < 0.0)
            { /* note: loc_x1 now negative! */
              bflag = 20;
              loc_x1 = Lx + loc_x1;
            }
        }

      if (loc_y1 >= Ly)
        {
          bflag = 11;
          loc_y1 = loc_y1 - Ly;
        }
      else
        {
          if (loc_y1 < 0.0)
            { /* note: loc_y1 now negative! */
              bflag = bflag + 21; /* warn about x1-case, too */
              loc_y1 = Ly + loc_y1;
            }
        }
#endif 

      /* correct for truncation errors: */

      if ((bflag >= 20) && (loc_x1 == Lx) )
        {
          loc_x1 = 0.0;
          fprintf(ofpw,"\n WARNING -- TRUNC ERROR: x1 = Lx; t = %e for ptlc %i;", t, i);
          fprintf(ofpw," bflag = %i\n ", (bflag));
          fprintf(ofpw," old pos = (%e, %e)\n", Part_x[i],Part_y[i]);
          fprintf(ofpw," new pos = (%e, %e)\n", loc_x1, loc_y1);
        }
      if ((bflag > 20) && (loc_y1 == Ly) )
        {
          loc_y1 = 0.0;
          fprintf(ofpw,"\n WARNING -- TRUNC ERROR: y1 = Ly;  t = %e for ptcl %i;", t, i);
          fprintf(ofpw," bflag = %i\n ", (bflag));
          fprintf(ofpw," old pos = (%e, %e)\n", Part_x[i],Part_y[i]);
          fprintf(ofpw," new pos = (%e, %e)\n", loc_x1, loc_y1);
        }

      Part_x[i] = loc_x1;
      Part_y[i] = loc_y1;

      /*
        Trace_Oscil( i,  loc_x1,  loc_y1 t, Nx,  Ny, Phi;
      */


    } /* end for each particle */

	
  /*********** migrate particles **************/

  { 
    MPI_Request reqs[Np];
    double sendings[Np*4];
    int sending_index=-1;
    short to_send[Np];
    int numreqs=0, particle_tag=938214;// tag is a random number
    for (i = 0; i < Np; ++i) {
      reqs[i] =0;
      to_send[i]=-1;
    }
    for (i = 0; i < Np ; ++i) {
      if(Part_x[i] == PARTICLE_EMPTY)
        continue;			
      
      // test for part_x here	
      if(Part_x[i]>Lx){ 
        // test if we are right border-process in cartesian system
        /*        if(cart_coords[1]==(dim_sizes[1]-1))
                  Part_x[i] = Lx; 
                  else{ // if not, i-send it to process on the right */
        Part_x[i] -= Lx; 
        to_send[i] = p_right;
      
      }else if(Part_x[i]<0){
        /*        if(cart_coords[1] == 0 )
                  Part_x[i] = 0;
                  else{ // if not, i-send it to process on the left*/
        Part_x[i] += Lx; 
        //printf("%d plans on a send left to %d\n", cart_rank, p_left);
        to_send[i] = p_left;
      
      }
  
      // test for Part_y here if not sent already
  
      if(to_send[i]==-1)
        if(Part_y[i]>Ly){ 
#ifdef PIC_SOR
          if(cart_coords[0]==(dim_sizes[1]-1))
            Part_y[i] = Ly;
          else
#endif
            {
              Part_y[i] -=Ly;
              to_send[i] = p_under;
            }
        }else if(Part_y[i]<0){ // upper border?
#ifdef PIC_SOR
          if(cart_coords[0]==0)
            Part_y[i] = 0;
          else
#endif
            {
              Part_y[i] +=Ly;
              to_send[i] = p_over;
            }
        }
			
			
      //if this particle is marked with a process, send it to destination
      if(to_send[i]>-1){
        sendings[++sending_index] = Part_x[i];
        sendings[++sending_index] = Part_y[i];
        sendings[++sending_index] = Vx[i];
        sendings[++sending_index] = Vy[i];
        //printf("%d -> %d : pending \n", cart_rank, to_send[i]); fflush(stdout);
				
        MPI_Isend( &(sendings[sending_index-3]), 1, mpi_particle, 
                   to_send[i], particle_tag, CART_COMM, &reqs[++numreqs] );
				
        /*printf("\n\n%d -> %d : particle sent with pos(%7.5lf, %7.lf) and speed(%7.5lf, %7.lf)\n", 
          cart_rank, to_send[i], Part_x[i], Part_y[i] , Vx[i], Vy[i]); 
          fflush(stdout);
        */
        // nullify the sent particle
        Vx[i] = PARTICLE_EMPTY;
        Vy[i] = PARTICLE_EMPTY;
        Part_x[i] = PARTICLE_EMPTY;
        Part_y[i] = PARTICLE_EMPTY;
      }
    }
    int sumreqs;
    /*int MPI_Allreduce ( void *sendbuf, void *recvbuf, int count, 
      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm )*/
    MPI_Allreduce( &numreqs, &sumreqs, 1, MPI_INT, MPI_SUM, CART_COMM );
		
    //printf("%d sees %d sends done\n", cart_rank, sumreqs);

    if(sumreqs >0){
      /* wait for all to finish sends until we start probing for messages*/
      MPI_Barrier(CART_COMM);
      //probe for incoming particles, receive if any
      MPI_Status probe_stat;
      int probe_flag=1, num_incoming=0;
						
      while(probe_flag){
        probe_flag =0;
        //printf("%d probing\n", cart_rank); fflush(stdout);
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG , CART_COMM, &probe_flag, &probe_stat);
				
        if( probe_flag == 0){
          //printf("%d has no more incoming particles, we skip recv\n", cart_rank); fflush(stdout);
          break;	// no incoming particles
        }		
					
        //printf("%d is still living, probe is %d, stat is %d\n", cart_rank, probe_flag, probe_stat); fflush(stdout);
        i=0;
        while(Part_x[i] != PARTICLE_EMPTY && i<Np)
          i++;
        double particle[4];
        MPI_Recv(&(particle[0]), 1, mpi_particle, 
                 MPI_ANY_SOURCE, particle_tag, CART_COMM, MPI_STATUS_IGNORE);
        Part_x[i] = particle[0];
        Part_y[i] = particle[1];	
        Vx[i] = particle[2];
        Vy[i] = particle[3];
        /*printf("\nNN -> %d particle successfully received\n", cart_rank);
          printf("%d got new particle to pos (%7.5lf, %7.5lf) with speed (%7.5lf, %7.5lf) \n", 
          cart_rank, Part_x[i], Part_y[i], Vx[i], Vy[i]);
        */

      }
      // wait until all particles are sent
      if( 0 < numreqs ){
        //printf("%d is wating for %d reqs to complete reqs-size: %d\n", cart_rank,numreqs, sizeof(reqs));
        //fflush(stdout);
        MPI_Waitall(numreqs, &(reqs[0]), MPI_STATUSES_IGNORE ); // TODO: use status for something?
      }
      //printf("%d migrating was a success\n", cart_rank);
      //mpexit();			
    }
  }
  MPI_Barrier(CART_COMM);

  /* ************************* END PUSH_LOC ******************************** */
  return 0;
}

/************************************************************************/
/************************************************************************/
/****************************   Trace_set_up  ***************************/
/************************************************************************/

/*    Anne C. Elster					Nov. 1993	*/
/*									*/
/*  Sets up trace files                                                 */
/*									*/

int Trace_set_up( int Nx, int Ny)

{
  /************************** open trace files **************************/
  /*	Track each particle by reading in their respective locations
   *	from arrays Part_x and Part_y and calling the apropriate functions.
   *	Write result to file named p.out
   */

  ofp = fopen("plot2d.out","w");
  ofpr = fopen("distr.out","w");
  ofp1 = fopen("p.out1", "w");
  ofpy1 = fopen("p.outy1", "w");
  ofp2 = fopen("p.out2", "w");
  ofp3 = fopen("p.out3", "w");
  gnuplot = fopen("gnuplot.initial", "w");
  /*
    ofp4 = fopen("p.out4", "w");
    ofp5 = fopen("p.out5", "w");
    ofp6 = fopen("p.out6", "w");
    ofp7 = fopen("p.out7", "w");
    ofp8 = fopen("p.out8", "w");
    ofp9 = fopen("p.out9", "w");
  */

  /* set up file for plotting using PLOT2dx: */

  fprintf(ofp, "* Output from mainhorz.c -- particle simulator \n");
  fprintf(ofp, "*  by Anne C. Elster Nov. 1992 \n* \n");
  fprintf(ofp, "MANIFEST: V4.4, PLOT.2D, NORMAL; \n");
  fprintf(ofp, "%i %i \n", Nx*Ny, 1);
  fprintf(ofp, "Potentials -- simulation: \n");

  /* set up file for plotting using PLOT2dx: */

  fprintf(ofpr, "* Output from mainhorz.c -- particle simulator \n");
  fprintf(ofpr, "*  by Anne C. Elster Nov. 1992 \n* \n");
  fprintf(ofpr, "MANIFEST: V4.4, PLOT.2D, NORMAL; \n");
  fprintf(ofpr, "%i %i \n", Nx*Ny, 1);
  fprintf(ofp, "Potentials at max.: \n");

  /* set up trace-files: */
  fprintf(ofp1, "\n \n Tracking of particle no. 1: \n\n");
  fprintf(ofpy1, "\n \n Tracking y-oscillations of particle no. 1: \n\n");
  fprintf(ofp2, "Tracking of particle no. 2: \n\n");
  fprintf(ofp3, "Tracking of particle no. 3: \n\n");
  /*
    fprintf(ofp4, "Tracking of particle no. 4: \n\n");
    fprintf(ofp5, "Tracking of particle no. 5: \n\n");
    fprintf(ofp6, "Tracking of particle no. 6: \n\n");
    fprintf(ofp7, "Tracking of particle no. 7: \n\n");
    fprintf(ofp8, "Tracking of particle no. 8: \n\n");
    fprintf(ofp9, "Tracking of particle no. 9: \n\n");
  */
  /********************** END Trace_set_up ******************************/
  return 0;
}

/************************************************************************/
/************************************************************************/
/****************************   Trace1  *********************************/
/************************************************************************/

/*    Anne C. Elster					Nov. 1993	*/
/*									*/
/* Prints traces to ofpt = test.trace and  opf = plot2d.out		*/
/*									*/

int Trace1(int Nx, int Ny, int Ng, int Np, int Px, int Py,
           double eps, double mass, double q, double rho0, double rho_k,
           double Lx, double Ly, double hx, double hy,
           double t, double del_t, double t_max,
           double Phi[], double Rho[], double Ex[], double Ey[],
           double Part_x[], double Part_y[], double Vx[], double Vy[])

{
  int i,j;


  /*************  write main varibales to trace file: ************************/

  fprintf(ofpt, "Trace of simulation of for plasma oscillations: \n\n");


  fprintf(ofpt,"q = %le, del_ t = %le , t_max = %le\n",q,del_t,t_max);
  fprintf(ofpt,"mass = %le, eps0 = %le , rh0 = %le\n\n",mass,eps,rho0);
  fprintf(ofpt," Lx = %le, Ly = %le , hx = %le  hy= %le\n\n",Lx, Ly, hx, hy);
	


  /*
    fprintf(ofpt, "Particle locations:\n\n");
    for (i = 0; i< Np; i++)
    {
    fprintf(ofpt, "     %7.6f,%7.6f \n",Part_x[i], Part_y[i]);
    }
  */
  fprintf(ofpt, " \n \n ");

  fprintf(ofpt, "Rho:\n");
  for (i=Ny-1; i>=0; i--)
    {
      for (j=0; j<=Nx-1; j++)
        {
          fprintf(ofpt, "%le ",A0INDEX(Rho,Nx,i,j));
        }        
      fprintf(ofpt, " -Rho\n");
    }
  fprintf(ofpt, "\n\n");

  for (i=Ny-1; i>=0; i--)
    {
      for (j=0; j<=Nx-1; j++)
        {
          fprintf(ofpt, "%le ",A0INDEX(Phi,Nx,i,j));
        }
      fprintf(ofpt, " -Phi\n");
    }
  fprintf(ofpt, "\n \n");


  for (i=Ny-1; i>=0; i--)
    {
      for (j=0; j<=Nx-1; j++)
        {
          fprintf(ofpt, "%le ",A0INDEX(Ex,Nx,i,j));
        }
      fprintf(ofpt, "-Ex\n");
    }
  fprintf(ofpt, "\n");

  for (i=Ny-1; i>=0; i--)
    {
      for (j=0; j<=Nx-1; j++)
        {
          fprintf(ofpt, "%le ",A0INDEX(Ey,Nx,i,j));
        }
      fprintf(ofpt,"-Ey\n");
    }
  fprintf(ofpt, "\n \n");

  /******************************** END Trace1() ************************/
  return 0;
}

int Trace_close()
     /********************  close open trace files:  ****************************/

{
  fclose(ofp);
  fclose(ofpt);
  fclose(ofp1);
  fclose(ofpy1);
  fclose(ofp2);
  fclose(ofp3);
  /*
    fclose(ofp4);
    fclose(ofp5);
    fclose(ofp6);
    fclose(ofp7);
    fclose(ofp8);
    fclose(ofp9);
  */
  /*********************** END Trace_close **********************************/
  return 0;
}

/************************************************************************/
/************************************************************************/
/*********************   inite ***************************************/
/************************************************************************/

/* Electron species initialization routine */

inite (double *xe, double *ye,
       double *vxe, double *vye,
       double Lx, double Ly,
       int npe,
       double vdrift) {
  /*  xe: array of x-positions of the electrons (m)
      ye: array of y-positions of the electrons (m)
      vxe: array of the x-velocities of the electrons (m/sec)
      vye: array of the y-velocities of the electrons (m/sec)
      Lx: length of the system in the x-direction
      Ly: length of the system in the y-direction
      npe: number of electrons (xe, ye, vxe, vye are assumed
      to be length of npe)
      vdrift: drift velocity of the electrons (m/sec)
  */
  int Px = 64, Py = 64;
  int i, j;
  double fract, vthe=1.0;

  /*  Load zero-order particle positions and velocities: */
  /*
    for (i=0; i<npe; i++) {
    xe[i] = Lx * (i+0.5) / npe;
    ye[i] = Ly * (i+0.5) / npe;
    nrev (i+1,&fract,2);
    ye[i] = Ly * fract;
    rndmaxw (&vxe[i],&vye[i],vthe);
    vxe[i] = vdrift;
    vye[i] = 0.0;
    }
  */
  for (i=0; i<Px; i++)
    for (j=0; j<Py; j++) {
      A0INDEX(xe,Px,i,j) = Lx * (i+0.5) / Px;
      A0INDEX(ye,Px,i,j) = Ly * (j+0.5) / Py;
    }

  /*  Attempt to put particles slightly outside the system
   *  back in:
   */
  for (i=0; i<npe; i++) {
    xe[i] = (xe[i]<0.0) ? xe[i] + Lx : xe[i];
    xe[i] = (xe[i]>=Lx) ? xe[i] - Lx : xe[i];
    ye[i] = (ye[i]<0.0) ? ye[i] + Ly : ye[i];
    ye[i] = (ye[i]>=Ly) ? ye[i] - Ly : ye[i];
  }

  /*  Check to see that all particles are in the system: */
  for (i=0; i<npe; i++) {
    if ( (xe[i]>=Lx) || (xe[i]<0.0) ) {
      fprintf
        (stderr, "Ptcl no. %i is outside the system, xe = %f\n",
         i, xe[i]);
    }
    if ( (ye[i]>=Ly) || (ye[i]<0.0) ) {
      fprintf
        (stderr, "Ptcl no. %i is outside the system, ye = %f\n",
         i, ye[i]);
    }
  }
  fprintf(ofpt,"The 16 first particle locations were set to the following right after inite: \n");

  for (i = 0; i < 16; i++)
    fprintf(ofpt," %le %le \n", xe[i], ye[i]);



}

/************************************************************************/
/************************************************************************/
/*********************   ConstInit **************************************/
/************************************************************************/

/* Routine for constant initializations  and array allocation.
 * for particle simulation routine  by Anne C. Elster Oct. 1993 
 * Constants are read in from input file named 'tst.dat'.
 */

/* Note: "%lf" format needed for sun; "%f" for KSR */

int ConstInit(int *global_Nx, int *global_Ny, int *Ng, int *Np, int *Px, int *Py,
              double *eps, double *mass, double *q, double *rho0, double *rho_k,
              double *drag, double *Lx, double *Ly, double *hx, double *hy,
              double *t, double *del_t, double *t_max)
     /*
       Arrays to be globally allocated : 
       double Phi[], double Rho[], double Ex[], double Ey[],
       double Part_x[], double Part_y[], double Vx[], double Vy[])
     */

{

  int 
    i,j,l;	/* loop counters */

  int	*Nx = malloc(sizeof(int)), *Ny = malloc(sizeof(int)); /* local values of Nx and Ny */
  double
    rho_p;		/* charge density represented by one sim. partickle */

  /***********   begin ConstInit ****************************************/

  /* Open input file */
  ifp = fopen("tst.dat","r");

  printf("ConstInt: Scanning input parameters from file named <tst.dat> ...\n\n");
  /* scan in fixed constants */


  fscanf(ifp, "%lf",q);
  printf("	q = %le ",(*q));

  fscanf(ifp,"%lf",eps);
  printf("	eps = %le \n",(*eps));

  fscanf(ifp, "%lf",mass);
  printf("	mass  = %le ",(*mass));

  fscanf(ifp, "%lf",rho0);
  printf("	rho0  = %le \n",(*rho0));

  fscanf(ifp, "%lf",drag);
  printf("	drag  = %le \n",(*drag));

  fscanf(ifp, "%lf",del_t);
  fscanf(ifp, "%lf",t_max);

  printf("	del_ t = %le , t_max = %le \n",
         (*del_t), (*t_max));




  /* Read in Lx and Ly values (system size) */

  fscanf(ifp,"%lf", Lx);
  fscanf(ifp, "%lf", Ly);
  printf("	System size = %6.4lf x % 6.4lf \n", (*Lx), (*Ly));
                

  /* Read in grid-size parameters from standard input: */
  fscanf(ifp, "%d %d Nx Ny", global_Nx, global_Ny);
  (*Nx) = (*global_Nx)/dim_sizes[1];
  (*Ny) = (*global_Ny)/dim_sizes[0];

  printf("\tProcessor grid size = %d x %d\n", dim_sizes[0], dim_sizes[1]);
  printf("\tGlobal field grid size = %i x %i  \n", (*global_Nx), (*global_Ny));
  printf("\tlocal grid size = %i x %i  \n", (*Nx), (*Ny));
  fflush(stdout);
  /* Read in no. particles to be tracked from standard input: */

  fscanf( ifp, "%i No of particles", Np);



  /* Calculate "global" parameters based on scanned in values: */

  (*Ng) = (*Nx) * (*Ny); /* number of grid pointss */

  printf("\n\tNp = %i, Ng = %i, rho0 = %le ", (*Np), (*Ng), (*rho0));
  printf(" Ng/Np = % 6.4lf \n", (double) (*Ng)/(*Np));
        
  rho_p = ( (double) (*Ng)/((double) (*Np))) * (*rho0);
  printf("\n\tmean charge per sim particle: rho_p = %le \n",rho_p);
            
            
  (*hx) = (*Lx)/(*global_Nx);			/* spacing in x direction */
  (*hy) = (*Ly)/(*global_Ny);			/* spacing in y direction */

  if ((*hx) != (*hy))
    {
      printf("WARNING: Code currently only tested for hx=hy!! \n");
    }

  printf("\n\tConstInit: hx = % 6.5lf  hy = % 6.5lf \n\n", (*hx),(*hy));


  /* set scale factor for PART_RHO to reduce no. of divisions:  */
  (*rho_k) = rho_p/((*hx) * (*hy));
  /**************************  end global parameter computations ***************/

  printf("\tConstInit: Allocating global arrays based on input parameters ...\n\n");

  /* Allocate arrays : */


  if((Phi = (double *)malloc(sizeof (double) * (*Ng)))==NULL)
    printf("grid allocation failed --- out of memory");

  if((Ex = (double*)malloc(sizeof (double) * (*Ng)))==NULL)
    printf("grid allocation failed --- out of memory");

  if((Ey = (double*)malloc(sizeof (double) * (*Ng)))==NULL)
    printf("grid allocation failed --- out of memory");

  if((Rho = (double*)malloc(sizeof (double) * (*Ng)))==NULL)
    printf("grid allocation failed --- out of memory");

  /* allocate arrays for storing particle locations: */

  if((Part_x = (double*)malloc(sizeof(double) * (*Np)))==NULL)
    printf("allocation of Part_x failed --- out of memory");

  if((Part_y = (double*)malloc(sizeof(double) * (*Np)))==NULL)
    printf("allocation of Part_y failed --- out of memory");

  /* allocate arrays for storing particle velocities: */

  if((Vx = (double*)malloc(sizeof(double) * (*Np)))==NULL)
    printf("allocation of Part_x failed --- out of memory");

  if((Vy = (double*)malloc(sizeof(double) * (*Np)))==NULL)
    printf("allocation of Part_y failed --- out of memory");


  /****************************  end dynamic allocation of arrays *****/


  fprintf(stderr,"[%d] ConstInit: Returning to caller (main) ... \n", cart_rank);

  return 0;

  /*******************  end ConstInit *********************************/
}
/************************************************************************/
/************************************************************************/
/*********************   DataInit ***************************************/
/************************************************************************/

/* Initialization routine, including data read in from input file,
   for particle simulation routine  by Anne C. Elster Oct. 1993 */

/* Uses globally defined two_pi = 6.2831853... */

/* Note: "%lf" format needed for sun; "%f" for KSR */

int   DataInit(int Ng, int Np, double *vdrift, double Lx, double Ly,
               double t,
               double Phi[], double Part_x[], double Part_y[], double Vx[],
               double Vy[])


{

  int 
    i,j,l;	/* loop counters */

  double 	lvdrift;


  /***********   begin DataInit ****************************************/

  /* set intial grid values (including boundary): */
#ifdef DEBUG
  if(world_rank==0)
    printf("DataInit: setting initial field grid values to zero ... \n");
#endif
  /*
    for (i=Ny-1; i>=0; i--)
    {
    for (j=0; j<=Nx-1; j++)
    {
    A0INDEX(Phi,Nx,i,j) = 0.0;
    }
    }
  */
  for (i = 0; i < Ng ; i++)
    Phi[i] = 0.0;
	

  /* Set init location and velocities of particles : */
  /* let half the particles have velocities of the oposite sign: */

#ifdef DEBUG
  if(world_rank==0)
    printf("DataInit:  Np = %i \n", Np);
#endif
  Uniform_Grid_Init(Np, Part_x, Part_y, Lx, Ly, t);

#ifdef DEBUG
  if(world_rank==0)
    printf("DataInit:  Reurned from Uniform_Grid_Init \n");
#endif
  /*
    lvdrift = omega_p * Lx; */ /* vdrift = omega_p * Lx */
#ifdef PIC_SOR
  lvdrift = 0.0;
#else 
  lvdrift = omega_p * 2* Lx; /* vdrift = omega_p * Lx */
  (*vdrift) = lvdrift;
#endif
  for (i = 0; i < Np ; i+=2)
    {
      Vx[i] = 0.0;
      Vy[i] = lvdrift;
    }

  for (i = 1; i < Np ; i+=2)
    {
      Vx[i] = 0.0;
      Vy[i] = -lvdrift;
    }

  /*
    inite(Part_x, Part_y, Vx, Vy, Lx, Ly, Np/2, 0.0);
    inite(&Part_x[Np/2], &Part_y[Np/2], Vx, Vy, Lx, Ly, Np/2, 0.0);
  */

  /* perturb velocities slightly: */
  for (i = 0; i < Np ; i++)
    {
      Vx[i] += cos(2*pi*Part_x[i]/Lx) * lvdrift/100;
      Vy[i] += cos(2*pi*Part_y[i]/Ly) * lvdrift/100;
    }

#ifdef DEBUG
  if(world_rank==0){
    printf("%d: The first 16 particle locations are: \n", world_rank);
    for (i=0; i < 16; i++)
      printf(" %le  %le  \n", Part_x[i], Part_y[i]);
  }
#endif
	
  /*
    printf("output hdf files of initializations ...\n");
    white_palette (palette);
    palette_entry (palette,254,0,0,0);
    scatterplot (image,nxpix,nypix,Part_x,Vx,Np,1,0.0,Lx, -4*lvdrift,4*lvdrift,254);
    put ("outhdf.00000",image,nxpix,nypix,palette); 
    scatterplot (image,nxpix,nypix,Vx,Vy,Np,1,-4*lvdrift,4*lvdrift,-4*lvdrift,4*lvdrift,254);
    put ("inithdf2",image,nxpix,nypix,palette); 
    scatterplot (image,nxpix,nypix,Part_x,Part_y,Np,1,0.0,Lx,0.0,Ly,254);
    put ("inithdf3",image,nxpix,nypix,palette); 
  
    printf(" Input parameters are all set -- see the following files for output: \n \n");
    printf("	plot2d.out	--	PLOT.2D potentials for Manifest \n");
    printf("	test.trace	--	Trace file of init. values and Rho \n");
    printf("	p.out1 .. p.out.9 --	Trace-files of individ. particles \n");
    printf(" \n Memory errors related to mallocs are written to standard output. \n \n");
  */
  /*********************** end read-in parameters ********************/

  return 0;

}
/**********************  End DataInit ***********************************/
/************************************************************************/


/************************************************************************/
/********************* Uniform_Grid_Init  *******************************/
/************************************************************************/
int  Uniform_Grid_Init(int Np, double Part_x[], double Part_y[],
                       double Lx, double Ly, double t)
{

  int	i,j,l, /* loop variables */
    index,
    lflag,  /* flag for detecting particles at -0.0 etc) */
    Px, Py, /* no. of particles in x and y, respectively */
    waves_x = 1,/* no. of sine waves in x desired in a system length */
    waves_y = 1 ;/* no. of sine waves in y desired in a system length */

  /* distances params for generating particle grid: */

  double	dist_px, dist_py, offset_x, offset_y, x_pos, y_pos,
    sin_displ_x, sin_displ_y ;
  double
    del_px,		/* off-set for particle locations -- plasma oscil tst */
    del_py;		/* off-set for particle locations -- plasma oscil tst */


  /**************  begin Uniform_Grid_Init ****************************/
  
  /********      only for the SOR-case      **********/
#ifdef PIC_SOR
  int  div = 2;

  if( ((cart_rank > 0) && (cart_rank < nump)) || (nump==1)){
    for (i = 0; i < Np/dim_sizes[0]; ++i) {
      Part_x[i] = (Lx/2.0) - (Lx/(2*div)) + (i+0.5)*(Lx/(Np*div));
      Part_y[i] = Ly - Ly/div;
    }
    for (i = Np/dim_sizes[0]; i < Np; ++i) {
      Part_x[i] = PARTICLE_EMPTY;
      Part_y[i] = PARTICLE_EMPTY;
    }
  }else 
    for(i=0;i<Np;i++){
      Part_x[i] = PARTICLE_EMPTY;
      Part_y[i] = PARTICLE_EMPTY;
    }
  return 0;
#endif

  Px = (int) sqrt(Np/nump);
  Py = Px;

  printf("Uniform_Grid_Init: Px = %i, Py = %i, Np = %i \n", Px, Py, Np);

  /* Set init location of particles: */

  dist_px = Lx/Px; /* distance btween particles in x-direction */
  dist_py = Ly/Py; /* distance btween particles in y-direction */

  /* Assume row-major storage of particle grid ... */

  /* off-set ("center") first particle away from 0.0;  */

  offset_x = dist_px/2; /* zero off-set first cell-row */
  offset_y = dist_py/2; /* zero off-set first cell-column */


  /* set all particles empty initially*/
  for(i =0; i<Np;i++){
    Part_x[i] = PARTICLE_EMPTY;
    Part_y[i] = PARTICLE_EMPTY;
  }

  printf("Uniform_Grid_Init: displacements and off-sets calculated. \n");

  for ( i = 0; i < Px; i++)
    { /* set particle array incl. off-sets */
      if (i == 0)  x_pos = offset_x;
      else    x_pos += dist_px;
      /*
        sin_displ_x = del_px * sin(two_pi * x_pos * ((double) waves_x) / Lx);
        x_pos += sin_displ_x;
      */

      for ( j = 0 ; j < Py; j++)
        {
          if (j == 0)  y_pos = offset_y;
          else    y_pos += dist_py;
          /*
            sin_displ_y = del_py * sin(two_pi * y_pos * ((double) waves_y) / Ly);
            y_pos += sin_displ_y;
          */

          index = i * Py + j;

          /* begin test for out-of-bounds: */

          if (x_pos >= Lx)
            {
              lflag = 10;
              x_pos = x_pos - Lx;
            }
          else
            {
              if (x_pos < 0.0)
                {  /* note: x_pos now negative! */

                  lflag = 20;
                  x_pos = Lx + x_pos;
                }
            }

          if (y_pos >= Ly)
            {
              lflag = 11;
              y_pos = y_pos - Ly;
            }
          else
            {
              if (y_pos < 0.0)
                {  /* note: y_pos now negative! */
                  lflag += 21;  /* warn about x-case, too */
                  y_pos = Ly + y_pos;
                }
            }

          /* correct for truncation errors: */

          if ((lflag >= 20) && (x_pos == Lx) )
            {
              x_pos = 0.0;
              fprintf(ofpw,"\n WARNING -- SET-UP OUT OF RANGE: x_pos = Lx; t = %e for ptlc %i;", t, i);
              fprintf(ofpw," lflag = %i\n ", lflag);
              fprintf(ofpw," old pos = (%e, %e)\n", Part_x[index],Part_y[index]);
              fprintf(ofpw," Corrected new pos = (%e, %e)\n", x_pos, y_pos);
            }
          if ((lflag > 20) && (y_pos == Ly) )
            {
              y_pos = 0.0;
              fprintf(ofpw,"\n WARNING -- SET-UP OUT OF RANGE: \
 y_pos = Ly;  t = %e for ptcl %i;", t, i);
              fprintf(ofpw," lflag = %i\n ", lflag);
              fprintf(ofpw," old pos = (%e, %e)\n", Part_x[index],Part_y[index]);
              fprintf(ofpw," Corrected new pos = (%e, %e)\n", x_pos, y_pos);
            } /* end test for out-of-bounds: */

          index = i * Py + j;
          Part_x[index] = x_pos;
          Part_y[index] = y_pos;

          /*
            fprintf(ofpt," sin_ displ_x = %le, sin_displ_y= %le \n "
            , sin_displ_x, sin_displ_y);
          */
        } /* end for j */

    } /* end for i */

  printf("Uniform_grid_Init: returning to caller (DataInit) ... \n \n");

  return 0;

}
/**************** end Uniform_Grid_Init *********************************/

/************************************************************************/
/************************************************************************/
/*********************  Band_Init  **************************************/
/************************************************************************/
int Band_Init(int Np, int Ny, double Lx, double Part_x[], double Part_y[])
{

  int	i, l;

  double 
    xin1,		/* position of test band 1 */
    xin2,		/* position of test band 2 */
    init_dy;	/* y-displacemetn between particles in test band */

  /* Let the particles be distributed in two vertical bands at
   * xin1 and xin2 set in input file:*/

  printf("\n Input x-coordinate of first band : ");
  scanf("%lf", &xin1);

  printf("\n Input x-coordinate of second band : ");
  scanf("%lf", &xin2);

  printf("\n Bands init. at x = % 6.4lf and x = % 6.4lf\n",xin1,xin2);

  init_dy = Lx/(Np/2); 

  printf("\n Dist. btw. each particle in band: % 9.8lf \n", init_dy);



  /* Generate lower band: */
  if (Np == (2 * Ny))

    { /* "center" particle in cell since 1 particle/cell */
      printf("Center particle in cell since 1 particle/cell\n");
	
      Part_x[0] = xin1;
      Part_y[0] = init_dy/2;
    }
  else
    { /* don't worry about centering ...*/
      Part_x[0] = xin1;
      Part_y[0] = init_dy;
    }
  for (l = 1; l<Np/2; l++)
    {
      Part_x[l] = xin1;
      Part_y[l] = Part_y[l-1] + init_dy;
    }
  /* Generate upper band: */
  l = Np/2;
  if (Np == 2*Ny)
    { /* center particle in cell if 1 particle/cell */
      Part_x[l] = xin2;
      Part_y[l] = init_dy/2;
    }
  else
    { /* don't worry about centering ...*/
      Part_x[l] = xin2;
      Part_y[l] = init_dy;
    }
	
  for (l = (Np/2)+1; l<Np; l++)
    {
      Part_x[l] = xin2;
      Part_y[l] = Part_y[l-1] + init_dy;
    }
  printf("DataInit: Particle locations:\n\n");
  for (i = 0; i< Np; i++)
    {
      if(Part_x[i] == PARTICLE_EMPTY)
      	continue;
      printf("DataInit:     %7.6f,%7.6f \n",Part_x[i], Part_y[i]);
    }
  printf(" \n \n ");

}
/****************************************************************************/
/****************************************************************************/
/************************   Trace_Oscil  ************************************/
/****************************************************************************/
/****************************************************************************/

int Trace_Oscil(int i, double loc_x1, double loc_y1,
		double t, int Nx, int Ny, double Phi[])

{

  int k,l;

  /**************************** start oscillation trace *********************/

  switch(i){
    case 0:
      temp1xc = loc_x1;
      /*
        fprintf(ofpt, "particle 1 at t= %e  : (%7.6f, %7.6f) \n", t, loc_x1, loc_y1);
      */
      if ((temp1xc < temp1xb) && (temp1xa < temp1xb))
        {
          fprintf(ofp1, "maximum at %8.7lf, %8.7lf -- t= %le\n", temp1xb, loc_y1, t);
          if (count == 0) 
            { /* trace Phi at first max */
              count = 1;
              for (k=Ny-1; k>=0; k--)
                {
                  for (l=0; l<=Nx-1; l++)
                    {
                      fprintf(ofpr, "%i %e \n",count, A0INDEX(Phi,Nx,k,l));
                      count = count+1;
                    }
                }
            }
        }
      else if  ((temp1xc > temp1xb) && (temp1xa > temp1xb))
        {
          fprintf(ofp1, "minimum at % 8.7lf, %8.7lf -- t= %le \n",temp1xb,loc_y1,t);
        } 
      temp1xa = temp1xb;
      temp1xb = temp1xc;

      temp1yc = loc_y1;
      if ((temp1yc < temp1yb) && (temp1ya < temp1yb))
        {
          fprintf(ofpy1, "maximum at %8.7lf, %8.7lf -- t= %le\n", loc_x1, temp1yb, t);
        }
      else if  ((temp1yc > temp1yb) && (temp1ya > temp1yb))
        {
          fprintf(ofpy1, "minimum at % 8.7lf, %8.7lf -- t= %le \n",loc_x1, temp1yb, t);
        } 
      temp1ya = temp1yb;
      temp1yb = temp1yc;
      break;
    case 1: 
      temp2xc = loc_x1;
      if ((temp2xc < temp2xb) && (temp2xa < temp2xb))
        {
          fprintf(ofp2, "max at % 8.7lf, %8.7lf -- t= %le \n",temp2xb,loc_y1,t);
        }
      else if  ((temp2xc > temp2xb) && (temp2xa > temp2xb))
        {
          fprintf(ofp2, "minimum at %8.7lf, %8.7lf -- t= %le \n",temp2xb,loc_y1,t);
        }
      temp2xa = temp2xb;
      temp2xb = temp2xc;
      break;
  } /* end switch */


}
/*********************************** end oscillation trace *****************/

#ifdef PIC_PETSC
int petsc_init(int Nx, int Ny, double *Rho){
	
  int i,j,err;
	
  VecCreateMPI(PETSC_COMM_WORLD,Nx, PETSC_DETERMINE, &f);
  for (i = 0; i < Nx*dim_sizes[1]; ++i) {
    VecSetValue(f, i, UPPER_DIRICHLET_BORDER, INSERT_VALUES);	
  }


  //	MatCreateMPI
  //MatCreateMPIDense(PETSC_COMM_WORLD, Ny, Nx, PETSC_DECIDE, PETSC_DECIDE, Rho,&petsc_rho);		
  /* set up sor solver as we want to, declared in declarations.h as global variable*/
  //	PCCreate(PETSC_COMM_WORLD, &petsc_pc_sor);
  //	PCSetType(petsc_pc_sor, PCBJACOBI);
  //	PCSORSetIterations(petsc_pc_sor, 3, 30 /*SOR_MAX_ITERATIONS*/); // its=25?
  //	PCSORSetOmega(petsc_pc_sor,SOR_OMEGA);
  //PCSetOperators(petsc_pc_sor, petsc_rho, petsc_rho, DIFFERENT_NONZERO_PATTERN);
  //PCSetUp(petsc_pc_sor);
  //	KSPSetUp(petsc_solver);

  /* create the solution vector */
  VecCreateMPI(PETSC_COMM_WORLD, Nx, PETSC_DECIDE, &foo_solution);

  /* create the oh great global jacobian matrix */
  MatCreateMPIAIJ(PETSC_COMM_WORLD, Ny, Nx, PETSC_DETERMINE, PETSC_DETERMINE, 0, PETSC_NULL, 0, PETSC_NULL, &petsc_rho);
	
  /* set up the SNES solver*/
  SNESCreate(PETSC_COMM_WORLD, &petsc_solver);	
  SNESSetFunction(petsc_solver, foo_solution, form_function, PETSC_NULL);
  SNESSetJacobian(petsc_solver, petsc_rho, petsc_rho, SNESDefaultComputeJacobian /*form_jacobian*/, PETSC_NULL);
  SNESSetFromOptions(petsc_solver);
}
#endif /*ifdef PIC_PETSC*/
