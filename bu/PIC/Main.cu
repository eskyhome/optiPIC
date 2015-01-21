#include "Particles.cuh"
#include <time.h>


int main() {
	Config cfg = getConfig();
	srand(time(NULL));
	Particle* initParticles = (Particle *)malloc(cfg.particles*sizeof(Particle));
	for (int i = 0; i<cfg.particles; i++){
		initParticles[i] = randParticle(cfg);
	};
		
	std::cout << "Starting execution...\n";

	Particle *d_particles;
	cudaPitchedPtr d_Rho, d_Phi, d_E, d_freq;
	allocateMemory3D(&d_particles, &d_Rho, &d_Phi, &d_E, &d_freq, cfg);
	std::cout << "Attempting to set up cuFFT plan...\n";

	// =-=-= Start cuFFT planning =-=-= //
	cufftHandle plan, iplan;
	cufftChk( cufftCreate(&plan));
	cufftChk( cufftCreate(&iplan));
	// =-=-= Advanced Data Layout =-=-= //
	int rank = 3,
		n[3] = {cfg.n.z, cfg.n.y, cfg.n.x},
		dist =  1,
		stride = 1,
		batch = 1,
		rhonembed[3] = {cfg.n.z, cfg.n.y, d_Rho.pitch/sizeof(double)},
		phinembed[3] = {cfg.n.z, cfg.n.y, d_Phi.pitch/sizeof(double)},
		freqnembed[3] = {cfg.n.w, cfg.n.y, d_freq.pitch/sizeof(cufftDoubleComplex)};
	// =-=-= =-=-= =-=  =-= =-=-= =-=-= //
	cufftChk( cufftPlanMany(&plan, rank, n, rhonembed, stride, dist, freqnembed, stride, dist, CUFFT_D2Z, batch));
	cufftChk( cufftPlanMany(&iplan, rank, n, freqnembed, stride, dist, phinembed, stride, dist, CUFFT_Z2D, batch));
	// =-=-=  End cuFFT planning  =-=-= //

	//Storage of complex frequency data
	std::cout << "Allocating space for frequency matrix\n";

	//debugging...
#if defined(DEBUGCHARGE) || defined(DEBUGPOTENTIAL)
	void *h_debug = malloc(d_E.xsize * d_E.ysize);
	memset(h_debug, -1, d_E.xsize * d_E.ysize);
#endif
#ifdef DEBUGFREQ
	cufftDoubleComplex *h_debugFreq = (cufftDoubleComplex*) malloc(d_freq.xsize * d_freq.ysize);
	memset(h_debugFreq, -1, d_freq.xsize * d_freq.ysize);
#endif
	///...
	//Particle Tracking Array:
	//Stores the values for each particle every iteration.
	//TODO: Decide between cost of storing in GPU global memory,
	//versus bandwith cost of running cudaMemCpy every iteration.
	cudaChk( cudaMemcpy(d_particles, (Particle *)initParticles, cfg.particles*sizeof(Particle), cudaMemcpyHostToDevice));
	char* particleTrackingArray = (char *)malloc(cfg.particles*sizeof(Particle)*(cfg.iterations+1));

	std::cout << "Attempting to copy particle data into time 0 of tracking array...\n";
	memcpy(particleTrackingArray, initParticles, cfg.particles*sizeof(Particle));
/**** Main loop ****/
	std::cout << "Beginning main loop:\n";
	for (int i = 1; i <= cfg.iterations; i++) {
		//Reset all values to zero.
		cudaChk(cudaMemset2D(d_Rho.ptr, d_Rho.pitch, 0, d_Rho.xsize, d_Rho.ysize));
		cudaChk(cudaMemset2D(d_Phi.ptr, d_Phi.pitch, 0, d_Phi.xsize, d_Phi.ysize));
		cudaChk(cudaMemset2D(d_E.ptr, d_E.pitch, 0, d_E.xsize, d_E.ysize));
		cudaChk(cudaMemset2D(d_freq.ptr, d_freq.pitch, 0, d_freq.xsize, d_freq.ysize));
		
		determineChargesFromParticles3D<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, d_Rho, cfg);
		cudaChk( cudaDeviceSynchronize());
#ifdef FFT_SOLVER
		cufftChk( cufftExecD2Z(plan, (cufftDoubleReal *) d_Rho.ptr, (cufftDoubleComplex *) d_freq.ptr));	
		solve3D<<<cfg.nbfreq, cfg.tbfreq>>>(d_freq, cfg);
		cufftChk( cufftExecZ2D(iplan, (cufftDoubleComplex *) d_freq.ptr, (cufftDoubleReal *) d_Phi.ptr));
#else//SOR_SOLVER
		cudaPitchedPtr temp;
		for (int k=0; k < 128; k++) {
			SOR3D<<<cfg.exec_cfg.nbf, cfg.exec_cfg.tbf>>>(d_Rho, d_Phi, cfg);
			temp = d_Rho;
			d_Rho = d_Phi;
			d_Phi = temp;
		}
		temp = d_Rho;
		d_Rho = d_Phi;
		d_Phi = temp;
#endif
		//scaleData<<<numBField, tPBField>>>(d_Phi);
		electricFieldFromPotential3D<<<cfg.exec_cfg.nbf, cfg.exec_cfg.tbf>>>(d_Phi, d_E, cfg);
		cudaChk( cudaDeviceSynchronize());
		updateParticles3D<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, d_E, cfg.ts, cfg);
		cudaChk( cudaDeviceSynchronize());
		cudaChk( cudaGetLastError());

		//Store values
		cudaChk( cudaMemcpy((void*)(particleTrackingArray + i*cfg.particles*sizeof(Particle)), d_particles, cfg.particles * sizeof(Particle), cudaMemcpyDeviceToHost));
		//debug
#ifdef DEBUGCHARGE
		cudaChk( cudaMemcpy2D(h_debug, d_Rho.xsize, d_Rho.ptr, d_Rho.pitch, d_Rho.xsize, d_Rho.ysize, cudaMemcpyDeviceToHost));
		printField("debug", (double *)h_debug, i);
		cudaChk( cudaDeviceSynchronize());
		cudaChk( cudaGetLastError());
#else
	#ifdef DEBUGPOTENTIAL
			cudaChk( cudaMemcpy2D(h_debug, d_Phi.xsize, d_Phi.ptr, d_Phi.pitch, d_Phi.xsize, d_Phi.ysize, cudaMemcpyDeviceToHost));
			printField("debug", (double *)h_debug, i);
			cudaChk( cudaDeviceSynchronize());
			cudaChk( cudaGetLastError());
	#endif
#endif
#ifdef DEBUGFREQ
		cudaChk( cudaMemcpy2D(h_debugFreq, d_freq.xsize, d_freq.ptr, d_freq.pitch, d_freq.xsize, d_freq.ysize, cudaMemcpyDeviceToHost));
		printFreq("debugFreq", h_debugFreq, i);
		cudaChk( cudaDeviceSynchronize());
		cudaChk( cudaGetLastError());
#endif
	}
	printStatistics("particles.xml", (Particle*)particleTrackingArray, cfg);
	cleanData(d_particles, d_Rho, d_Phi, d_E, d_freq);
	cufftDestroy(plan);
	cufftDestroy(iplan);
	
	/*=--=*/

	return 0;
}