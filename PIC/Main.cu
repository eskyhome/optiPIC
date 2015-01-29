#include "Particles.cuh"
#include <time.h>
#include <Windows.h>
#define FFT_SOLVER
//#define TRACE


bool setupCufft(cufftHandle *plan, cufftHandle *iplan, Config cfg, int rhoElements, int phiElements, int freqElements){
	// =-=-= Start cuFFT planning =-=-= //
	cufftChk( cufftCreate(plan));
	cufftChk( cufftCreate(iplan));
	// =-=-= Advanced Data Layout =-=-= //
	int rank = 3,
		n[3] = {cfg.n.z, cfg.n.y, cfg.n.x},
		dist =  1,
		stride = 1,
		batch = 1,
		rhonembed[3] = {cfg.n.z, cfg.n.y, rhoElements},
		phinembed[3] = {cfg.n.z, cfg.n.y, phiElements},
		freqnembed[3] = {cfg.n.w, cfg.n.y, freqElements};
	// =-=-= =-=-= =-=  =-= =-=-= =-=-= //
	cufftResult_t res = cufftPlanMany(plan, rank, n, rhonembed, stride, dist, freqnembed, stride, dist, CUFFT_D2Z, batch);
	cufftResult_t ires = cufftPlanMany(iplan, rank, n, freqnembed, stride, dist, phinembed, stride, dist, CUFFT_Z2D, batch);
	// =-=-=  End cuFFT planning  =-=-= //
	if (res != CUFFT_SUCCESS || ires != CUFFT_SUCCESS){
		cufftChk(res);
		cufftChk(ires);
		return false;
	}
	return true;
}

//Performance testing
void StartTimer(_int64 *pt1){
	QueryPerformanceCounter((LARGE_INTEGER*)pt1);
}
double StopTimer( _int64 t1){
	_int64 t2, ldFreq;

	QueryPerformanceCounter( (LARGE_INTEGER*)&t2 );
	QueryPerformanceFrequency( (LARGE_INTEGER*)&ldFreq );
	return ((double)(t2 - t1) / (double)ldFreq) * 1000.0;
}

double pic(Config cfg) {
	//std::cout << "\nBeginning setup...\n";

	srand(time(NULL));
	Particle* initParticles = (Particle *)malloc(cfg.particles*sizeof(Particle));
	if (initParticles == NULL){
		std::cout << "Failed to allocate memory for particles...\n\n";
		return 0;
	}
	for (int i = 0; i<cfg.particles; i++){
		initParticles[i] = randParticle(cfg);
	};

	Particle *d_particles;
	cudaPitchedPtr d_Rho, d_Phi, d_E, d_freq;
	allocateMemory3D(&d_particles, &d_Rho, &d_Phi, &d_E, &d_freq, cfg);
	
#ifdef FFT_SOLVER
	cufftHandle plan, iplan;
	bool success = setupCufft(&plan, &iplan, cfg, d_Rho.pitch/sizeof(double), d_Phi.pitch/sizeof(double), d_freq.pitch/sizeof(cufftDoubleComplex));
	if (!success){
		std::cout << "Failed to create cufft plan...\n";
		cleanData(d_particles, d_Rho, d_Phi, d_E, d_freq);
		cufftDestroy(plan);
		cufftDestroy(iplan);
		std::cout << "Cleanup complete...\n\n";
		return 0;
	}
#endif

	cudaChk( cudaMemcpy(d_particles, (Particle *)initParticles, cfg.particles*sizeof(Particle), cudaMemcpyHostToDevice));

#ifdef TRACE
	char* particleTrackingArray = (char *)malloc(cfg.particles*sizeof(Particle)*(1+cfg.iterations/cfg.trace_interval));
	memcpy(particleTrackingArray, initParticles, cfg.particles*sizeof(Particle));
#endif
	free(initParticles);

	cudaExtent
		cd_ext =	make_cudaExtent(d_Rho.pitch,	cfg.n.y, cfg.n.z),
		pot_ext =	make_cudaExtent(d_Phi.pitch,	cfg.n.y, cfg.n.z),
		E_ext =		make_cudaExtent(d_E.pitch,		cfg.n.y, cfg.n.z),
		freq_ext =	make_cudaExtent(d_freq.pitch,	cfg.n.y, cfg.n.z);

	for (int i = 1; i <= cfg.iterations; i++) {
		//Reset all values to zero.
		cudaChk(cudaMemset3D(d_Rho, 0, cd_ext));
		cudaChk(cudaMemset3D(d_Phi, 0, pot_ext));
		cudaChk(cudaMemset3D(d_E,	0, E_ext));
		cudaChk(cudaMemset3D(d_freq, 0, freq_ext));

		determineChargesFromParticles<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, d_Rho, cfg);
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());

#ifdef FFT_SOLVER
		cufftChk( cufftExecD2Z(plan, (cufftDoubleReal *) d_Rho.ptr, (cufftDoubleComplex *) d_freq.ptr));
		solve<<<cfg.exec_cfg.nbfreq, cfg.exec_cfg.tbfreq>>>(d_freq, cfg);
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		cufftChk( cufftExecZ2D(iplan, (cufftDoubleComplex *) d_freq.ptr, (cufftDoubleReal *) d_Phi.ptr));
#else
		initSOR<<<cfg.exec_cfg.nbg, cfg.exec_cfg.tbg>>>(d_Rho, d_Phi, cfg);//Phi = rho * h2 / eps0
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		for (int k=0; k < cfg.sor_iterations; k++) {
			SOR<<<cfg.exec_cfg.nbsor, cfg.exec_cfg.tbg>>>(d_Phi, cfg, 0);// Red
			cudaChk( cudaPeekAtLastError());
			cudaChk( cudaDeviceSynchronize());
			SOR<<<cfg.exec_cfg.nbsor, cfg.exec_cfg.tbg>>>(d_Phi, cfg, 1);// Black
			cudaChk( cudaPeekAtLastError());
			cudaChk( cudaDeviceSynchronize());
		}
#endif

		electricFieldFromPotential<<<cfg.exec_cfg.nbg, cfg.exec_cfg.tbg>>>(d_Phi, d_E, cfg);

		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());

		updateParticles<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, d_E, cfg.ts, cfg);

		cudaChk(cudaPeekAtLastError());
		cudaChk(cudaDeviceSynchronize());

		//Store values

#ifdef TRACE
		if(i%cfg.trace_interval==0)
			cudaChk( cudaMemcpy((void*)(particleTrackingArray + i/cfg.trace_interval*cfg.particles*sizeof(Particle)), d_particles, cfg.particles * sizeof(Particle), cudaMemcpyDeviceToHost));
#endif
		
		//generateParticles<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, cfg);
	}

#ifdef TRACE
#ifdef FFT_SOLVER
	printTracefile("fft", (Particle*)particleTrackingArray, cfg);
#else
	printTracefile("sor", (Particle*)particleTrackingArray, cfg);
#endif
#endif
	cleanData(d_particles, d_Rho, d_Phi, d_E, d_freq);
#ifdef FFT_SOLVER
	cufftDestroy(plan);
	cufftDestroy(iplan);
#endif
	//std::cout << "Completed.\n";
	/*=--=*/
return 0;
}


int main(int argc, const char* argv[]){
	Config cfg = getConfig();
#ifndef TRACE
	cfg.iterations = 1;
#endif
	pic(cfg);
	cudaChk(cudaDeviceSynchronize());
	return 0;
}
