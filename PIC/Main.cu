#include "Particles.cuh"
#include <time.h>
#include <Windows.h>
#define DEBUG DEBUG_NONE
//#define FFT_SOLVER
//#define TRACE

#define iterations1 0
#define grid1 1
#define grid2 2
#define particles1 3
#define particles2 4
#define grid3a 5
#define grid3b 6
#define iterations2 7
#define kernel 8
#define solver 9

#define kernel1 1
#define kernel2 2
#define kernel3 3

#ifdef TRACE
#define test -1
#else

//#define test iterations1
//#define test iterations2
//#define test grid1
//#define test grid2
//#define test grid3a
//#define test grid3b
//#define test particles1
//#define test particles2
//#define test kernel
#define test solver

//#define kerneltest kernel1
//#define kerneltest kernel2
//#define kerneltest kernel3
#endif
#if test != kernel && kerneltest != -1
#define kerneltest -1
#endif

#ifdef TRACE
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\trace.exe\"")
#elif test == iterations1
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\iterations1.exe\"")

#elif test == iterations2
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\iterations2.exe\"")

#elif test == grid1
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\grid1.exe\"")

#elif test == grid2
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\grid2.exe\"")

#elif test == grid3a
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\grid3a.exe\"")

#elif test == grid3b
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\grid3b.exe\"")

#elif test == particles1
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\particles1.exe\"")

#elif test == particles2
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\particles2.exe\"")

#elif test == kernel
#if kerneltest == kernel1
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\kernel1.exe\"")

#elif kerneltest == kernel2
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\kernel2.exe\"")

#elif kerneltest == kernel3
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\kernel3.exe\"")

#endif
#elif test == solver
#ifdef FFT_SOLVER
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\fft.exe\"")

#else
#pragma comment(linker, "/OUT:\"E:\\Git\\PIC\\Release\\sor.exe\"")

#endif
#endif







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
	//====//
#if DEBUG != DEBUG_NONE
	std::wcout << "\n" << cfg.charge_by_mass << "\n\n";
	double
		hx = cfg.l.x / cfg.n.x,
		hy = cfg.l.y / cfg.n.y,
		hz = cfg.l.z / cfg.n.z;
	int
		i = (int)(initParticles[0].position.x / hx),
		j = (int)(initParticles[0].position.y / hy),
		k = (int)(initParticles[0].position.z / hz);
	double
		a = initParticles[0].position.x - hx * i,
		b = initParticles[0].position.y - hy * j,
		c = initParticles[0].position.z - hz * k;

	std::cout << "\nhx\t" << hx;
	std::cout << "\nhy\t" << hy;
	std::cout << "\nhz\t" << hz;

	std::cout << "\ni\t" << i;
	std::cout << "\nj\t" << j;
	std::cout << "\nk\t" << k;

	std::cout << "\na\t" << a;
	std::cout << "\nb\t" << b;
	std::cout << "\nc\t" << c;

	std::cout << "\np_x\t" << initParticles[0].position.x << "\t" << i*hx + a;
	std::cout << "\np_y\t" << initParticles[0].position.y << "\t" << j*hy + b;
	std::cout << "\np_z\t" << initParticles[0].position.z << "\t" << k*hz + c;
	
	std::cout << "\n\n";
	
	std::cout <<  (hz - c) * (hy - b)  * (hx - a) * cfg.rho_k << "\n\n";
	std::cout <<  (hz - c) *		b  * (hx - a) * cfg.rho_k << "\n\n"; 
	std::cout <<  (hz - c) * (hy - b)  *	   a  * cfg.rho_k << "\n\n";
	std::cout <<  (hz - c) *		b  *	   a  * cfg.rho_k << "\n\n";
	
	std::cout << 		 c	* (hy - b) * (hx - a) * cfg.rho_k << "\n\n";
	std::cout << 		 c	*		b  * (hx - a) * cfg.rho_k << "\n\n"; 
	std::cout << 		 c	* (hy - b) *	   a  * cfg.rho_k << "\n\n";
	std::cout << 		 c	*		b  *	   a  * cfg.rho_k << "\n\n";
#endif
	//====//

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
	
#if DEBUG == DEBUG_FREQ
	//Debug data Matrices
	cufftDoubleComplex *h_debugFreq = (cufftDoubleComplex*) malloc(d_freq.xsize * d_freq.ysize);
	memset(h_debugFreq, -1, d_freq.xsize * d_freq.ysize);
#elif DEBUG
	void *h_debug = malloc(d_E.xsize * d_E.ysize);
	memset(h_debug, -1, d_E.xsize * d_E.ysize);
#endif
	///...
	//Particle Tracking Array:
	//Stores the values for each particle every iteration.
	//TODO: Decide between cost of storing in GPU global memory,
	//versus bandwith cost of running cudaMemCpy every iteration.
	cudaChk( cudaMemcpy(d_particles, (Particle *)initParticles, cfg.particles*sizeof(Particle), cudaMemcpyHostToDevice));
#ifdef TRACE
	char* particleTrackingArray = (char *)malloc(cfg.particles*sizeof(Particle)*(1+cfg.iterations/cfg.trace_interval));
	memcpy(particleTrackingArray, initParticles, cfg.particles*sizeof(Particle));
#endif
	free(initParticles);
	//std::cout << "Setup completed.\n\n";
/**** Main loop ****/
	//std::cout << "Beginning main loop...\n";
	cudaExtent
		cd_ext =	make_cudaExtent(d_Rho.pitch,	cfg.n.y, cfg.n.z),
		pot_ext =	make_cudaExtent(d_Phi.pitch,	cfg.n.y, cfg.n.z),
		E_ext =		make_cudaExtent(d_E.pitch,		cfg.n.y, cfg.n.z),
		freq_ext =	make_cudaExtent(d_freq.pitch,	cfg.n.y, cfg.n.z);
	//int update = 1 + (cfg.iterations-1)/100;
	//float percent = 100.0 / cfg.iterations;
#if test != iterations1 && test != kernel
	_int64 t1;
	StartTimer(&t1);
#if test == solver
	double t;
#endif
#elif test == kernel
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float t = 0.f;
#endif
	for (int i = 1; i <= cfg.iterations; i++) {
		//if(i%update==0)
			//std::cout << "Iteration " << i << " of " << cfg.iterations << " - " << i * percent << "%\n";
		//Reset all values to zero.
		cudaChk(cudaMemset3D(d_Rho, 0, cd_ext));
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		cudaChk(cudaMemset3D(d_Phi, 0, pot_ext));
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		cudaChk(cudaMemset3D(d_E,	0, E_ext));
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		cudaChk(cudaMemset3D(d_freq, 0, freq_ext));
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
#if kerneltest == kernel1
		t = 0.f;
		cudaEventRecord(start);
#endif
		determineChargesFromParticles3D<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, d_Rho, cfg);
#if kerneltest == kernel1
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&t, start, stop);
#endif
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());

#if test == solver
		_int64 t1;
		StartTimer(&t1);
#endif
#ifdef FFT_SOLVER
		cufftChk( cufftExecD2Z(plan, (cufftDoubleReal *) d_Rho.ptr, (cufftDoubleComplex *) d_freq.ptr));
		solve3D<<<cfg.exec_cfg.nbfreq, cfg.exec_cfg.tbfreq>>>(d_freq, cfg);
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		cufftChk( cufftExecZ2D(iplan, (cufftDoubleComplex *) d_freq.ptr, (cufftDoubleReal *) d_Phi.ptr));
#else
		initSOR<<<cfg.exec_cfg.nbg, cfg.exec_cfg.tbg>>>(d_Rho, d_Phi, cfg);//Phi = rho * h2 / eps0
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
		for (int k=0; k < cfg.sor_iterations; k++) {
			SOR3D<<<cfg.exec_cfg.nbsor, cfg.exec_cfg.tbg>>>(d_Phi, cfg, 0);// Red
			cudaChk( cudaPeekAtLastError());
			cudaChk( cudaDeviceSynchronize());
			SOR3D<<<cfg.exec_cfg.nbsor, cfg.exec_cfg.tbg>>>(d_Phi, cfg, 1);// Black
			cudaChk( cudaPeekAtLastError());
			cudaChk( cudaDeviceSynchronize());
		}
#endif
#if test == solver
		t = StopTimer(t1);
#endif
#if kerneltest == kernel2
		t = 0.f;
		cudaEventRecord(start);
#endif
		electricFieldFromPotential3D<<<cfg.exec_cfg.nbg, cfg.exec_cfg.tbg>>>(d_Phi, d_E, cfg);
#if kerneltest == kernel2
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&t, start, stop);
#endif
		cudaChk( cudaPeekAtLastError());
		cudaChk( cudaDeviceSynchronize());
#if kerneltest == kernel3
		t = 0.f;
		cudaEventRecord(start);
#endif
		updateParticles3D<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, d_E, cfg.ts, cfg);
#if kerneltest == kernel3
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&t, start, stop);
#endif
		cudaChk(cudaPeekAtLastError());
		cudaChk(cudaDeviceSynchronize());

		//Store values

#ifdef TRACE
		if(i%cfg.trace_interval==0)
			cudaChk( cudaMemcpy((void*)(particleTrackingArray + i/cfg.trace_interval*cfg.particles*sizeof(Particle)), d_particles, cfg.particles * sizeof(Particle), cudaMemcpyDeviceToHost));
#endif
		
		//generateParticles<<<cfg.exec_cfg.nbp, cfg.exec_cfg.tbp>>>(d_particles, cfg);
		//debug
#if DEBUG == DEBUG_CHARGE
		debug(h_debug, d_Rho, 1, i, cfg);
#elif DEBUG == DEBUG_POTENTIAL
		debug(h_debug, d_Phi, 1, i, cfg);
#elif DEBUG == DEBUG_EFIELD
		debug(h_debug, d_E, 3, i, cfg);
#elif DEBUG == DEBUG_FREQ
		debug(h_debugFreq, d_freq, 2, i, cfg);
#endif

	}
#if test != iterations1 && test != solver && test != kernel
	double t = StopTimer(t1);
#endif
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
#if test != iterations1
	return t;
#else
	return 0;
#endif
}


int main(int argc, const char* argv[]){
	Config cfg = getConfig();
#ifndef TRACE
	cfg.iterations = 1;
#endif
	pic(cfg);
	cudaChk(cudaDeviceSynchronize());
#ifndef TRACE
#if test == iterations1 || test == iterations2
#define length 32
#define lengthinner 1

#elif test == grid1 || test == solver
#define length 256
#define lengthinner 1

#elif test == grid2
#define length 63
#define lengthinner 256

#elif test == grid3a || test == grid3b
#define length 128
#define lengthinner 1

#elif test == particles1
#define length 1
#define lengthinner 256

#elif test == particles2 || test == kernel
#define length 256
#define lengthinner 256
#endif

#define size length*lengthinner

	double* timing = (double*)malloc(sizeof(double) * size);
	int* param1 = (int*)malloc(sizeof(int) * size);
	int* param2 = (int*)malloc(sizeof(int) * size);


#if test == solver
	cfg.iterations = 1;
#endif
	_int64 t1;
	std::cout << 0 << " out of " << length << " values timed...\n";
	for(int i=0; i<length; i++){
		for (int j=0; j<lengthinner; j++){
#if test == iterations1 || test == iterations2
			cfg.iterations = i * i;
#endif
#if test == particles1 || test == particles2 || test == kernel
			cfg.particles = (j + 1) * (j + 1);
#endif
#if test == grid1 || test == particles2 || test == solver || test == kernel
			cfg.n.x = i + 3;
			cfg.n.y = cfg.n.x;
			cfg.n.z = cfg.n.x;
			cfg.n.w = cfg.n.z / 2 + 1;
#elif test == grid2
			cfg.n.x = 4*i + 4;
			cfg.n.y = cfg.n.x;
			cfg.n.z = j + 3;
			cfg.n.w = cfg.n.z / 2 + 1;

#elif test == grid3a
			cfg.n.x = i * 2 + 4;
			cfg.n.y = cfg.n.x;
			cfg.n.z = cfg.n.x;
			cfg.n.w = cfg.n.z / 2 + 1;

#elif test == grid3b
			cfg.n.x = i * 2 + 3;
			cfg.n.y = cfg.n.x;
			cfg.n.z = cfg.n.x;
			cfg.n.w = cfg.n.z / 2 + 1;
#endif
			StartTimer(&t1);
			double result = 0.0;
			result = pic(cfg);
#if test == iterations1
			timing[i*lengthinner + j] = StopTimer(t1);
#else
			timing[i*lengthinner + j] = result;
#endif
#if test == particles1
			param2[i*lengthinner + j] = cfg.particles;
#endif
#if test == grid1 || test == grid2 || test == grid3a || test == grid3b || test == particles2 || test == solver || test == kernel
			param1[i*lengthinner + j] = cfg.n.x;
#if test == particles2 || test == kernel
			param2[i*lengthinner + j] = cfg.particles;
#else
			param2[i*lengthinner + j] = cfg.n.z;
#endif
#endif
#if test == iterations1 || test == iterations2
			param1[i * lengthinner + j] = cfg.iterations;
#endif
#if lengthinner != 1
			//std::cout << j + 1 << " out of " << lengthinner << " values timed...\n";
			cudaChk(cudaDeviceSynchronize());
#endif
		}
#if length != 1
		std::cout << i+1 << " out of " << length << " values timed...\t" << i+1 << "\n\n";
#endif
	}
	//printTimingdata(timing, param1, param2, true, length*lengthj, "grid2");
	//printTimingdata(timing, param1, param2, true, length/2*lengthj, "particle2");
#if test == iterations1
	printTimingdata(timing, param1, param2, false, size, "iterations1");
#elif test == iterations2
	printTimingdata(timing, param1, param2, false, size, "iterations2");
#elif test == grid1
	printTimingdata(timing, param1, param2, false, size, "grid1");
#elif test == grid2
	printTimingdata(timing, param1, param2, true, size, "grid2");
#elif test == grid3a
	printTimingdata(timing, param1, param2, false, size, "grid3a");
#elif test == grid3b
	printTimingdata(timing, param1, param2, false, size, "grid3b");
#elif test == particles1
	printTimingdata(timing, param2, param1, false, size, "particles1");
#elif test == particles2
	printTimingdata(timing, param2, param1, true, size, "particles2");
#elif test == solver
#ifdef FFT_SOLVER
	printTimingdata(timing, param1, param2, false, size, "fft");
#else
	printTimingdata(timing, param1, param2, false, size, "sor");
#endif
#elif test == kernel
#if kerneltest == kernel1
	printTimingdata(timing, param1, param2, true, size, "distributecharge");
#endif
#if kerneltest == kernel2
	printTimingdata(timing, param1, param2, true, size, "electricfield");
#endif
#if kerneltest == kernel3
	printTimingdata(timing, param1, param2, true, size, "updateparticles");
#endif
#endif
	cudaDeviceReset();
	free(timing);
	free(param1);
	free(param2);
#endif
	return 0;
}
