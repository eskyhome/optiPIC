#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cufft.h>
#include <math.h>

#include <iostream>
//#include "Constants.cuh"

//====//
#define cudaChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stdout, "CUDA error: %s in file %s, line %d\n", cudaGetErrorString(code), file, line);
	}
}
#define cufftChk(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult_t code, const char *file, int line, bool abort=true) {
	if (code != CUFFT_SUCCESS) {
		fprintf(stdout, "CUFFT error: %d in file %s, line %d\n", code, file, line);
	}
}
//====//

struct Particle {
	double3 position;
	double3 velocity;
	double3 eAtP;
	double3 padding;
};

struct Config{
	int4 n;	// Number of elements in each direction.
	double3 l;		// Metric length in each direction.
	double
		rho_k,
		charge_by_mass,
		solve_factor,
		ts;
	
	double
		omega,
		threshold;
	int
		iterations,
		particles;
	struct {
		dim3
			tbp,
			nbp,
			tbf,
			nbf,
			tbfreq,
			nbfreq;
	} exec_cfg;
};

inline double randDouble(double min, double max) {
	double r = ((double)rand())/RAND_MAX;
	return min + r * (max - min);
}
inline Particle randParticle(Config cfg) {
	Particle p = {
		make_double3(randDouble(0, cfg.l.x), randDouble(0, cfg.l.y), randDouble(0, cfg.l.z)),
		//make_double2(randDouble(-HX, HX), randDouble(-HY, HY)),
		make_double3(0, 0, 0),
		make_double3(0, 0, 0)
	};
	return p;
}


//====//

__global__ void determineChargesFromParticles3D(Particle *particles, cudaPitchedPtr chargeDensity, Config cfg);

__global__ void electricFieldFromPotential3D(cudaPitchedPtr potential, cudaPitchedPtr E, Config cfg);

__global__ void updateParticles3D(Particle *particles, cudaPitchedPtr E, double timeStep, Config cfg);

__global__ void solve3D(cudaPitchedPtr freq, Config cfg);

__global__ void SOR3D(cudaPitchedPtr in, cudaPitchedPtr out, Config cfg);

void allocateMemory3D(Particle **particles, cudaPitchedPtr *chargeDensity, cudaPitchedPtr *potential, cudaPitchedPtr *E, cudaPitchedPtr *freq, Config cfg);

void cleanData(Particle *particles, cudaPitchedPtr chargeDensity, cudaPitchedPtr potential, cudaPitchedPtr E, cudaPitchedPtr freq);

void printField(char *fieldname, double *field, int i);

void printFreq(char *fieldname, cufftDoubleComplex *field, int i);

void printStatistics(char *filename, Particle* pta, Config cfg);

Config getConfig();