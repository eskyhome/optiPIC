#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cufft.h>
#include <math.h>

#include <iostream>
#define DEBUG_NONE 0
#define DEBUG_CHARGE 1
#define DEBUG_POTENTIAL 2
#define DEBUG_EFIELD 3
#define DEBUG_FREQ 4

//====//
#define cudaChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stdout, "CUDA error: %s in file %s, line %d\n", cudaGetErrorString(code), file, line);
	}
}

static char *mkString[9] =
{
	"CUFFT_SUCCESS",
	"CUFFT_INVALID_PLAN",
	"CUFFT_ALLOC_FAILED",
	"CUFFT_INVALID_TYPE",
	"CUFFT_INVALID_VALUE",
	"CUFFT_INTERNAL_ERROR",
	"CUFFT_EXEC_FAILED",
	"CUFFT_SETUP_FAILED",
	"CUFFT_INVALID_SIZE"
};
#define cufftChk(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult_t code, const char *file, int line, bool abort=true) {
	if (code != CUFFT_SUCCESS) {
		fprintf(stdout, "CUFFT error: %s in file %s, line %d\n", mkString[code], file, line);
	}
}

#define errCheck(fun) {kernelErrCheck(__FILE__, __LINE__)}
inline void kernelErrCheck(const char *f, int l){
cudaError_t err = cudaPeekAtLastError();
cudaAssert(err, f, l);
err = cudaDeviceSynchronize();
cudaAssert(err, f, l);
}
//====//


struct Particle {
	double3 position;
	double3 velocity;
	//double2 padding;
};

struct Config{
	int4 n;	// Number of elements in each direction.
	double3 l;		// Metric length in each direction.
	double
		rho_k,
		charge_by_mass,
		epsilon,
		ts,
		drag;
	struct {
		double
			kxt,
			kyt,
			kzt,
			constant_factor;
	} solve;
	
	double
		omega,
		threshold;
	int
		sor_iterations,
		iterations,
		trace_interval,
		particles;
	struct {
		dim3
			tbp,
			nbp,
			tbg,
			nbg,
			tbfreq,
			nbfreq,
			nbsor;
	} exec_cfg;
};

inline double randDouble(double min, double max) {
	double r = ((double)rand())/RAND_MAX;
	return min + r * (max - min);
}
inline Particle randParticle(Config cfg) {
	double
		max_radius = cfg.l.x / 16,
		radius = randDouble(-max_radius, max_radius),
		x = randDouble(-1.0, 1.0) * randDouble(-1.0, 1.0) * radius,
		rx = std::sqrt(radius*radius - x*x),

		y = randDouble(-1.0, 1.0) * randDouble(-1.0, 1.0) * rx,
		z = std::sqrt(rx*rx - y*y);
	if(radius<0.0)
		z *= -1.0;
	Particle p = {
		make_double3(
			cfg.l.x/2 + x,
			cfg.l.y/2 + y,
			cfg.l.z/2 + z
		),
		//make_double2(randDouble(-HX, HX), randDouble(-HY, HY)),
		make_double3(0/*cfg.l.x/(500 * cfg.ts)*/, 0, 0)
	};
	return p;
}

__global__ void generateParticles(Particle *particles, Config cfg);

__global__ void determineChargesFromParticles(Particle *particles, cudaPitchedPtr chargeDensity, Config cfg);

__global__ void electricFieldFromPotential(cudaPitchedPtr potential, cudaPitchedPtr E, Config cfg);

__global__ void updateParticles(Particle *particles, cudaPitchedPtr E, double timeStep, Config cfg);

__global__ void solve(cudaPitchedPtr freq, Config cfg);

__global__ void initSOR(cudaPitchedPtr Rho, cudaPitchedPtr Phi, Config cfg);
__global__ void SOR(cudaPitchedPtr Phi, Config cfg, int flag);

void allocateMemory3D(Particle **particles, cudaPitchedPtr *chargeDensity, cudaPitchedPtr *potential, cudaPitchedPtr *E, cudaPitchedPtr *freq, Config cfg);

void cleanData(Particle *particles, cudaPitchedPtr chargeDensity, cudaPitchedPtr potential, cudaPitchedPtr E, cudaPitchedPtr freq);

void printTracefile(char *filename, Particle* pta, Config cfg);

void printTimingdata(double* timing, int* param1, int* param2, bool two_dim_plot, int length, char* testid);

Config getConfig();

//====//
void debug(void* host, cudaPitchedPtr device, char val, int iteration, Config cfg);
//====//