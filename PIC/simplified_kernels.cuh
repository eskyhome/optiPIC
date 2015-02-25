#include "dependencies.cuh"
#include "particle.cuh"
#include "structs.cuh"

__global__ void fftsolver_kernel(cufftDoubleComplex* freq, Config cfg);
__device__ void fftsolver_device(cufftDoubleComplex* freq, Config cfg);
__global__ void sorsolver_kernel(double* phi, Config cfg, size_t flag);
__global__ void sorinit_kernel(double* phi, double* rho, Config cfg);
__global__ void electricfield_kernel(double* phi, double4* E, Config cfg);
__global__ void particleUpdate_kernel(Particle* particles, double4* E, double* rho, Config cfg);