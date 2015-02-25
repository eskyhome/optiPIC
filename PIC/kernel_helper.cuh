#include "dependencies.cuh"

#ifndef PIC_KERNEL_HELPERS_DEFINED
#define PIC_KERNEL_HELPERS_DEFINED

//Atomic add for doubles
__device__  void atomicAdd(double *address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
#if __CUDA_ARCH__ >= 110
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
#endif
	} while (assumed != old);
}

//Add two vectors
__device__ inline double3 add_v(double3 a, double3 b){
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
//Add a scalar value to each of the vector's elements
__device__ inline double3 add_s(double3 a, double b){
	return make_double3(a.x + b, a.y + b, a.z + b);
}
//Multiply vector elements by elements from another vector
__device__ inline double3 mul_v(double3 a, double3 b){
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
//Multiply vector elements by a scalar
__device__ inline double3 mul_s(double3 a, double b){
	return make_double3(a.x * b, a.y * b, a.z * b);
}
//Divide vector elements by a scalar
__device__ inline double3 div_s(double3 a, double b) {
	return make_double3(a.x / b, a.y / b, a.z / b);
}
//Divide vector elements by the elements of another vector
__device__ inline double3 div_v(double3 a, int4 b) {
	return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
//Returns the product of a vectors elements
__device__ inline double vol(double3 size) {
	return size.x * size.y * size.z;
}


//Determine distances and weights for a particle's neighbour vertices
__device__ void pCfg(Particle p, Config cfg, double3 h, ParticleCfg* pcfg) {
	size_t
		i = (size_t)(p.position.x / h.x),
		j = (size_t)(p.position.y / h.y),
		k = (size_t)(p.position.z / h.z);
	double
		a = p.position.x - h.x*i,
		b = p.position.y - h.y*j,
		c = p.position.z - h.z*k;

	size_t
		il = i,
		jd = j * cfg.n.x,
		kf = k * cfg.n.x * cfg.n.y,
		ir = (i != cfg.n.x - 1 ? i + 1 : 0),
		ju = (j != cfg.n.y - 1 ? j + 1 : 0) * cfg.n.x,
		kb = (k != cfg.n.z - 1 ? k + 1 : 0) * cfg.n.x * cfg.n.y;

	//Offset for the neighbouring vertices.
	*pcfg = {
		{//Offset
			il + jd + kf,
			ir + jd + kf,
			il + ju + kf,
			ir + ju + kf,
			il + jd + kb,
			ir + jd + kb,
			il + ju + kb,
			ir + ju + kb
		},
		{//Weights
			(h.z - c) * (h.y - b) * (h.x - a),
			(h.z - c) * (h.y - b) * a,
			(h.z - c) * b * (h.x - a),
			(h.z - c) * b * a,
			c * (h.y - b) * (h.x - a),
			c * (h.y - b) * a,
			c * b * (h.x - a),
			c * b * a
		}
	};
}
#endif