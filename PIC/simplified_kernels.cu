#include "dependencies.cuh"
#include "particle.cuh"
#include "structs.cuh"
#include "kernel_helper.cuh"

//Solvers

__device__ void fftsolver_device(cufftDoubleComplex* freq, Config cfg){
	size_t
		i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.w) { return; }
	size_t offset = i + cfg.n.x * (j + cfg.n.y * k);

	if (i > cfg.n.x / 2)
		i = cfg.n.x - i;
	if (j > cfg.n.y / 2)
		j = cfg.n.y - j;
	if (k > cfg.n.z / 2)
		k = cfg.n.z - k;

	double K_sqrd = cfg.solve.kxt * i*i + cfg.solve.kyt * j*j + cfg.solve.kzt * k*k;
	if (i + j + k == 0)
		K_sqrd = 1.0;

	double scale = cfg.solve.constant_factor / K_sqrd;
	cufftDoubleComplex val = freq[offset];
	val.x *= scale;
	val.y *= scale;
	freq[offset] = val;
}
//FFT-solver
__global__ void fftsolver_kernel(cufftDoubleComplex* freq, Config cfg) {
	fftsolver_device(freq, cfg);
}
//SOR-solver
__global__ void sorsolver_kernel(double* phi, Config cfg, size_t flag) {
	size_t
		i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = 2 * (blockIdx.z * blockDim.z + threadIdx.z) + (i + j + flag) % 2;
	if (i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) { return; }
	
	size_t
		l = (i != 0 ? i - 1 : i),
		r = (i != cfg.n.x - 1 ? i + 1 : i),

		d = (j != 0 ? j - 1 : j) * cfg.n.x,
		u = (j != cfg.n.y - 1 ? j + 1 : j) * cfg.n.x,

		f = (k != 0 ? k - 1 : k) * cfg.n.x * cfg.n.y,
		b = (k != cfg.n.z - 1 ? k + 1 : k) * cfg.n.x * cfg.n.y;

	j = j * cfg.n.x;
	k = k * cfg.n.x * cfg.n.y;
	
	double
		center	= phi[i + j + k],
		left	= phi[l + j + k],
		right	= phi[r + j + k],
		down	= phi[i + d + k],
		up		= phi[i + u + k],
		front	= phi[i + j + f],
		back	= phi[i + j + b],
		tmp, val;

	tmp = (left + right + down + up + front + back) / 6;
	val = center + cfg.omega * (tmp - center);

	phi[i + j + k] = val;
}
//Saturate phi with initial values from rho
__global__ void sorinit_kernel(double* phi, double* rho, Config cfg) {
	size_t
		i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) { return; }
	size_t offset = i + cfg.n.x * (j + cfg.n.y * k);

	double
		h2 = cfg.l.x / cfg.n.x + cfg.l.y / cfg.n.y + cfg.l.z / cfg.n.z;

	phi[offset] = rho[offset] * h2 / (cfg.epsilon *  6);
}

//Electric field calculation
__global__ void electricfield_kernel(double* phi, double4* E, Config cfg) {
	size_t
		i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) { return; }

	size_t
		left	= (i != 0 ? -1 : 0),
		down	= (j != 0 ? -1 : 0) * cfg.n.x,
		front	= (k != 0 ? -1 : 0) * cfg.n.x * cfg.n.y,
		right	= (i != cfg.n.x - 1 ? 1 : 0),
		up		= (j != cfg.n.y - 1 ? 1 : 0) * cfg.n.x,
		back	= (k != cfg.n.z - 1 ? 1 : 0) * cfg.n.x * cfg.n.y;

	j *= cfg.n.x;
	k *= cfg.n.x * cfg.n.y;
	size_t index = i + j + k;

	double
		x = phi[index + left]	- phi[index + right],
		y = phi[index + up]		- phi[index + down],
		z = phi[index + front]	- phi[index + back];
	E[index] = make_double4(x, y, z, 0.0);
}

//Particle updates and charge distribution
__global__ void particleUpdate_kernel(Particle* particles, double4* E, double* rho, Config cfg) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cfg.particles) { return; }

	double3 h = div_v(cfg.l, cfg.n);
	Particle p = particles[idx];
	ParticleCfg pcfg;
	pCfg(p, cfg, h, &pcfg);

	double3 a;
	a.x = E[pcfg.o.fll].x * pcfg.w.zyx
		+ E[pcfg.o.flr].x * pcfg.w.zya
		+ E[pcfg.o.ful].x * pcfg.w.zbx
		+ E[pcfg.o.fur].x * pcfg.w.zba
		+ E[pcfg.o.bll].x * pcfg.w.cyx
		+ E[pcfg.o.blr].x * pcfg.w.cya
		+ E[pcfg.o.bul].x * pcfg.w.cbx
		+ E[pcfg.o.bur].x * pcfg.w.cba,

	a.y = E[pcfg.o.fll].y * pcfg.w.zyx
		+ E[pcfg.o.flr].y * pcfg.w.zya
		+ E[pcfg.o.ful].y * pcfg.w.zbx
		+ E[pcfg.o.fur].y * pcfg.w.zba
		+ E[pcfg.o.bll].y * pcfg.w.cyx
		+ E[pcfg.o.blr].y * pcfg.w.cya
		+ E[pcfg.o.bul].y * pcfg.w.cbx
		+ E[pcfg.o.bur].y * pcfg.w.cba,

	a.z	= E[pcfg.o.fll].z * pcfg.w.zyx
		+ E[pcfg.o.flr].z * pcfg.w.zya
		+ E[pcfg.o.ful].z * pcfg.w.zbx
		+ E[pcfg.o.fur].z * pcfg.w.zba
		+ E[pcfg.o.bll].z * pcfg.w.cyx
		+ E[pcfg.o.blr].z * pcfg.w.cya
		+ E[pcfg.o.bul].z * pcfg.w.cbx
		+ E[pcfg.o.bur].z * pcfg.w.cba;

	double c_by_h = cfg.charge_by_mass / vol(h);

	a = mul_s(a, c_by_h);
	p.velocity = add_v(p.velocity, mul_s(a, cfg.ts));

	p.position = add_v(p.position, mul_s(p.velocity, cfg.ts));

	particles[idx] = p;

	pCfg(p, cfg, h, &pcfg);

	//Add the particle's contribution to each neighbouring vertices.
	atomicAdd(&rho[pcfg.o.fll], pcfg.w.zyx * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.flr], pcfg.w.zya * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.ful], pcfg.w.zbx * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.fur], pcfg.w.zba * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.bll], pcfg.w.cyx * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.blr], pcfg.w.cya * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.bul], pcfg.w.cbx * cfg.rho_k);
	atomicAdd(&rho[pcfg.o.bur], pcfg.w.cba * cfg.rho_k);
}