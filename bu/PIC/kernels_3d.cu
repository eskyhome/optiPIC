#include "Particles.cuh"

__device__  void atomicAdd(double *address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
}

//Determine effect of a particle on its closest grid vertices. NOW WORKS CORRECTLY (need to verify atomicity...)
__global__ void determineChargesFromParticles3D(Particle *particles, cudaPitchedPtr chargeDensity, Config cfg) {
		int hx = cfg.l.x / cfg.n.x,
		hy = cfg.l.y / cfg.n.y,
		hz = cfg.l.z / cfg.n.z;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Ensure index is within bounds.
	if (idx >= cfg.particles) {return;}
	
	Particle p = particles[idx];

	int i = (int)(p.position.x / hx),
		j = (int)(p.position.y / hy),
		k = (int)(p.position.z / hz);
	double a = p.position.x/hx - i,
		b = p.position.y/hy - j,
		c = p.position.z/hz - k;
	
	double
		*fLowerRow = (double*)(((char*)chargeDensity.ptr) + j				* chargeDensity.pitch + k * cfg.n.y * chargeDensity.pitch),
		*fUpperRow = (double*)(((char*)chargeDensity.ptr) + (j+1)%cfg.n.y	* chargeDensity.pitch + k * cfg.n.y * chargeDensity.pitch),
		*fll = &fLowerRow[i],
		*flr = &fLowerRow[(i+1)%cfg.n.x],
		*ful = &fUpperRow[i],
		*fur = &fUpperRow[(i+1)%cfg.n.x],

		*bLowerRow = (double*)(((char*)chargeDensity.ptr) + j				* chargeDensity.pitch + (k+1)%cfg.n.z * cfg.n.y * chargeDensity.pitch),
		*bUpperRow = (double*)(((char*)chargeDensity.ptr) + (j+1)%cfg.n.y	* chargeDensity.pitch + (k+1)%cfg.n.z * cfg.n.y * chargeDensity.pitch),
		*bll = &bLowerRow[i],
		*blr = &bLowerRow[(i+1)%cfg.n.x],
		*bul = &bUpperRow[i],
		*bur = &bUpperRow[(i+1)%cfg.n.x];

	//       bul___________________bur
	//       /|                   /|
	//      / |                  / |
	//     /  |                 /  |
	//    /   |___a___         /   |
	//   /   /|      /|       /    |
	//  /   /_|_____p |      /     |
	// ful_/__|____/|_|____fur     |
	// |  /  bll__/_|_|_____|_____blr
	// | /   /   /  | /     |     /
	// |/   /___/___|/      |    /
	// |___a___/    /       |   /
	// |  /    |   /        |  /
	// | /     b  c         | /
	// |/      | /          |/
	// fll_____|/__________flr
	// (i, j)            (i+1, j)
	
	//TODO Synchronize/Optimize write to memory:
	atomicAdd(fll, (hz - c) * (hy - b) * (hx - a) * cfg.rho_k);
	atomicAdd(ful, (hz - c) *		b  * (hx - a) * cfg.rho_k); 
	atomicAdd(flr, (hz - c) * (hy - b) *	   a  * cfg.rho_k);
	atomicAdd(fur, (hz - c) *		b  *	   a  * cfg.rho_k);

	atomicAdd(bll,		 c	* (hy - b) * (hx - a) * cfg.rho_k);
	atomicAdd(bul,		 c	*		b  * (hx - a) * cfg.rho_k); 
	atomicAdd(blr,		 c	* (hy - b) *	   a  * cfg.rho_k);
	atomicAdd(bur,		 c	*		b  *	   a  * cfg.rho_k);
}

__global__ void electricFieldFromPotential3D(cudaPitchedPtr potential, cudaPitchedPtr E, Config cfg) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) {return;}

	//        top  back
	//          \  /
	// left __ __\/__ __ right
	//           /\
	//          /  \
	//      front  down

	double
		top		= *(double*)(((char*) potential.ptr) + (j+1)%cfg.n.y * potential.pitch + k * cfg.n.y * potential.pitch + i*sizeof(double) ),
		bottom	= *(double*)(((char*) potential.ptr) + (j-1)%cfg.n.y * potential.pitch + k * cfg.n.y * potential.pitch + i*sizeof(double) ),

		front	= *(double*)(((char*) potential.ptr) + j * potential.pitch + (k+1)%cfg.n.z * cfg.n.y * potential.pitch + i*sizeof(double) ),
		back	= *(double*)(((char*) potential.ptr) + j * potential.pitch + (k-1)%cfg.n.z * cfg.n.y * potential.pitch + i*sizeof(double) ),
		
		left	= *(double*)(((char*) potential.ptr) + j * potential.pitch + k * cfg.n.y * potential.pitch + (i-1)%cfg.n.x*sizeof(double) ),
		right	= *(double*)(((char*) potential.ptr) + j * potential.pitch + k * cfg.n.y * potential.pitch + (i+1)%cfg.n.x*sizeof(double) ),

		E_x = left - right,
		E_y = top  - bottom,
		E_z = front - back;

	((double3*) E.ptr)[i + E.pitch * j] = make_double3(E_x, E_y, E_z);
}

__device__ double3 electricFieldAtPoint3D(double3 position, cudaPitchedPtr E, Config cfg) {
	int hx = cfg.l.x / cfg.n.x,
		hy = cfg.l.y / cfg.n.y,
		hz = cfg.l.z / cfg.n.z;

	int i = (int)(position.x / hx),
		j = (int)(position.y / hy),
		k = (int)(position.z / hz);
	double a = position.x/hx - i,
		b = position.y/hy - j,
		c = position.z/hz - k;
	
	double3
		*fLowerRow = (double3*)(((char*)E.ptr) + j				* E.pitch + k * cfg.n.y * E.pitch),
		*fUpperRow = (double3*)(((char*)E.ptr) + (j+1)%cfg.n.y	* E.pitch + k * cfg.n.y * E.pitch),
		fll = fLowerRow[i],
		flr = fLowerRow[(i+1)%cfg.n.x],
		ful = fUpperRow[i],
		fur = fUpperRow[(i+1)%cfg.n.x],

		*bLowerRow = (double3*)(((char*)E.ptr) + j				* E.pitch + (k+1)%cfg.n.z * cfg.n.y * E.pitch),
		*bUpperRow = (double3*)(((char*)E.ptr) + (j+1)%cfg.n.y	* E.pitch + (k+1)%cfg.n.z * cfg.n.y * E.pitch),
		bll = bLowerRow[i],
		blr = bLowerRow[(i+1)%cfg.n.x],
		bul = bUpperRow[i],
		bur = bUpperRow[(i+1)%cfg.n.x],
		
		val;

	// (i, j+1)        (i+1, j+1)
	// ul------------------ur     b
	// |                    |     /
	// |                    |    /
	// |                    |   /
	// |---a---p            |  f
	// |       |            |
	// |       b            |
	// |       |            |
	// ll------------------lr
	// (i, j)            (i+1, j)

	val.x
		= fll.x * (hz - c) * (hy - b) * (hx - a)
		+ flr.x * (hz - c) * (hy - b) * a
		+ ful.x * (hz - c) * b * (hx - a)
		+ fur.x * (hz - c) * b * a

		+ bll.x * c * (hy - b) * (hx - a)
		+ blr.x * c * (hy - b) * a
		+ bul.x * c * b * (hx - a)
		+ bur.x * c * b * a;

	val.y
		= fll.y * (hz - c) * (hy - b) * (hx - a)
		+ flr.y * (hz - c) * (hy - b) * a
		+ ful.y * (hz - c) * b * (hx - a)
		+ fur.y * (hz - c) * b * a

		+ bll.y * c * (hy - b) * (hx - a)
		+ blr.y * c * (hy - b) * a
		+ bul.y * c * b * (hx - a)
		+ bur.y * c * b * a;

	val.z
		= fll.z * (hz - c) * (hy - b) * (hx - a)
		+ flr.z * (hz - c) * (hy - b) * a
		+ ful.z * (hz - c) * b * (hx - a)
		+ fur.z * (hz - c) * b * a

		+ bll.z * c * (hy - b) * (hx - a)
		+ blr.z * c * (hy - b) * a
		+ bul.z * c * b * (hx - a)
		+ bur.z * c * b * a;

	return val;
}

__device__ double wrap(double value, double max) {
		double res = fmod(value, max);
	if (res < 0)
		return res+max;
	else
		return res;
}

//Update speed and position of the particles.
__global__ void updateParticles3D(Particle particles[], cudaPitchedPtr E, double timeStep, Config cfg) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Ensure index is within bounds.
	if (idx > cfg.particles) {return;}
	Particle p = particles[idx];
	
	double3 eAtPt = electricFieldAtPoint3D(p.position, E, cfg);
	p.eAtP = eAtPt;
	// F = q * E
	// a = F / m
	// v = v0 + a * t 
	// v = v0 + (q/m) * E * t
	p.velocity.x += eAtPt.x * cfg.charge_by_mass * cfg.ts;
	p.velocity.y += eAtPt.y * cfg.charge_by_mass * cfg.ts;
	p.velocity.z += eAtPt.z * cfg.charge_by_mass * cfg.ts;

	// p = p0 + v * t
	p.position.x += p.velocity.x * cfg.ts;
	p.position.y += p.velocity.y * cfg.ts;
	p.position.z += p.velocity.z * cfg.ts;

	// Ensure p is within [0, N * H]
	p.position.x = wrap(p.position.x, cfg.l.x);
	p.position.y = wrap(p.position.y, cfg.l.y);
	p.position.z = wrap(p.position.z, cfg.l.z);
	
	particles[idx] = p;
}

//Solve and normalize
__global__ void solve3D(cudaPitchedPtr freq, Config cfg) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.w) {return;}

	cufftDoubleComplex* row = (cufftDoubleComplex*)(((char*) freq.ptr) + j * freq.pitch + k * cfg.n.y * freq.pitch);
	row[i].x *= cfg.solve_factor;
	row[i].y *= cfg.solve_factor;
}

__global__ void SOR3D (cudaPitchedPtr in, cudaPitchedPtr out, Config cfg) {
	// Make red-black!
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) {return;}

	char *cptr = (char*)in.ptr,
		ptch = in.pitch;

	double
		*center	= &((double*)(cptr  + j	* ptch * k * cfg.n.y * ptch))[i],
		*left	= &((double*)(cptr  + j	* ptch * k * cfg.n.y * ptch))[(cfg.n.x + i-1)%cfg.n.x],
		*right	= &((double*)(cptr  + j	* ptch * k * cfg.n.y * ptch))[(i+1)%cfg.n.x],
		*up		= &((double*)(cptr  + (j+1)%cfg.n.y	* ptch * k * cfg.n.y * ptch))[i],
		*low	= &((double*)(cptr  + (cfg.n.y + j-1)%cfg.n.y * ptch * k * cfg.n.y * ptch))[i],
		*front	= &((double*)(cptr  + j	* ptch * (cfg.n.z + k-1)%cfg.n.z * cfg.n.y * ptch))[i],
		*back	= &((double*)(cptr  + j	* ptch * (k+1)%cfg.n.z * cfg.n.y * ptch))[i],
		tmp, diff;

	tmp = (*left + *right + *up + *low + *front + *back)/9;
	diff = tmp - *center;
	((double*) out.ptr)[i + out.pitch * j] += cfg.omega * diff;
}