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

//
__global__ void generateParticles(Particle *particles, Config cfg){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cfg.particles)
		return;
	Particle p = particles[idx];
	if (p.position.x < cfg.l.x && p.position.x >= 0)
		return;

	Particle q ={
		make_double3(0, cfg.l.y/2, cfg.l.z/2),
		make_double3(cfg.l.x / 100 * cfg.ts, 0, 0)
	};

	particles[idx] = q;
}

//Determine effect of a particle on its closest grid vertices.
__global__ void determineChargesFromParticles(Particle *particles, cudaPitchedPtr chargeDensity, Config cfg) {
		

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Ensure index is within bounds.
	if (idx >= cfg.particles) {return;}
	
	Particle p = particles[idx];
	
	double
		hx = cfg.l.x / cfg.n.x,
		hy = cfg.l.y / cfg.n.y,
		hz = cfg.l.z / cfg.n.z;
	int
		i = (int)(p.position.x / hx),
		j = (int)(p.position.y / hy),
		k = (int)(p.position.z / hz);
	double
		a = p.position.x - hx*i,
		b = p.position.y - hy*j,
		c = p.position.z - hz*k;
	
	size_t pitch = chargeDensity.pitch;
	char* ptr = (char*)chargeDensity.ptr;

	int i_left = i * sizeof(double),
		i_right = (i != cfg.n.x-1 ? i+1 : 0) * sizeof(double),
		j_lower = j * pitch,
		j_upper = (j != cfg.n.y-1 ? j+1 : 0) * pitch,
		k_front = k * pitch * cfg.n.y,
		k_back	= (k != cfg.n.z-1 ? k+1 : 0) * pitch * cfg.n.y;

	double
		*front_lower_left	= (double*)(ptr + i_left	 + j_lower	+ k_front),
		*front_lower_right	= (double*)(ptr + i_right	 + j_lower	+ k_front),
		*front_upper_left	= (double*)(ptr + i_left	 + j_upper	+ k_front),
		*front_upper_right	= (double*)(ptr + i_right	 + j_upper	+ k_front),
		*back_lower_left	= (double*)(ptr + i_left	 + j_lower	+ k_back),
		*back_lower_right	= (double*)(ptr + i_right	 + j_lower	+ k_back),
		*back_upper_left	= (double*)(ptr + i_left	 + j_upper	+ k_back),
		*back_upper_right	= (double*)(ptr + i_right	 + j_upper	+ k_back);

	atomicAdd(front_lower_left,	   (hz - c) * (hy - b) * (hx - a) * cfg.rho_k); 
	atomicAdd(front_lower_right,   (hz - c) * (hy - b) *	   a  * cfg.rho_k);
	atomicAdd(front_upper_left,	   (hz - c) *		b  * (hx - a) * cfg.rho_k);
	atomicAdd(front_upper_right,   (hz - c) *		b  *	   a  * cfg.rho_k);
	atomicAdd(back_lower_left,			 c  * (hy - b) * (hx - a) * cfg.rho_k); 
	atomicAdd(back_lower_right,			 c  * (hy - b) *	   a  * cfg.rho_k);
	atomicAdd(back_upper_left,			 c  *		b  * (hx - a) * cfg.rho_k);
	atomicAdd(back_upper_right,			 c  *		b  *	   a  * cfg.rho_k);
	return;
}

__global__ void electricFieldFromPotential(cudaPitchedPtr potential, cudaPitchedPtr E, Config cfg) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) {return;}

	char *pptr = (char*)potential.ptr;
	size_t ptch = potential.pitch;

	size_t i_center = i * sizeof(double),
		j_center = j * ptch,
		k_center = k * ptch * cfg.n.y,

//TODO: Allow periodic boundaries.
		i_left	= (i != 0		  ? i-1 : i) * sizeof(double),
		i_right = (i != cfg.n.x-1 ? i+1 : i) * sizeof(double),
	
		j_down	= (j != 0		  ? j-1 : j) * ptch,
		j_up	= (j != cfg.n.y-1 ? j+1 : j) * ptch,
	
		k_front = (k != 0		  ? k-1 : k) * ptch * cfg.n.y,
		k_back	= (k != cfg.n.z-1 ? k+1 : k) * ptch * cfg.n.y;

	double
		*left	= (double*)(pptr + i_left   + j_center + k_center),
		*right	= (double*)(pptr + i_right  + j_center + k_center),
		*down	= (double*)(pptr + i_center + j_down   + k_center),
		*up		= (double*)(pptr + i_center + j_up	   + k_center),
		*front	= (double*)(pptr + i_center + j_center + k_front),
		*back	= (double*)(pptr + i_center + j_center + k_back),

		E_x = *left - *right,
		E_y = *up  - *down,
		E_z = *front - *back;
	double3 E_val = make_double3(E_x, E_y, E_z);
	*(double3*)(((char*) E.ptr) + i * sizeof(double3) + j * E.pitch + k * cfg.n.y * E.pitch) = E_val;
}

__device__ double3 electricFieldAtPoint(double3 position, cudaPitchedPtr E, Config cfg) {
	double
		hx = cfg.l.x / cfg.n.x,
		hy = cfg.l.y / cfg.n.y,
		hz = cfg.l.z / cfg.n.z;

	int i = (int)(position.x / hx),
		j = (int)(position.y / hy),
		k = (int)(position.z / hz);
	double
		a = position.x - hx * i,
		b = position.y - hy * j,
		c = position.z - hz * k;
	//==//
	size_t pitch = E.pitch;
	char* ptr = (char*)E.ptr;

	int i_left	= i * sizeof(double),
		j_lower = j * pitch,
		k_front = k * pitch * cfg.n.y,
#ifdef FFT_SOLVER
		i_right = (i != cfg.n.x-1 ? i+1 : 0) * sizeof(double),
		j_upper = (j != cfg.n.y-1 ? j+1 : 0) * pitch,
		k_back	= (k != cfg.n.z-1 ? k+1 : 0) * pitch * cfg.n.y;
#else
		i_right = (i != cfg.n.x-1 ? i+1 : i) * sizeof(double),
		j_upper = (j != cfg.n.y-1 ? j+1 : j) * pitch,
		k_back	= (k != cfg.n.z-1 ? k+1 : k) * pitch * cfg.n.y;
#endif

	double3
		fll	= *(double3*)(ptr + i_left  + j_lower + k_front),
		flr	= *(double3*)(ptr + i_right + j_lower + k_front),
		ful	= *(double3*)(ptr + i_left  + j_upper + k_front),
		fur	= *(double3*)(ptr + i_right + j_upper + k_front),
		bll	= *(double3*)(ptr + i_left  + j_lower + k_back),
		blr	= *(double3*)(ptr + i_right + j_lower + k_back),
		bul	= *(double3*)(ptr + i_left  + j_upper + k_back),
		bur	= *(double3*)(ptr + i_right + j_upper + k_back),
		val;

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

	double h = hx*hy*hz;
	val.x /= h;
	val.y /= h;
	val.z /= h;
	return val;//make_double3(val.x, val.y, val.z);
}

__device__ double wrap(double value, double max) {
	double res = fmod(value, max);
	if (res < 0)
		return res+max;
	else
		return res;
}

//Update speed and position of the particles.
__global__ void updateParticles(Particle particles[], cudaPitchedPtr E, double timeStep, Config cfg) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Ensure index is within bounds.
	if (idx >= cfg.particles) {return;}
	Particle p = particles[idx];
	
	double3 electricfield = electricFieldAtPoint(p.position, E, cfg);
	// F = q * E
	// a = F / m
	// v = v0 + a * t - drag
	// v = v0 + (q/m) * E * t
	double
		ax = electricfield.x * cfg.charge_by_mass,
		ay = electricfield.y * cfg.charge_by_mass,
		az = electricfield.z * cfg.charge_by_mass,
		//v^(n+1/2) * prev = v^(n-1/2)
		prev = (1 - cfg.drag * cfg.ts);

	p.velocity.x = p.velocity.x * prev + ax * cfg.ts;
	p.velocity.y = p.velocity.y * prev + ay * cfg.ts;
	p.velocity.z = p.velocity.z * prev + az * cfg.ts;

	// p = p0 + v * t
	p.position.x += p.velocity.x * cfg.ts;
	p.position.y += p.velocity.y * cfg.ts;
	p.position.z += p.velocity.z * cfg.ts;

	// Ensure p is within grid, x dimension handled by generateParticles?
	p.position.x = wrap(p.position.x, cfg.l.x);
	p.position.y = wrap(p.position.y, cfg.l.y);
	p.position.z = wrap(p.position.z, cfg.l.z);
	
	particles[idx] = p;
}

//Solve and normalize
__global__ void solve(cudaPitchedPtr freq, Config cfg) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.w) {return;}
	size_t offset = i*sizeof(cufftDoubleComplex) + (j + k * cfg.n.y) * freq.pitch;
	if (i > cfg.n.x/2)
		i = cfg.n.x - i;
	if (j > cfg.n.y/2)
		j = cfg.n.y - j;
	if (k > cfg.n.z/2)
		k = cfg.n.z - k;

	double K_sqrd = cfg.solve.kxt * i*i + cfg.solve.kyt * j*j + cfg.solve.kzt * k*k;
	if(i+j+k==0)
		K_sqrd = 1.0;

	double scale = cfg.solve.constant_factor / K_sqrd;
	
	cufftDoubleComplex val = *(cufftDoubleComplex*)((char*)freq.ptr + offset);
	val.x *= scale;
	val.y *= scale;
	*(cufftDoubleComplex*)((char*)freq.ptr + offset) = val;
}

__global__ void initSOR(cudaPitchedPtr Rho, cudaPitchedPtr Phi, Config cfg) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
			j = blockIdx.y * blockDim.y + threadIdx.y,
			k = blockIdx.z * blockDim.z + threadIdx.z;
		if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) {return;}

		double
		h2 = cfg.l.x / cfg.n.x + cfg.l.y / cfg.n.y + cfg.l.z / cfg.n.z;
		
		size_t offset = i*sizeof(double) + (j + k * cfg.n.y) * Rho.pitch;

		double charge = *(double*)((char*)Rho.ptr + offset);
		*(double*)(((char *) Phi.ptr) + offset) = charge * h2 / (cfg.epsilon * 6);
}
__global__ void SOR(cudaPitchedPtr Phi, Config cfg, int flag) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y,
		k = 2 * (blockIdx.z * blockDim.z + threadIdx.z) + (i+j+flag)%2;
	if(k==cfg.n.z)
		k = 0;
		//Flag is 0 or 1 for red or black iterations
		//red black: (i+j+flag)=odd => (i, j, k+1)
	
	//TODO: ensure coalesced reads/writes!

	if(i >= cfg.n.x || j >= cfg.n.y || k >= cfg.n.z) {return;}

	char *pptr = (char*)Phi.ptr;

	int i_center = i * sizeof(double),
		j_center = j * Phi.pitch,
		k_center = k * Phi.pitch * cfg.n.y,

		i_left	= (i != 0		  ? i-1 : i) * sizeof(double),
		i_right = (i != cfg.n.x-1 ? i+1 : i) * sizeof(double),
	
		j_down	= (j != 0		  ? j-1 : j) * Phi.pitch,
		j_up	= (j != cfg.n.y-1 ? j+1 : j) * Phi.pitch,

		k_front = (k != 0		  ? k-1 : k) * Phi.pitch * cfg.n.y,
		k_back	= (k != cfg.n.z-1 ? k+1 : k) * Phi.pitch * cfg.n.y;

	double
		center	= *(double*)(pptr + i_center + j_center	+ k_center),
		left	= *(double*)(pptr + i_left	 + j_center	+ k_center),
		right	= *(double*)(pptr + i_right	 + j_center	+ k_center),
		down	= *(double*)(pptr + i_center + j_down	+ k_center),
		up		= *(double*)(pptr + i_center + j_up		+ k_center),
		front	= *(double*)(pptr + i_center + j_center	+ k_front),
		back	= *(double*)(pptr + i_center + j_center	+ k_back),
		tmp, val;

	tmp = (left + right + down + up + front + back)/6;
	val = center + cfg.omega * (tmp - center);

	//Assume input and output to have same dimensions and padding (pitch).
	*(double*)(pptr + i_center + j_center + k_center) = val;
}