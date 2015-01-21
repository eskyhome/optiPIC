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
__global__ void determineChargesFromParticles(Particle *particles, cudaPitchedPtr chargeDensity) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Ensure index is within bounds.
	if (idx >= NR_OF_PARTICLES) {return;}
	
	Particle p = particles[idx];

	int i = (int)(p.position.x / HX),
		j = (int)(p.position.y / HY);
	double a = p.position.x/HX - i,
		b = p.position.y/HY - j;
	
	double
		*lowerRow = (double*)(((char*)chargeDensity.ptr)  + j * chargeDensity.pitch),
		*upperRow = (double*)(((char*)chargeDensity.ptr)  + (j+1)%N_Y * chargeDensity.pitch),
		*ll = &lowerRow[i],
		*lr = &lowerRow[(i+1)%N_X],
		*ul = &upperRow[i],
		*ur = &upperRow[(i+1)%N_X];
	
	// (i, j+1)        (i+1, j+1)
	// ul------------------ur
	// |                    |
	// |                    |
	// |                    |
	// |---a---p            |
	// |       |            |
	// |       b            |
	// |       |            |
	// ll------------------lr
	// (i, j)            (i+1, j)
	
	//TODO Synchronize/Optimize write to memory:
	atomicAdd(ll, (HY - b) * (HX - a) * RHO_K);
	atomicAdd(ul, b * (HX - a) * RHO_K); 
	atomicAdd(lr, (HY - b) * a * RHO_K);
	atomicAdd(ur, b * a * RHO_K);
}

__global__ void electricFieldFromPotential(cudaPitchedPtr potential, cudaPitchedPtr E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= N_X || j >= N_Y) {return;}

	double
		*upperrow = (double*)(((char*) potential.ptr) + (j-1)%N_Y * potential.pitch),
		*middlerow = (double*)(((char*) potential.ptr) + j * potential.pitch),
		*lowerrow = (double*)(((char*) potential.ptr) + (j+1)%N_Y * potential.pitch),
		E_x = (middlerow[(i-1)%N_X] - middlerow[(i+1)%N_X])/(2*HX),
		E_y = (upperrow[i] - lowerrow[i])/(2*HY);
	((double2*) E.ptr)[i + E.pitch * j] = make_double2(E_x, E_y);
}

__device__ double2 electricFieldAtPoint(double2 position, cudaPitchedPtr E) {
	int i = (int)(position.x / HX),
		j = (int)(position.y / HY);
	double a = position.x - i*HX,
		b = position.y - j*HY;

	double2 val,
		*lowerRow = (double2*)(((char*)E.ptr)  + j * E.pitch),
		*upperRow = (double2*)(((char*)E.ptr)  + (j+1)%N_Y * E.pitch),
		ll = lowerRow[i],
		lr = lowerRow[(i+1)%N_X],
		ul = upperRow[i],
		ur = upperRow[(i+1)%N_X];

	// (i, j+1)        (i+1, j+1)
	// ul------------------ur
	// |                    |
	// |                    |
	// |                    |
	// |---a---p            |
	// |       |            |
	// |       b            |
	// |       |            |
	// ll------------------lr
	// (i, j)            (i+1, j)

	val.x = ll.x * (HY - b) * (HX - a)
		+ lr.x * (HY - b) * a
		+ ul.x * b * (HX - a)
		+ ur.x * b * a;

	val.y = ll.y * (HY - b) * (HX - a)
		+ lr.y * (HY - b) * a
		+ ul.y * b * (HX - a)
		+ ur.y * b * a;
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
__global__ void updateParticles(Particle particles[], cudaPitchedPtr E, double timeStep) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Ensure index is within bounds.
	if (idx > NR_OF_PARTICLES) {return;}
	Particle p = particles[idx];
	
	double2 eAtPt = electricFieldAtPoint(p.position, E);
	p.eAtP = eAtPt;
	// F = q * E
	// a = F / m
	// v = v0 + a * t 
	// v = v0 + (q/m) * E * t
	p.velocity.x += eAtPt.x * CHARGE_BY_MASS * TS;
	p.velocity.y += eAtPt.y * CHARGE_BY_MASS * TS;

	// p = p0 + v * t
	p.position.x += p.velocity.x * TS;
	p.position.y += p.velocity.y * TS;

	// Ensure p is within [0, N * H]
	p.position.x = wrap(p.position.x, LX);
	p.position.y = wrap(p.position.y, LY);
	
	particles[idx] = p;
}

//Solve and normalize
__global__ void solve(cudaPitchedPtr freq) {
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= N_X || j >= N_F) {return;}

	cufftDoubleComplex* row = (cufftDoubleComplex*)(((char*) freq.ptr) + j * freq.pitch);
	row[i].x *= SOLVE_FACTOR;
	row[i].y *= SOLVE_FACTOR;
}

//Normalize result of ifft(fft(p(x, y)))
__global__ void scaleData(cudaPitchedPtr potential){
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= N_X || j >= N_Y) {return;}

	((double*) potential.ptr)[i + potential.pitch * j] /= N_X * N_Y;
}

__global__ void SOR (cudaPitchedPtr in, cudaPitchedPtr out) {
	// Make red-black!
	int i = blockIdx.x * blockDim.x + threadIdx.x,
		j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= N_X || j >= N_Y) {return;}

	char *cptr = (char*)in.ptr;

	double
		*center	= &((double*)(cptr  + j			* in.pitch))[i],
		*left	= &((double*)(cptr  + j			* in.pitch))[(N_X + i-1)%N_X],
		*right	= &((double*)(cptr  + j			* in.pitch))[(N_X + i+1)%N_X],
		*up		= &((double*)(cptr  + (j+1)%N_Y * in.pitch))[i],
		*low	= &((double*)(cptr  + (j-1)%N_Y * in.pitch))[i],
		tmp, diff;

	tmp = (*left + *right + *up + *low)/4;
	diff = tmp - *center;
	((double*) out.ptr)[i + out.pitch * j] += OMEGA * diff;
}