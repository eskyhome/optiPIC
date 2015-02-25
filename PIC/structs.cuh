#include "dependencies.cuh"

#ifndef PIC_STRUCTS_DEFINED
#define PIC_STRUCTS_DEFINED
struct Offset{ size_t fll, flr, ful, fur, bll, blr, bul, bur; };
struct Weights{ double zyx, zya, zbx, zba, cyx, cya, cbx, cba; };
struct ParticleCfg { Offset o; Weights w; };

struct Config{
	int4 n;	// Number of elements in each direction.
	double3 l;		// Metric length in each direction.
	double
		rho_k,
		charge_by_mass,
		epsilon,
		ts;
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
			tbg,
			nbp,
			nbg,
			nbfreq,
			nbsor;
	} exec_cfg;
};
#endif