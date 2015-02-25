#include <iomanip>
#include <iostream>
#include <random>
#include "dependencies.cuh"
#include "structs.cuh"

#ifndef PIC_PARTICLE_DEFINED
#define PIC_PARTICLE_DEFINED
class Particle {

public:
	double3 position;
	double3 velocity;
	//double2 padding;

	Particle(Config cfg) {
		double
			v_scale = 1 / cfg.ts,
			v_var = 0.0,
			o_scale = cfg.l.x <= cfg.l.y ? (cfg.l.x <= cfg.l.x ? cfg.l.x : cfg.l.z) : (cfg.l.y <= cfg.l.z ? cfg.l.y : cfg.l.z),
			o_var = 1.0 / 64.0;

		double3
			o = make_double3(cfg.l.x / 2, cfg.l.y / 2, cfg.l.z / 2),
			v = make_double3(0 * v_scale / 4, 0.0, 0.0);
		position = vary(o, o_scale * o_var);
		velocity = vary(v, v_scale * v_var);

		if (position.x > cfg.l.x)
			std::cout << (cfg.l.x - position.x);
	};

	Particle() {
		position = make_double3(0.0, 0.0, 0.0);
		velocity = make_double3(0.0, 0.0, 0.0);
	};

	std::string str(){
		std::stringstream st;
		st << std::setprecision(5) << std::setw(12) << position.x << ", " << std::setprecision(5) << std::setw(12) << position.y << ", " << std::setprecision(5) << std::setw(12) << position.z;
		return st.str();
	};

	std::string str_v(){
		std::stringstream st;
		st << std::setprecision(5) << std::setw(12) << velocity.x << ", " << std::setprecision(5) << std::setw(12) << velocity.y << ", " << std::setprecision(5) << std::setw(12) << velocity.z;
		return st.str();
	};

private:
	static std::random_device sg;
	static double seed;
	static std::mt19937_64 re;
	static std::uniform_real_distribution<double> urd;

	static double rval() {
		static std::mt19937 engine{ std::random_device{}() };
		static std::uniform_real_distribution<double> distribution{ -1.0, 1.0 };
		return distribution(engine);
	};
	/*
	static inline double rval(double min = -1.0, double max = 1.0){
		return urd(re);
	};*/
	static inline void between(double val, double min, double max){
		assert(val == val);
		assert(val >= min);
		assert(val <= max);
	};
	static inline void check3(double3 val, double3 def, double var){
		between(val.x, def.x - var, def.x + var);
		between(val.y, def.y - var, def.y + var);
		between(val.z, def.z - var, def.z + var);
	};
	//Provide a double3 with values in the range [default - var, default + var].
	static double3 vary(double3 def, double var) {
		double3 val = def;
		val.x += var * rval();
		val.y += var * rval();
		val.z += var * rval();

		check3(val, def, var);
		return val;
	};
};
#endif