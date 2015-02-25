#include "cfg.cuh"

Config getConfig() {
	std::ifstream ifs("settings.cfg");
	if (!ifs.good()) {
		fprintf(stderr, "ERROR OPENING FILE!");
		return defaultConfig();
	}
	std::map<std::string, double> dict;

	std::string key;
	double value;
	std::string line;
	while (std::getline(ifs, line)) {
		std::istringstream iss(line);

		std::getline(iss, key, '=');
		iss >> value;
		dict[key] = (value);

	}

	if (!ifs.eof()){
		fprintf(stderr, "ERROR READING FILE!");
		return defaultConfig();
	}

	Config cfg =
	{
		//Number of elements (x, y, z, z_fft)
		make_int4((int)dict["nx"], (int)dict["ny"], (int)dict["nz"], (int)dict["nz"]),
		//Size of simulation space in meters
		make_double3(dict["lx"], dict["ly"], dict["lz"]),
		//Derived values, set below
		0, 0, 0,
		//Time step between iterations
		dict["time_step"],
		{//Precomputed values for fft-solver, see computation below.
			0.0,
			0.0,
			0.0,
			0.0
		},

		//SOR
		//Omega
		dict["omega"],
		//Threshold
		dict["threshold"],
		//SOR-iterations
		(int)dict["sor_iterations"],

		//Iterations
		(int)dict["iterations"],
		//trace interval
		32,
		//Number of particles
		(int)dict["particles"],
		{
			//Threads per block for particle based kernels
			dim3((int)dict["tbp"], 1, 1),
			//Threads per block for grid based kernels
			dim3((int)dict["tbx"], (int)dict["tby"], (int)dict["tbz"]),

			//Number of blocks for particle based kernels
			dim3(1, 1, 1),
			//Number of blocks for grid based kernels
			dim3(1, 1, 1),
			//Number of blocks for frequency based kernels
			dim3(1, 1, 1),
			//Number of blocks for SOR-kernels
			dim3(1, 1, 1)
		}
	};
	get_derived_vals(&cfg);
	return cfg;
}

Config defaultConfig() {
	//Constants
	double
		//Pi
		pi = 3.14159265359,
		//Permittivity of free space
		eps_0 = 8.854187817e-12,
		//Electron charge
		e_q = -1.60217657e-19,
		//Electron mass
		e_m = 9.10938291e-31;

	Config cfg =
	{
		//Number of elements (x, y, z, z_fft)
		make_int4(64, 64, 64, 33),
		//Size of simulation space in meters
		make_double3(0.2, 0.2, 0.2),
		//Derived values, set below
		0, 0, 0,
		//Time step between iterations
		1e-6,
		{//Precomputed values for fft-solver, see computation below.
			0.0,
			0.0,
			0.0,
			0.0
		},

		//SOR
		//Omega
		1.78,
		//Threshold
		1e-9,
		//SOR-iterations
		128,

		//Iterations
		2048,
		//trace interval
		32,
		//Number of particles
		256,
		{
			//Threads per block for particle based kernels
			dim3(256, 1, 1),
			//Threads per block for grid based kernels
			dim3(16, 4, 4),

			//Number of blocks for particle based kernels
			dim3(1, 1, 1),
			//Number of blocks for grid based kernels
			dim3(1, 1, 1),
			//Number of blocks for frequency based kernels
			dim3(1, 1, 1),
			//Number of blocks for SOR-kernels
			dim3(1, 1, 1)
		}
	};
	get_derived_vals(&cfg);
	return cfg;
}

void get_derived_vals(Config *cfg) {
	//Constants
	double
		//Pi
		pi = 3.14159265359,
		//Permittivity of free space
		eps_0 = 8.854187817e-12,
		//Electron charge
		e_q = -1.60217657e-19,
		//Electron mass
		e_m = 9.10938291e-31;

	cfg->n.w = cfg->n.z / 2 + 1;
	double h = (cfg->l.x / cfg->n.x) * (cfg->l.y / cfg->n.y) * (cfg->l.z / cfg->n.z);
	cfg->rho_k = e_q / h;
	cfg->charge_by_mass = e_q / e_m;
	cfg->epsilon = eps_0;
	//====//
	// Precomputed values for fft-solver, usage:
	// k^2 = kx^2 + ky^2 + kz^2
	// kx = kxt * i^2 etc.
	// Phi(k) = rho(k) * constant_factor/k^2
	cfg->solve.constant_factor = -1 / (eps_0 * cfg->n.x * cfg->n.y * cfg->n.z);
	cfg->solve.kxt = 4 * pi*pi / (cfg->l.x * cfg->l.x);
	cfg->solve.kyt = 4 * pi*pi / (cfg->l.y * cfg->l.y);
	cfg->solve.kzt = 4 * pi*pi / (cfg->l.z * cfg->l.z);
	//====//
	cfg->trace_interval = (cfg->iterations - 1) / 2 + 1;
	//cfg.exec_cfg.tbp = dim3(256);
	cfg->exec_cfg.nbp = dim3(1 + (cfg->exec_cfg.tbp.x - 1) / cfg->particles);
	cfg->exec_cfg.nbg = dim3(
		1 + (cfg->n.x - 1) / cfg->exec_cfg.tbg.x,
		1 + (cfg->n.y - 1) / cfg->exec_cfg.tbg.y,
		1 + (cfg->n.z - 1) / cfg->exec_cfg.tbg.z
		);
	cfg->exec_cfg.nbfreq = dim3(
		1 + (cfg->n.x - 1) / cfg->exec_cfg.tbg.x,
		1 + (cfg->n.y - 1) / cfg->exec_cfg.tbg.y,
		1 + cfg->n.z / (2 * cfg->exec_cfg.tbg.z)
		);
	cfg->exec_cfg.nbsor = dim3(
		cfg->exec_cfg.nbg.x,
		cfg->exec_cfg.nbg.y,
		1 + (cfg->n.z / 2 - 1) / cfg->exec_cfg.tbg.z
		);
}
