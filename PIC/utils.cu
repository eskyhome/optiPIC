#include "Particles.cuh"
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>

void allocateMemory3D(Particle **particles, cudaPitchedPtr *chargeDensity, cudaPitchedPtr *potential, cudaPitchedPtr *E, cudaPitchedPtr *freq, Config cfg) {
	cudaExtent
		cd_ext =	make_cudaExtent(cfg.n.x * sizeof(double),				cfg.n.y, cfg.n.z),
		pot_ext =	make_cudaExtent(cfg.n.x * sizeof(double),				cfg.n.y, cfg.n.z),
		E_ext =		make_cudaExtent(cfg.n.x * sizeof(double3),				cfg.n.y, cfg.n.z),
		freq_ext =	make_cudaExtent(cfg.n.x * sizeof(cufftDoubleComplex),	cfg.n.y, cfg.n.z);

	cudaChk( cudaMalloc(particles, sizeof(Particle) * cfg.particles));
	cudaChk( cudaMalloc3D(chargeDensity, cd_ext));
	cudaChk( cudaMalloc3D(potential, pot_ext));
	cudaChk( cudaMalloc3D(E, E_ext));
	cudaChk( cudaMalloc3D(freq, freq_ext));
	cudaChk( cudaGetLastError());
}

void cleanData(Particle *particles, cudaPitchedPtr chargeDensity, cudaPitchedPtr potential, cudaPitchedPtr E, cudaPitchedPtr freq) {
	cudaChk( cudaFree(particles));
	cudaChk( cudaFree(chargeDensity.ptr));
	cudaChk( cudaFree(potential.ptr));
	cudaChk( cudaFree(E.ptr));
	cudaChk(cudaFree(freq.ptr));
	cudaChk(cudaGetLastError());
}

void printTimingdata(double* timing, int* param1, int* param2, bool two_dim_plot, int length, char* testid){
	std::ofstream file;
	std::stringstream filename;
	filename << "results/timing_" << testid << ".xml";
	file.open(filename.str());
	if(two_dim_plot){
		for (int i = 0; i < length; i++){
			file << timing[i] << ", " << param1[i] << ", " << param2[i] << "\n";
		}
	}
	else {
		for (int i = 0; i < length; i++){
			file << timing[i] << ", " << param1[i] << "\n";
		}
	}
	file.close();
}

void printTracefile(char *solver, Particle *pta, Config cfg) {
	std::ofstream file;
	std::stringstream filename;
	filename << "trace/tracefile{"
		<< "[solver=" << solver << "]"
		<< "[it=" << cfg.iterations << "]"
		<< "[np=" << cfg.particles << "]"
		<< "[nx=" << cfg.n.x << "]"
		<< "[ny=" << cfg.n.y << "]"
		<< "[nz=" << cfg.n.z << "]"
		<< "}.xml";
	file.open(filename.str());;
	file << "<root n_iterations=\"" << cfg.iterations << "\" n_particles=\"" << cfg.particles << "\" particle_interval=\"" << cfg.trace_interval << "\" >\n\n";
	for (int i = 0; i <= cfg.iterations/cfg.trace_interval; i++) {
		file << "<iteration time=\"" << i*cfg.trace_interval*cfg.ts << "\" >\n";
		for (int j = 0; j < cfg.particles; j++) {
			Particle p = *((Particle*)((char*)pta + i*cfg.particles*sizeof(Particle) + j*sizeof(Particle)));
			file << "\t<particle id=\"" << j << "\" >\n"
				<< "\t\t<position x=\"" << p.position.x << "\" y=\"" << p.position.y << "\" z=\"" << p.position.z << "\" />\n"
				<< "\t\t<velocity x=\"" << p.velocity.x << "\" y=\"" << p.velocity.y << "\" z=\"" << p.velocity.z <<  "\" />\n"
				<< "\t\t<electric x=\"" << p.electricfield.x << "\" y=\"" << p.electricfield.y << "\" z=\"" << p.electricfield.z <<  "\" />\n"
				<< "\t</particle>\n";
		}
		file << "</iteration>\n\n";
	}
	file << "</root>";
	file.close();
}

void toNumpy(char *filename){
	std::ofstream file;
	file.open(filename);
	file << "\\x93NUMPY" << "\\x01" << "\\x00";
	file << "header_len" << "{'decr': [('')]\
							'fortran_order': False,\
							'shape': }";

}

void printPhiRho(char *fieldname, double *field, int iteration, Config cfg){
	char title[32];
	sprintf_s(title, "%s%d.txt", fieldname, iteration);

	std::ofstream file;
	file.open(title);
	file << "Iteration " << iteration <<"\n";
	for (int j = 0; j < cfg.n.y; j++) {
		file << std::setw(4) << std::setprecision(4) << cfg.l.y - (j + 1)*cfg.l.y/cfg.n.y << "\t|\t";
		for (int i = 0; i < cfg.n.x; i++) {
			double val = field[i + cfg.n.x * (cfg.n.y - j - 1)];
			std::ostringstream strs;
			strs << ((val<0)?"":" ") << std::setw(4) << std::setprecision(4) << val;
			std::string s = strs.str();
			file << strs.str() << std::string(4 - s.length()/4, '\t');
		}
		file << "\n";
	}
	file << "y______\t|" << std::string(cfg.n.x * 16, '_') << "\n\t\t| x\t";
	for (int i = 0; i < cfg.n.x; i++) {
		std::ostringstream strs;
		double val = i*cfg.l.x/cfg.n.x;
		strs << ((val<0)?"":" ") << std::setw(4) << std::setprecision(4) << val;
		std::string s = strs.str();
		file << strs.str() << std::string(4 - s.length()/4, '\t');
	}
	file.close();
}

void printEField(char *fieldname, double3 *field, int iteration, Config cfg){
	char title[32];
	sprintf_s(title, "%s%d.txt", fieldname, iteration);

	std::ofstream file;
	file.open(title);
	file << "Iteration " << iteration <<"\n";
	for (int j = 0; j < cfg.n.y; j++) {
		file << std::setw(4) << std::setprecision(4) << cfg.l.y - (j + 1)*cfg.l.y/cfg.n.y << "\t|\t";
		for (int i = 0; i < cfg.n.x; i++) {
			double3 val3 = field[i + cfg.n.x * (cfg.n.y - j - 1)];
			double val = val3.x + val3.y + val3.z;
			std::ostringstream strs;
			strs << ((val<0)?"":" ") << std::setw(4) << std::setprecision(4) << val << " ";
			std::string s = strs.str();
			file << strs.str() << std::string(4 - s.length()/4, '\t');
		}
		file << "\n";
	}
	file << "y______\t|" << std::string(cfg.n.x * 16, '_') << "\n\t\t| x\t";
	for (int i = 0; i < cfg.n.x; i++) {
		std::ostringstream strs;
		double val = i*cfg.l.x/cfg.n.x;
		strs << ((val<0)?"":" ") << std::setw(4) << std::setprecision(4) << val;
		std::string s = strs.str();
		file << strs.str() << std::string(4 - s.length()/4, '\t');
	}
	file.close();
}

void printFreq(char *fieldname, cufftDoubleComplex *field, int iterations, Config cfg) {
	char title[32];
	sprintf_s(title, "%s%dfreq.txt", fieldname, iterations);

	std::ofstream file;
	file.open(title);
	file << "Iteration " << iterations <<"\n";
	for (int j = 0; j < cfg.n.y; j++) {
		for (int i = 0; i < cfg.n.x; i++) {
			//file << "\t[x:" << std::setprecision(3) << field[i + N_X * j].x << ", y:" << field[i + N_X * j].y << "],";
			file << "\t[" << std::setprecision(5) << field[i + cfg.n.x * j].x << "," << std::setprecision(5) << field[i + cfg.n.x * j].y << "]";
		}
		file << "\n";
	}
	file.close();
}

void debug(void* host, cudaPitchedPtr device, char val, int iteration, Config cfg){
	void * ptr = (void*)(((char*)device.ptr) + cfg.n.z/2 * device.pitch * cfg.n.y);
	cudaChk(cudaMemcpy2D(host, device.xsize, ptr, device.pitch, device.xsize, device.ysize, cudaMemcpyDeviceToHost));
	if (val == 2)
		printFreq("x_debug", (cufftDoubleComplex*) host, iteration, cfg);
	else if (val == 3)
		printEField("x_debug", (double3 *)host, iteration, cfg);
	else
		printPhiRho("x_debug", (double *)host, iteration, cfg);
	cudaChk( cudaDeviceSynchronize());
	cudaChk( cudaGetLastError());
}

//Defines constants and other runtime parameters
Config getConfig() {
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
		//Drag term
		0,
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
			//Number of blocks for particle based kernels
			dim3(1, 1, 1),

			//Threads per block for grid based kernels
			dim3(16, 4, 4),
			//Number of blocks for grid based kernels
			dim3(1, 1, 1),

			//Threads per block for frequency domain kernels
			dim3(16, 4, 4),
			//Number of blocks for frequency based kernels
			dim3(1, 1, 1),
			
			//Number of blocks for SOR-kernels
			dim3(1, 1, 1)
		}
	};
	cfg.n.w = cfg.n.z/2 + 1;
	double h = (cfg.l.x / cfg.n.x) * (cfg.l.y / cfg.n.y) * (cfg.l.z / cfg.n.z);
	cfg.rho_k = e_q / h;
	cfg.charge_by_mass = e_q/e_m;
	cfg.epsilon = eps_0;
	//====//
	// Precomputed values for fft-solver, usage:
	// k^2 = kx^2 + ky^2 + kz^2
	// kx = kxt * i^2 etc.
	// Phi(k) = rho(k) * constant_factor/k^2
	cfg.solve.constant_factor = -1/(eps_0 * cfg.n.x * cfg.n.y * cfg.n.z);
	cfg.solve.kxt = 4 * pi*pi / (cfg.l.x * cfg.l.x);
	cfg.solve.kyt = 4 * pi*pi / (cfg.l.y * cfg.l.y);
	cfg.solve.kzt = 4 * pi*pi / (cfg.l.z * cfg.l.z);
	//====//
	cfg.trace_interval = (cfg.iterations-1)/2 + 1;
	//cfg.exec_cfg.tbp = dim3(256);
	cfg.exec_cfg.nbp = dim3(1 + (cfg.exec_cfg.tbp.x - 1) / cfg.particles);
	cfg.exec_cfg.nbg = dim3(
		1 + (cfg.n.x -1) / cfg.exec_cfg.tbg.x,
		1 + (cfg.n.y -1) / cfg.exec_cfg.tbg.y,
		1 + (cfg.n.z -1) / cfg.exec_cfg.tbg.z
	);
	cfg.exec_cfg.nbfreq = dim3(
		1 + (cfg.n.x -1) / cfg.exec_cfg.tbg.x,
		1 + (cfg.n.y -1) / cfg.exec_cfg.tbg.y,
		1 + cfg.n.z / (2 * cfg.exec_cfg.tbg.z)
	);
	cfg.exec_cfg.nbsor = dim3(
		cfg.exec_cfg.nbg.x,
		cfg.exec_cfg.nbg.y,
		1 + (cfg.n.z/2 -1) / cfg.exec_cfg.tbg.z
	);
	return cfg;
}