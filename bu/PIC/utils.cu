#include "Particles.cuh"
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>


void allocateMemory(Particle **particles, cudaPitchedPtr *chargeDensity, cudaPitchedPtr *potential, cudaPitchedPtr *E, cudaPitchedPtr *freq, Config cfg) {
	cudaChk( cudaMalloc(particles, sizeof(Particle) * cfg.particles));
	cudaChk( cudaMallocPitch(&(*chargeDensity).ptr, &(*chargeDensity).pitch, (*chargeDensity).xsize, (*chargeDensity).ysize));
	cudaChk( cudaMallocPitch(&(*potential).ptr, &(*potential).pitch, (*potential).xsize, (*potential).ysize));
	cudaChk( cudaMallocPitch(&(*E).ptr, &(*E).pitch, (*E).xsize, (*E).ysize));
	cudaChk( cudaMallocPitch(&(*freq).ptr, &(*freq).pitch, (*freq).xsize, (*freq).ysize));
	cudaChk( cudaGetLastError());
}
void allocateMemory3D(Particle **particles, cudaPitchedPtr *chargeDensity, cudaPitchedPtr *potential, cudaPitchedPtr *E, cudaPitchedPtr *freq, Config cfg) {
	cudaExtent
		cd_ext =	make_cudaExtent(cfg.n.x * sizeof(double),				cfg.n.y, cfg.n.z),
		pot_ext =	make_cudaExtent(cfg.n.x * sizeof(double),				cfg.n.y, cfg.n.z),
		E_ext =		make_cudaExtent(cfg.n.x * sizeof(double2),				cfg.n.y, cfg.n.z),
		freq_ext =	make_cudaExtent(cfg.n.x * sizeof(cufftDoubleComplex),	cfg.n.y, cfg.n.w);

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
	cudaChk( cudaFree(freq.ptr));
}

//TODO: print metadata
void printStatistics(char *filename, Particle *pta, Config cfg) {
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i <= cfg.iterations; i++) {
		file << "<iteration time=" << i*cfg.ts << " >\n";
		for (int j = 0; j < cfg.particles; j++) {
			Particle p = *((Particle*)((char*)pta + j*cfg.particles + i*sizeof(Particle)));
			file << "\t<particle id=" << j << " >\n"
				<< "\t\t<position x=" << p.position.x << " y=" << p.position.y << " z=" << p.position.z << " >\n"
				<< "\t\t<velocity x=" << p.velocity.x << " y=" << p.velocity.y << " z=" << p.velocity.z <<  " >\n"
				<< "\t\t<electric x=" << p.eAtP.x << " y=" << p.eAtP.y << " z=" << p.eAtP.z <<  " >\n"
				<< "\t</particle>\n";
		}
		file << "</iteration>\n";
	}
	file.close();
}

void printField(char *fieldname, double *field, int i, Config cfg){
	char title[32];
	sprintf_s(title, "%s%d.txt", fieldname, i);

	std::ofstream file;
	file.open(title);
	file << "Iteration " << i <<"\n";
	for (int j = 0; j < cfg.n.y; j++) {
		file << cfg.l.y - (j + 1)*cfg.l.y/cfg.n.y << "\t|\t\t";
		for (int i = 0; i < cfg.n.x; i++) {
			std::ostringstream strs;
			strs << std::setprecision(3) << field[i + cfg.n.x * (cfg.n.y - j - 1)];
			std::string s = strs.str();
			file << strs.str() << std::string(3 - s.length()/4, '\t');
		}
		file << "\n";
	}
	file << "y__\t|" << std::string(cfg.n.x * 12, '_') << "\n/ x";
	for (int i = 0; i < cfg.n.x; i++) {
		file << "\t\t\t" << i*cfg.l.x/cfg.n.x;
	}
	file.close();
}

void printFreq(char *fieldname, cufftDoubleComplex *field, int i, Config cfg) {
	char title[32];
	sprintf_s(title, "%s%d.txt", fieldname, i);

	std::ofstream file;
	file.open(title);
	file << "Iteration " << i <<"\n";
	for (int i = 0; i < cfg.n.x; i++) {
		for (int j = 0; j < cfg.n.w; j++) {
			//file << "\t[x:" << std::setprecision(3) << field[i + N_X * j].x << ", y:" << field[i + N_X * j].y << "],";
			file << "\t[" << std::setprecision(5) << field[i + cfg.n.x * j].x << "," << std::setprecision(5) << field[i + cfg.n.x * j].y << "]";
		}
		file << "\n";
	}
	file.close();
}

Config getConfig() {
	double
		pi = 3.14159265359,
		eps_0 = 8.854187817e-12,
		e_q = 1.60217657e-19,
		e_m = 9.10938291e-31;

	Config cfg =
	{
		make_int4(16, 4, 4, 0),
		make_double3(1, 1, 1),
		0, 0, 0,
		1e-1,

		1.78,
		1e-9,
		10,
		12,
		{
			dim3(1, 1, 1),
			dim3(1, 1, 1),
			dim3(16, 4, 4),
			dim3(1, 1, 1),
			dim3(16, 4, 4),
			dim3(1, 1, 1)
		}
	};
	cfg.n.w = cfg.n.z/2 + 1;
	double h = cfg.l.x / cfg.n.x * cfg.l.y / cfg.n.y * cfg.l.z / cfg.n.z;
	cfg.rho_k = e_q / h;
	cfg.charge_by_mass = e_q/e_m;
	cfg.solve_factor = 1/(eps_0 * 4*pi*pi * ((cfg.l.x*cfg.l.x + cfg.l.y*cfg.l.y + cfg.l.z*cfg.l.z)/h));
	//====//
	cfg.exec_cfg.tbp = dim3(256);
	cfg.exec_cfg.nbp = dim3(1 + (cfg.exec_cfg.tbp.x - 1) / cfg.particles);
	cfg.exec_cfg.nbf = dim3(
		1 + (cfg.n.x -1) / cfg.exec_cfg.tbf.x,
		1 + (cfg.n.y -1) / cfg.exec_cfg.tbf.y,
		1 + (cfg.n.z -1) / cfg.exec_cfg.tbf.z
	);
	cfg.exec_cfg.nbfreq = dim3(
		1 + (cfg.n.x -1) / cfg.exec_cfg.tbf.x,
		1 + (cfg.n.y -1) / cfg.exec_cfg.tbf.y,
		1 + cfg.n.z / (2 * cfg.exec_cfg.tbf.z)
	);
	return cfg;
}