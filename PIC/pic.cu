#include "dependencies.cuh"
#include "structs.cuh"
#include "simplified_kernels.cuh"
#include "pic_utils.cuh"
#include "cfg.cuh"
#include "Tracer.cuh"
#include "ArrayPrinter.cuh"
#include "Particle.cuh"

//#define TRACE

namespace pic{
	double* d_rhophi;
	double4* d_efreq;
	Particle* d_particles;

	cufftResult_t setupCufft(cudaStream_t streamCalc, cufftHandle* plan, cufftHandle* iplan, Config cfg) {
		cufftChk(cufftCreate(plan));
		cufftResult_t res = cufftPlan3d(plan, cfg.n.z, cfg.n.y, cfg.n.x, CUFFT_D2Z);
		cufftSetStream(*plan, streamCalc);

		cufftChk(cufftCreate(iplan));
		cufftResult_t ires = cufftPlan3d(iplan, cfg.n.w, cfg.n.y, cfg.n.x, CUFFT_Z2D);
		cufftSetStream(*iplan, streamCalc);

		//#define PIC_CUFFT_CALLBACK_SETUP
#ifdef PIC_CUFFT_CALLBACK_SETUP
		__device__ void (*device_solver_callback_ptr)(cufftDoubleComplex* freq, Config cfg) = fftsolve;
		void (*host_solver_callback_ptr)(cufftDoubleComplex* freq, Config cfg);
		cudaMemcpyFromSymbol(&host_solver_callback_ptr, device_solver_callback_ptr, sizeof(host_solver_callback_ptr));
		cufftXtSetCallback(host_solver_callback_ptr);
#endif

		if (res != CUFFT_SUCCESS) {
			cufftChk(res);
			cufftChk(ires);
			cufftChk(cufftDestroy(*plan));
			cufftChk(cufftDestroy(*iplan));
			return res;
		}
		else if (ires != CUFFT_SUCCESS){
			cufftChk(res);
			cufftChk(ires);
			cufftChk(cufftDestroy(*plan));
			cufftChk(cufftDestroy(*iplan));
			return ires;
		}

		return CUFFT_SUCCESS;
	}

	void fftsolver_host(Config cfg, cufftHandle plan, cufftHandle iplan){
		cufftChk(cufftExecD2Z(plan, (cufftDoubleReal *)d_rhophi, (cufftDoubleComplex *)d_efreq));
		fftsolver_kernel <<<cfg.exec_cfg.nbfreq, cfg.exec_cfg.tbg >>>((cufftDoubleComplex *)d_efreq, cfg);
		errCheck();
		cudaChk(cudaMemset(d_rhophi, 0, cfg.n.x * cfg.n.y * cfg.n.z * sizeof(double)));
		cufftChk(cufftExecZ2D(iplan, (cufftDoubleComplex *)d_efreq, (cufftDoubleReal *)d_rhophi));
	}

	void sorsolver_host(cudaStream_t streamCalc, Config cfg){
		sorinit_kernel << <cfg.exec_cfg.nbg, cfg.exec_cfg.tbg, 0, streamCalc >> >(d_rhophi, d_rhophi, cfg);
		errCheck();
		size_t shared_size = 2 * cfg.exec_cfg.tbg.x * 2 * cfg.exec_cfg.tbg.y * 2 * cfg.exec_cfg.tbg.z * sizeof(double);

		for (int it_sor = 0; it_sor < cfg.sor_iterations; it_sor++) {
			sorsolver_shared << <cfg.exec_cfg.nbg, cfg.exec_cfg.tbg, shared_size, streamCalc >> >(d_rhophi, cfg, 0);
			errCheck();
			//sorsolver_kernel << <cfg.exec_cfg.nbsor, cfg.exec_cfg.tbg, shared_size, streamCalc >> >(d_rhophi, cfg, 1);
			//errCheck();
		}
	}

	void initParticles(Particle* d_particles, Config cfg) {
		Particle* h_particles = new Particle[cfg.particles];


		for (int i = 0; i < cfg.particles; i++) {
			h_particles[i] = Particle(cfg);
		}
		cudaMemcpy(d_particles, h_particles, cfg.particles * sizeof(Particle), cudaMemcpyHostToDevice);
		delete h_particles;
	}

	int run(Config cfg, bool fft = false) {
		//setup
		cufftHandle plan, iplan;

		//Create streams
		cudaStream_t streamCalc, streamTrace;
		cudaChk(cudaStreamCreate(&streamCalc));
		cudaChk(cudaStreamCreate(&streamTrace));

		if (setupCufft(streamCalc, &plan, &iplan, cfg) != CUFFT_SUCCESS)
			return EXIT_FAILURE;
		
		//Allocate arrays
		cudaChk(cudaMalloc(&d_particles, cfg.particles * sizeof(Particle)));
		initParticles(d_particles, cfg);
		Tracer<Particle> *tf = new Tracer<Particle>(streamTrace, (cfg.iterations + 1), cfg.particles, d_particles, "_particles");
		tf->appendTrace();

		size_t
			rhophiSize = cfg.n.x * cfg.n.y * cfg.n.z * sizeof(double),
			efreqSize = cfg.n.x * cfg.n.y * cfg.n.z * sizeof(double4);
		cudaChk(cudaMalloc(&d_rhophi, rhophiSize));
		cudaChk(cudaMalloc(&d_efreq, efreqSize));
		ArrayPrinter<double> *rhophiPrinter = new ArrayPrinter<double>(cfg.iterations, cfg.n.x, cfg.n.y, cfg.n.z, d_rhophi, "rhopi");
		ArrayPrinter<double4> *efreqPrinter = new ArrayPrinter<double4>(cfg.iterations, cfg.n.x, cfg.n.y, cfg.n.z, d_efreq, "efreq");

		//loop
		cudaChk(cudaMemset(d_efreq, 0, efreqSize));
		int iteration = 0;
		while (iteration < cfg.iterations) {

			cudaStreamSynchronize(streamTrace);
			cudaChk(cudaMemset(d_rhophi, 0, rhophiSize));
			particleUpdate_kernel << <cfg.exec_cfg.nbp, cfg.exec_cfg.tbp, 0, streamCalc >> >(d_particles, d_efreq, d_rhophi, cfg); errCheck();
			
			cudaStreamSynchronize(streamCalc);
			tf->appendTrace(); errCheck();	

			rhophiPrinter->appendValues(); errCheck();
						
			cudaChk(cudaMemset(d_efreq, 0, efreqSize));
			if (fft)
				fftsolver_host(cfg, plan, iplan);
			else
				sorsolver_host(streamCalc, cfg);

			cudaChk(cudaMemset(d_efreq, 0, efreqSize));
			electricfield_kernel << <cfg.exec_cfg.nbg, cfg.exec_cfg.tbg, 0, streamCalc >> >(d_rhophi, d_efreq, cfg); errCheck();

			efreqPrinter->appendValues(); errCheck();
	
			iteration++;
		}
		tf->print();
		rhophiPrinter->print();
		efreqPrinter->print();
		delete tf;
		delete rhophiPrinter, efreqPrinter;

		//clean
		cudaChk(cudaFree(d_rhophi));
		cudaChk(cudaFree(d_efreq));
		cudaChk(cudaFree(d_particles));
		cufftChk(cufftDestroy(plan));
		cufftChk(cufftDestroy(iplan));

		return EXIT_SUCCESS;
	}
}

int main(int argc, const char* argv[]){
	Config cfg = getConfig();
#ifndef TRACE
	cfg.iterations = 4;
#endif
	int res = pic::run(cfg, false);
	return res;
}
