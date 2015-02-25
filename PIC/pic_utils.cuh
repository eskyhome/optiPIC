#include "dependencies.cuh"
//====//
#ifndef PIC_UTILS_DEFINED
#define PIC_UTILS_DEFINED

#define cudaChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stdout, "CUDA error: %s in file %s, line %d\n", cudaGetErrorString(code), file, line);
	}
}

static char *mkString[9] =
{
	"CUFFT_SUCCESS",
	"CUFFT_INVALID_PLAN",
	"CUFFT_ALLOC_FAILED",
	"CUFFT_INVALID_TYPE",
	"CUFFT_INVALID_VALUE",
	"CUFFT_INTERNAL_ERROR",
	"CUFFT_EXEC_FAILED",
	"CUFFT_SETUP_FAILED",
	"CUFFT_INVALID_SIZE"
};
#define cufftChk(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult_t code, const char *file, int line, bool abort = true) {
	if (code != CUFFT_SUCCESS) {
		fprintf(stdout, "CUFFT error: %s in file %s, line %d\n", mkString[code], file, line);
	}
}

#define errCheck() {kernelErrCheck(__FILE__, __LINE__);}
inline void kernelErrCheck(const char *f, int l){
	cudaError_t cudaErrCheckResult = cudaPeekAtLastError();
	cudaAssert(cudaErrCheckResult, f, l);
	cudaErrCheckResult = cudaDeviceSynchronize();
	cudaAssert(cudaErrCheckResult, f, l);
}
//====//
#endif