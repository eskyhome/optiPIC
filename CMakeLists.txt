cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_52,code=sm_52)

include_directories(
	CUDA_INC_PATH
)

link_directories(
	/link/path/to/lib
	cuda.lib
	cudart.lib
)

cuda_add_executable(
	pic
	pic/Main.cu
	pic/utils.cu
	pic/kernels3d.cu
	pic/Particles.cuh
)

target_link_libraries(
	pic
	
)