BIN := 

CUDA_INSTALL_PATH ?= /cuda path/
CUDA_SDK_PATH ?= $(HOME)/NVIDIA_GPU_Computing_SDK

NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" - I"$(CUDA_INSTALL_PATH)/include" -I"./"
LIBS = -L"/libcuda path/" -lcuda -L"$(CUDA_INSTALL_PATH)/lib64" -lcudart -lcublas -lcufft -L"$(CUDA_SDK_PATH)/C/common/lib" $(CUDA_SDK_PATH)/C/lib/libcutil$(LIBSUFFIX).a -lstdc++ -lpthread

CUDA_SDK ?=  6.5
COMMONFLAGS = -DCUDA_SDK=$(CUDA_SDK)
CXXFLAGS		:= -O3 -g
NVCCFLAGS		:= --ptxas-options=-v -O3 -G -g -arch sm_52

#files
CPP_SOURCES		:= 
CU_SOURCES		:= Main.cu, kernels_3d.cu, utils.cu
HEADERS			:= Particle.cuh
CPP_OBJS		:= $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS			:= $(patsubst %.cu, %.cu_o, $(CU_SOURCES))



%.o : %.cpp
$(CXX) -c $(CXXFLAGS) $(INCD) -o %@ $<

%.cu_o : %.cpp
$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(COMMONFLAGS) $(INCD) $(LIBS)

clean: rm -f $(BIN) *.o *.cu_o