#include "dependencies.cuh"

#ifndef DEFINE_TRACER_IMPLEMENTATION
#define DEFINE_TRACER_IMPLEMENTATION
template <class Object> class Tracer
{
private:
	size_t
		next,
		iterations,
		objects;
	Object *target;
	Object *data;
	std::ofstream *file;
	cudaStream_t stream;


public:
	Tracer(cudaStream_t stream, size_t p_iterations, size_t p_objects, Object *p_target, char *p_filename = "trace_particles"){
		this->stream = stream;
		this->iterations = p_iterations;
		this->objects = p_objects;
		this->next = 0;
		cudaChk(cudaMallocHost((void**)&data, p_iterations * p_objects * sizeof(Object)));
		target = p_target;

		std::stringstream fn = std::stringstream();
		fn << "trace" << p_filename << ".csv";

		file = new std::ofstream(fn.str(), std::ofstream::out);
	};

	~Tracer() {
		file->close();
		cudaFreeHost(data);
	};

	void appendTrace() {
		cudaChk(cudaMemcpyAsync(&data[next], target, objects * sizeof(Object), cudaMemcpyDeviceToHost, stream));
		next++;
	};

	void print() {
		for (size_t o = 0; o < objects; o++){
			*file << o << ":\t";
			for (size_t i = 0; i < iterations; i++){
				*file << data[i * objects + o].str() << ",    ";
			}
			*file << "\n\t";
			for (size_t i = 0; i < iterations; i++){
				*file << data[i * objects + o].str_v() << ",    ";
			}
			*file << "\n";
		}
	};
};
/*
template<class Object> Tracer<Object>::Tracer(size_t p_rows, size_t p_cols, Object* p_target, char* p_filename) {
	this->rows = p_rows;
	this->cols = p_cols;
	this->next = 0;
	data = new Object[p_rows*p_cols]; //(Object**)malloc(p_rows * p_cols * sizeof(Object));
	target = p_target;

	std::stringstream fn = std::stringstream();
	fn << "trace" << p_filename << ".csv";

	file = new std::ofstream(fn.str(), std::ofstream::out);
}

template<class Object> Tracer<Object>::~Tracer() {
	file->close();
	delete data;
}

template<class Object> void Tracer<Object>::appendTrace() {
	cudaMemcpy(&data[next], target, cols * sizeof(Object), cudaMemcpyDeviceToHost);
	next++;
}

template<> void Tracer<Particle>::print() {
	for (size_t p = 0; p < cols; p++){
		*file << p << ",\t";
		for (size_t i = 0; i < rows; i++){
			double3 pos = data[i * cols + p].position;
			*file << pos.x << ", " << pos.y << ", " << pos.z << "," << "\t";
		}
		*file << "\n";
	}
}*/
#endif