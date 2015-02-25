#include "dependencies.cuh"
#include "structs.cuh"
#include "Tracer.cuh"

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

template<class Object> Tracer<Object>::~Tracer(){
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
}


//Print tracefiles and clean up.
*/