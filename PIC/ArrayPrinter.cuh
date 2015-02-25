#include "dependencies.cuh"

#ifndef PIC_AP_DEFINED
#define IPC_AP_DEFINED
template <typename type>class ArrayPrinter{
	type
		*target,
		*data;
	size_t
		next,
		iterations,
		width,
		height,
		depth,
		size;
	std::ofstream *file;


public:
	ArrayPrinter(size_t iterations, size_t width, size_t height, size_t depth, type* target, char* filename = "debug_field"){
		this->iterations = iterations;
		this->width = width;
		this->height = height;
		this->depth = depth;
		this->size = height * width * depth;
		this->data = new type[iterations * size * sizeof(type)]; //(Object**)malloc(p_rows * p_cols * sizeof(Object));
		this->target = target;
		this->next = 0;

		std::stringstream fn = std::stringstream();
		fn << "debug_" << filename << ".csv";

		this->file = new std::ofstream(fn.str(), std::ofstream::out);
	};

	~ArrayPrinter() {
		file->close();
		delete data;
	};

	void appendValues() {
		cudaMemcpy(&data[next], target, size * sizeof(type), cudaMemcpyDeviceToHost);
		next += size;
	};

	void print() {
		for (size_t iter = 0; iter < iterations; iter++) {
			*file << "iteration: " << iter << "\n";
			for (size_t i = 0; i < depth; i++){
				*file << "\tdepth: " << i << "\n";
				for (size_t j = 0; j < height; j++){
					*file << "\t\t";
					for (size_t k = 0; k < width; k++){
						*file << std::setprecision(5) << std::setw(12) << data[iter * size + (i * height + j) * width + k] << ";\t";
					}
					*file << "\n";
				}
				*file << "\n";
			}
			*file << "\n";
		}
	};
};

void ArrayPrinter<double4>::print() {
	for (size_t iter = 0; iter < iterations; iter++) {
		*file << "iteration: " << iter << "\n";
		for (size_t i = 0; i < depth; i++){
			*file << "\tdepth: " << i << "\n";
			for (size_t j = 0; j < height; j++){
				*file << "\t\t";
				for (size_t k = 0; k < width; k++){
					*file << std::setprecision(5) << std::setw(12) << data[iter * size + (i * height + j) * width + k].x << ", ";
					*file << std::setprecision(5) << std::setw(12) << data[iter * size + (i * height + j) * width + k].y << ", ";
					*file << std::setprecision(5) << std::setw(12) << data[iter * size + (i * height + j) * width + k].z << ";\t";
				}
				*file << "\n";
			}
			*file << "\n";
		}
		*file << "\n";
	}
}
#endif