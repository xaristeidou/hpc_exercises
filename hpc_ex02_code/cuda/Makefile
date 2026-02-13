NVCC = nvcc
NVCCFLAGS = -O2 -lm

all: complex_matrix_mul complex_matrix_mul_cublas

complex_matrix_mul: complex_matrix_mul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

complex_matrix_mul_cublas: complex_matrix_mul_cublas.cu
	$(NVCC) $(NVCCFLAGS) -lcublas -o $@ $<

clean:
	rm -f complex_matrix_mul complex_matrix_mul_cublas

.PHONY: all clean
