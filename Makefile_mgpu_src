INCLUDE_DIR=../include
NVCC_OPTS=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -O3

libmgpu.a: sparsematrix.o mgpucontext.o mgpuutil.o mmio.o
	ar rcs $@ $^
	cp libmgpu.a ../

sparsematrix.o: sparsematrix.cpp
	nvcc -c $(NVCC_OPTS) $< -I $(INCLUDE_DIR)

mgpucontext.o: mgpucontext.cu
	nvcc -c $(NVCC_OPTS) $< -I $(INCLUDE_DIR)

mgpuutil.o: mgpuutil.cpp
	nvcc -c $(NVCC_OPTS) $< -I $(INCLUDE_DIR)

mmio.o: mmio.cpp
	nvcc -c $(NVCC_OPTS) $< -I $(INCLUDE_DIR)

clean:
	rm *.o libmgpu.a
