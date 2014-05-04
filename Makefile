NVCC = nvcc
MGPU_PATH = ../moderngpu
#NVCC_OPTS = -O3 --restrict -Xptxas -dlcm=cg -I$(MGPU_PATH)/include -L$(MGPU_PATH)
NVCC_OPTS = -O3 -Xptxas -abi=no -I$(MGPU_PATH)/include -L$(MGPU_PATH)
NVCC_ARCHS = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
#NVCC_ARCHS = -gencode arch=compute_20,code=sm_20
LD_LIBS = -lz -lmgpu 


#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

HEADERS = graphio.h util.h refgas.h gpugas.h gpugas_kernels.cuh

BINARIES = pagerank sssp bfs connected_component createCCGraph mtx2gr gr2mtx

all: $(BINARIES) libvertexAPI2.a

util.o: util.cpp util.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

pagerank.o: pagerank.cu primitives/scatter_if_mgpu.h $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

pagerank: pagerank.o graphio.o util.o
	nvcc $(NVCC_OPTS) $(NVCC_ARCHS) -o $@ $^ $(LD_LIBS)

sssp.o: sssp.cu primitives/scatter_if_mgpu.h $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

sssp: sssp.o graphio.o util.o
	nvcc $(NVCC_OPTS) $(NVCC_ARCHS) -o $@ $^ $(LD_LIBS)

bfs.o: bfs.cu primitives/scatter_if_mgpu.h $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

bfs: bfs.o graphio.o util.o
	nvcc $(NVCC_OPTS) $(NVCC_ARCHS) -o $@ $^ $(LD_LIBS)

connected_component.o: connected_component.cu primitives/scatter_if_mgpu.h $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

connected_component: connected_component.o graphio.o util.o
	nvcc $(NVCC_OPTS) $(NVCC_ARCHS) -o $@ $^ $(LD_LIBS)

createCCGraph: createCCGraph.cpp graphio.o
	g++ -o $@ $< -I . graphio.o -lz

mtx2gr: mtx2gr.cpp graphio.o util.o
	g++ -O3 -o mtx2gr mtx2gr.cpp -I. graphio.o util.o -lz

gr2mtx: gr2mtx.cpp graphio.o util.o
	g++ -O3 -o gr2mtx gr2mtx.cpp -I. graphio.o util.o -lz

clean:
	rm -f $(BINARIES) *.o libvertexAPI2.a

libvertexAPI2.a: graphio.o util.o
	ar cruv $@ $^ 

