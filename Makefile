NVCC = nvcc

NVCC_OPTS = -O3 --restrict -Xptxas -dlcm=cg
NVCC_ARCHS = -gencode arch=compute_20,code=sm_20
LD_LIBS = -lz


#The rules need to be cleaned up, but we're probably going to use cmake, so
#just hacking it for now.

HEADERS = graphio.h util.h refgas.h

BINARIES = pagerank sssp bfs

all: $(BINARIES)

util.o: util.cpp util.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

graphio.o: graphio.cpp graphio.h Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS)

pagerank.o: pagerank.cu $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

pagerank: pagerank.o graphio.o util.o
	nvcc -o $@ $^ $(LD_LIBS)

sssp.o: sssp.cu $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

sssp: sssp.o graphio.o util.o
	nvcc -o $@ $^ $(LD_LIBS)

bfs.o: bfs.cu $(HEADERS) Makefile
	nvcc -c -o $@ $< $(NVCC_OPTS) $(NVCC_ARCHS) 

bfs: bfs.o graphio.o util.o
	nvcc -o $@ $^ $(LD_LIBS)

clean:
	rm -f $(BINARIES) *.o


