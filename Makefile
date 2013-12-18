NVCC       = nvcc
MGPU_PATH  = ../moderngpu
CUDA_PATH  = /usr/local/cuda
DEFINES    =
INCLUDES   = -I$(MGPU_PATH)/include -I$(CUDA_PATH)/include
LD_LIBDIRS = -L$(MGPU_PATH) -L.
#NVCC_OPTS  = -O3 --restrict -Xptxas -dlcm=cg
NVCC_OPTS  = -O3 -Xptxas -abi=no
#NVCC_OPTS  = -g
NVCC_ARCHS = -gencode arch=compute_20,code=sm_20 #-gencode arch=compute_30,code=sm_30
LD_LIBS    = -lz -lmgpu -lvertexAPI2
MAKEDEP    = g++ -MM -x c++

ifeq ($(MPI),1)
SUFFIX = _mpi
DEFINES += -DVERTEXAPI_USE_MPI
LD_LIBS += -lmpich
endif

LIB_SOURCES  = graphio util
LIB_OBJECTS := $(LIB_SOURCES:%=%$(SUFFIX).o)
LIB = libvertexAPI2$(SUFFIX).a

#each source in this list should have a main()
PROGRAM_SOURCES = outdeg createCCGraph pagerank sssp bfs connected_component
PROGRAM_OBJECTS = $(PROGRAM_SOURCES:%=%$(SUFFIX).o)
PROGRAMS        = $(PROGRAM_SOURCES:%=%$(SUFFIX))
ALL_SOURCES    := $(LIB_SOURCES) $(PROGRAM_SOURCES)

DEPDIR = .deps
DEPS   = $(ALL_SOURCES:%=.deps/%$(SUFFIX).dep)

all: $(PROGRAMS)

dep: $(DEPS)

$(DEPDIR):
	mkdir $(DEPDIR)

$(DEPDIR)/%$(SUFFIX).dep: %.cu | $(DEPDIR)
	$(MAKEDEP) -MT $*$(SUFFIX).o $(INCLUDES) $< -o $@

$(DEPDIR)/%$(SUFFIX).dep: %.cpp | $(DEPDIR)
	$(MAKEDEP) -MT $*$(SUFFIX).o $(INCLUDES) $< -o $@

-include $(DEPS)

%$(SUFFIX).o : %.cpp
	nvcc -c $(DEFINES) $(INCLUDES) $(NVCC_OPTS) $(NVCC_ARCHS) $< -o $@

%$(SUFFIX).o : %.cu
	nvcc -c $(DEFINES) $(INCLUDES) $(NVCC_OPTS) $(NVCC_ARCHS) $< -o $@

$(PROGRAMS): % : %.o $(LIB)
	nvcc -o $@ $(NVCC_OPTS) $(NVCC_ARCHS) $< $(LD_LIBDIRS) $(LD_LIBS)

$(LIB): $(LIB_OBJECTS)
	ar cruv $@ $^

obj-clean:
	rm -f $(LIB_OBJECTS) $(PROGRAM_OBJECTS)

clean: obj-clean
	rm -f $(LIB) $(PROGRAMS)

dep-clean:
	rm -f $(DEPS)
	rmdir $(DEPDIR)

all-clean: clean dep-clean
