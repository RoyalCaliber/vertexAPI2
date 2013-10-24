#Makefile for generating regression golds.  This will all be converted to cmake
#later.

#Set this to your local installation of graphlab
GRAPHLAB_DIR = /home/vishal/compile/graphlabapi
#GRAPHLAB_DIR = /home/erich/graphlabapi


#We need a nice way to import the compile and link flags from the graphlab
#build.  Until then we are using a horrible method to build the reference
#executables - we copy each cpp file to a temporary project directory
#within the graphlab directory, build it there and copy the executable back

#INCLUDES = -I$(GRAPHLAB_DIR)/src -I$(GRAPHLAB_DIR)/deps/local/include

#This may need to be tweaked for your system
#CFLAGS = -O3 -g3 -ggdb -march=native -mtune=native  -Wall -g  -fopenmp -DHAS_BUILTIN_CRC32

#LDFLAGS = -g3 -ggdb -L$(GRAPHLAB_DIR)/release/src/graphlab -L$(GRAPHLAB_DIR)/deps/local/lib

#LDLIBS = -lgraphlab -lboost_filesystem -lboost_program_options -lboost_system \
  -lboost_iostreams -lboost_date_time -lz -lpthread -ltcmalloc -levent \
  -levent_pthreads -lmpichcxx -lmpich -ljson -lgomp
  
ALGORITHMS = pagerank sssp sssp_gather bfs

BINARIES = $(foreach x,$(ALGORITHMS),$x.x)

#OBJECTS = $(foreach x,$(BINARIES),$x.o)

all: $(BINARIES)


#%.o: %.cpp
#	g++ $(INCLUDES) $(CFLAGS) -c -o $@ $< 

%.x: %.cpp
	$(eval APPDIR = $(GRAPHLAB_DIR)/demoapps/tmp_$*)
	$(eval APPBUILDDIR = $(GRAPHLAB_DIR)/release/demoapps/tmp_$*)
	mkdir -p $(APPDIR)
	cp $< ../graphio.cpp ../graphio.h $(APPDIR)/
	printf 'project(GraphLab)\nadd_graphlab_executable(tmp_$@ $< graphio.cpp)\n' > $(APPDIR)/CMakeLists.txt
	cd $(GRAPHLAB_DIR) && ./configure
	cd $(APPBUILDDIR) && make
	cp $(APPBUILDDIR)/tmp_$@ $@
	rm $(APPDIR)/CMakeLists.txt $(APPDIR)/$< $(APPDIR)/graphio.cpp $(APPDIR)/graphio.h
	rmdir $(APPDIR)

#pagerank: pagerank.cpp
#	g++ $(LDFLAGS)  pagerank.o -o pagerank $(LDLIBS)

clean:
	rm -f $(BINARIES)

.PHONY: all clean
