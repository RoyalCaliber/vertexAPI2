GRAPH_DIR = ../test-graphs

#Can restrict this to the graphs you have downloaded
GRAPHS = ak2010 belgium_osm delaunay_n13 coAuthorsDBLP delaunay_n21 webbase-1M #soc-LiveJournal1 kron_g500-logn21

#which algorithms to check
ALGORITHMS = pagerank #sssp bfs

#location of reference implementations
PG_REF_DIR = ../PowerGraphReferenceImplementations

#Match this to number of cores on your system
POWERGRAPH_OPTS = --ncpus 4

#Number of MPI processes
#We should ideally autoregress over multiple values of this, right now fixed
#if you change this, you should also 'make clean-parts'
MPI_NUM_NODES = 4

GOLD_BINARIES = $(foreach x,$(ALGORITHMS),../PowerGraphReferenceImplementations/$x.x)
GOLD_FILES = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.gold))
TEST_FILES = $(GOLD_FILES:.gold=.test)
REGRESSIONS = $(TEST_FILES:.test=.pass)

all: regress

gold: $(GOLD_BINARIES) $(GOLD_FILES)

test: $(TEST_FILES)

regress: $(REGRESSIONS)

define MAKEGOLD
  rm -f __$(1).$(2).out* ;
  $(PG_REF_DIR)/$(2).x $(POWERGRAPH_OPTS) --graph $(GRAPH_DIR)/$(1)/$(1).mtx --graph_opts ingress=batch --save __$(1).$(2).out | awk '/Finished Running/{print $$5}' > $(1).$(2).timing ; 
  cat __$(1).$(2).out* | sort -n > $(1).$(2).gold ;
  rm -f __$(1).$(2).out* ;
endef

.SECONDARY:
.DELETE_ON_ERROR:

#this dependency is missing because gnu make doesn't like 
#%.deg: $(GRAPH_DIR)/%/%.mtx
%.deg:
	$(GRAPH_DIR)/outdeg.py $(GRAPH_DIR)/$*/$*.mtx $@
	$(GRAPH_DIR)/partition.py $(GRAPH_DIR)/$*/$*.mtx $(MPI_NUM_NODES) $*.part.mtx_$(MPI_NUM_NODES)_

%.pagerank.gold: $(PG_REF_DIR)/pagerank.x
	$(call MAKEGOLD,$*,pagerank)

%.pagerank.test: ../pagerank %.deg
	mpirun -n $(MPI_NUM_NODES) ../pagerank $*.part.mtx $*.deg __tmp$* | awk '/Took/{print $$2}' > $*.pagerank.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.pagerank.pass: %.pagerank.test %.pagerank.gold
	./checkPageRank.py $*.pagerank.test $*.pagerank.gold
	touch $*.pagerank.pass

%.sssp.gold: $(PG_REF_DIR)/sssp.x
	$(call MAKEGOLD,$*,sssp)

%.sssp.test: ../sssp
	../sssp -m $(GRAPH_DIR)/$*/$*.mtx 0 __tmp$* | awk '/Took/{print $$2}' > $*.sssp.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.sssp.pass: %.sssp.test %.sssp.gold
	./checkDist.py $*.sssp.test $*.sssp.gold
	touch $*.sssp.pass

%.bfs.gold: $(PG_REF_DIR)/bfs.x
	$(call MAKEGOLD,$*,bfs)

%.bfs.test: ../bfs
	../bfs -m $(GRAPH_DIR)/$*/$*.mtx 0 __tmp$* | awk '/Took/{print $$2}' > $*.bfs.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.bfs.pass: %.bfs.test %.bfs.gold
	./checkDist.py $*.bfs.test $*.bfs.gold
	touch $*.bfs.pass

clean:
	rm -f *.test *.timing_gpu *.pass 

clean-parts:
	rm -f *.deg *.part.mtx*

clean-gold:
	rm -f *.gold *.timing

clean-timings:
	rm -f *.timing *.gpu_timing

clean-all: clean clean-gold clean-parts clean-timings
