GRAPH_DIR = ../../largePerformanceGraphs

#Can restrict this to the graphs you have downloaded
GRAPHS = ak2010 belgium_osm delaunay_n13 coAuthorsDBLP delaunay_n21 webbase-1M #soc-LiveJournal1 kron_g500-logn21

#which algorithms to check
ALGORITHMS = pagerank #sssp bfs

#location of reference implementations
PG_REF_DIR = ../../vertexAPI/PowerGraphReferenceImplementations

#Match this to number of cores on your system
POWERGRAPH_OPTS = --ncpus 4

GOLD_BINARIES = $(foreach x,$(ALGORITHMS),../PowerGraphReferenceImplementations/$x.x)
GOLD_FILES = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.gold))
TEST_FILES = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.test))

REGRESSIONS = $(foreach P,$(ALGORITHMS),$(foreach G,$(GRAPHS),$G.$P.pass))

all: regress

gold: $(GOLD_BINARIES) $(GOLD_FILES)

test: $(TEST_FILES)

regress: $(REGRESSIONS)

define MAKEGOLD
  rm -f __$(1).$(2).out* ;
  $(PG_REF_DIR)/$(2).x --graph $(GRAPH_DIR)/$(1)/$(1).mtx --graph_opts ingress=batch --save __$(1).$(2).out | awk '/Finished Running/{print $$5}' > $(1).$(2).timing ; 
  cat __$(1).$(2).out* | sort -n > $(1).$(2).gold ;
  rm -f __$(1).$(2).out* ;
endef

%.pagerank.gold: $(PG_REF_DIR)/pagerank.x
	$(call MAKEGOLD,$*,pagerank)

%.pagerank.test: ../pagerank
	../pagerank $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.pagerank.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.pagerank.pass: %.pagerank.test %.pagerank.gold
	./checkPageRank.py $*.pagerank.test $*.pagerank.gold
	touch $*.pagerank.pass

%.sssp.gold: $(PG_REF_DIR)/sssp.x
	$(call MAKEGOLD,$*,sssp)

%.sssp.test: ../sssp
	../sssp $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.sssp.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.sssp.pass: %.sssp.test %.sssp.gold
	diff -q $*.sssp.test $*.sssp.gold
	touch $*.sssp.pass

%.bfs.gold: $(PG_REF_DIR)/bfs.x
	$(call MAKEGOLD,$*,bfs)

%.bfs.test: ../bfs
	../bfs $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.bfs.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.bfs.pass: %.bfs.test %.bfs.gold
	diff -q $*.bfs.test $*.bfs.gold
	touch $*.bfs.pass

clean:
	rm -f *.test *.timing_gpu *.pass

clean-gold:
	rm -f *.gold *.timing

clean-all: clean clean-gold
