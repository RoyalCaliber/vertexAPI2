############################################################################
#Copyright 2013 SYSTAP, LLC. http://www.systap.com
#
#Written by Erich Elsen and Vishal Vaidyanathan
#of Royal Caliber, LLC
#Contact us at: info@royal-caliber.com
#
#This file was taken from mpgraph v0.1 which was (partially) funded by the
#DARPA XDATA program under AFRL Contract #FA8750-13-C-0002.  The file has
#been modified by Royal Caliber, LLC.
#
#Copyright 2013, 2014 Royal Caliber LLC. (http://www.royal-caliber.com)
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
###########################################################################



GRAPH_DIR = ../test-graphs

#Can restrict this to the graphs you have downloaded
GRAPHS = ak2010 belgium_osm delaunay_n13 coAuthorsDBLP delaunay_n21 webbase-1M soc-LiveJournal1 #kron_g500-logn21

#which algorithms to check
ALGORITHMS = pagerank sssp bfs connected_component

#location of reference implementations
PG_REF_DIR = ../PowerGraphReferenceImplementations

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
  $(PG_REF_DIR)/$(2).x $(POWERGRAPH_OPTS) --graph $(GRAPH_DIR)/$(1)/$(1).mtx --graph_opts ingress=batch --save __$(1).$(2).out | awk '/Finished Running/{print $$5}' > $(1).$(2).timing ; 
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

%.connected_component.gold: $(PG_REF_DIR)/connected_component.x
	$(call MAKEGOLD,$*,connected_component)

%.connected_component.test: ../connected_component
	../connected_component $(GRAPH_DIR)/$*/$*.mtx __tmp$* | awk '/Took/{print $$2}' > $*.connected_component.timing_gpu
	sort -n __tmp$* > $@
	rm -f __tmp$*

%.connected_component.pass: %.connected_component.test %.connected_component.gold
	./checkDist.py $*.connected_component.test $*.connected_component.gold
	touch $*.connected_component.pass

clean:
	rm -f *.test *.timing_gpu *.pass

clean-gold:
	rm -f *.gold *.timing

clean-all: clean clean-gold
