#!/usr/bin/python

#writes out a list of out-degrees of all vertices in the graph
#this is used as auxiliary input to pagerank and bfs.

import sys
import graphutil

try:
  inFilename, outFilename = sys.argv[1:]
except ValueError:
  print('Usage: outdeg graph.mtx output')
  sys.exit(1)

inFile  = open(inFilename)
header  = graphutil.readMTXHeader(inFile)
edges   = list(graphutil.readEdgeListMTX(inFile, header))
nVerts  = max(max(e[0], e[1]) for e in edges) + 1
dsts, v = graphutil.edgeListToCSR(edges)
outFile = open(outFilename, 'w')
for src in range(nVerts):
  outFile.write('%d\n' % len(dsts[src]))

