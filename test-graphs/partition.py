#!/usr/bin/python

#script to partition a graph
#this version does an ad-hoc (unoptimized) partition
#this is used only for correctness testing presently
#we will use metis or other algorithms in subsequent versions.

import sys
import graphutil


if __name__ == '__main__':
  try:
    inFilename, nNodes, outPrefix = sys.argv[1:]
  except ValueError:
    print('Usage: partition file.mtx numnodes outprefix')
    sys.exit(1)
    
  nNodes = int(nNodes)
  with open(inFilename) as f:
    header = graphutil.readMTXHeader(f)
    edges = list( graphutil.readEdgeListMTX(f, header) )
  nEdges = len(edges)
  nEdgesPerNode = (nEdges + nNodes - 1) // nNodes
  start = 0
  end   = 0
  for node in range(nNodes):
    start = end
    end   = min(nEdges, end + nEdgesPerNode)
    outFilename = '%s%d' % (outPrefix, node)
    graphutil.writeMTX(open(outFilename, 'w'), header, edges[start:end])
