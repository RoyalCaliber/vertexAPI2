#!/usr/bin/python3
#Generates a summary of results and timings

import glob
import itertools

def timingsDict(algos, graphs):
  return dict((x, dict((y,None) for y in graphs)) for x in algos) 


def readTimings(algo, graph, ext):
  fn = '%s.%s.%s' % (graph, algo, ext)  
  try:
    return float(open(fn).read())
  except(IOError, ValueError):
    return -1


def printSummary(algos, graphs, cpuTimings, gpuTimings):
  for algo in algos:
    print("---------------------------------------------------")
    print("Algorithm: %s" % algo)
    print("---------------------------------------------------")
    for graph in graphs:
      print("%-20.20s  %8.1f  %8.1f  %8.1fx" % (graph
        , cpuTimings[algo][graph], gpuTimings[algo][graph]
        , cpuTimings[algo][graph] / gpuTimings[algo][graph]))
    print("---------------------------------------------------")
      

cpuTimingFiles = glob.glob('*.timing')
gpuTimingFiles = glob.glob('*.timing_gpu')
graphs = set(x.split('.')[0] for x in cpuTimingFiles + gpuTimingFiles)
algos  = set(x.split('.')[1] for x in cpuTimingFiles + gpuTimingFiles)
cpuTimings = timingsDict(algos, graphs)
gpuTimings = timingsDict(algos, graphs)

for algo, graph in itertools.product(algos, graphs):
  cpuTimings[algo][graph] = readTimings(algo, graph, 'timing')
  gpuTimings[algo][graph] = readTimings(algo, graph, 'timing_gpu')

printSummary(algos, graphs, cpuTimings, gpuTimings)
