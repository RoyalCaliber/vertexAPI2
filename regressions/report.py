#!/usr/bin/python3
#Generates a summary of results and timings

import glob
import itertools
import sys

def timingsDict(algos, graphs):
  return dict((x, dict((y,None) for y in graphs)) for x in algos) 

def gpuTimingsDict(algos, graphs, methods):
  return dict((x, dict((y, dict((z, None) for z in methods)) for y in graphs)) for x in algos) 


def readTimings(fn):
  try:
    return float(open(fn).read())
  except(IOError, ValueError):
    return -1


def printSummary(algos, graphs, methods, cpuTimings, gpuTimings):
  for algo in algos:
    print(5 * "--------------------")
    print("Algorithm: %s" % algo)
    sys.stdout.write("%-20.20s  %8s" % ('Graph', 'GraphLab'))
    for method in methods:
      sys.stdout.write("{:^24s}".format(method))
    sys.stdout.write("\n")
    print(5 * "--------------------")
    for graph in graphs:
      sys.stdout.write("%-20.20s  %8.1f" % (graph, cpuTimings[algo][graph]))
      for method in methods:
        sys.stdout.write("  %8.1f  %8.1fx" % (gpuTimings[algo][graph][method]
          , cpuTimings[algo][graph] / gpuTimings[algo][graph][method]))
      sys.stdout.write('\n')
    print(5 * "--------------------")
      

cpuTimingFiles = glob.glob('*.timing')
gpuTimingFiles = glob.glob('*.timing_gpu')
graphs = set(x.split('.')[0] for x in cpuTimingFiles + gpuTimingFiles)
algos  = set(x.split('.')[1] for x in cpuTimingFiles + gpuTimingFiles)
methods = set(x.split('.')[2] for x in gpuTimingFiles)
cpuTimings = timingsDict(algos, graphs)
gpuTimings = gpuTimingsDict(algos, graphs, methods)
print(methods)
for algo, graph in itertools.product(algos, graphs):
  cpuTimings[algo][graph] = readTimings('%s.%s.timing' % (graph, algo))
  for method in methods:
    gpuTimings[algo][graph][method] = readTimings('%s.%s.%s.timing_gpu' % (graph, algo, method))

printSummary(algos, graphs, methods, cpuTimings, gpuTimings)
