#!/usr/bin/python3

#########################################################################
#Copyright 2013 Royal Caliber LLC. (http://www.royal-caliber.com)
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
#########################################################################

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
