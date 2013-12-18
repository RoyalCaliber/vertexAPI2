#!/usr/bin/python

#Uses python 3
#Runs regressions.  Sick of GNU make's limitations.  Think of this
#as a pythonic micro-implementation of make.

import os
import sys
from os.path import join as pcat
from os import system
from itertools import product as cartprod
import inspect

class MakeFailure(Exception):
  pass


#for pure dependency rules
def doNothing():
  pass


#if your shell is not bash, remove the 'set -o pipefail; '
def cmd(s):
  print(s)
  ret = os.system('set -o pipefail; ' + s)
  if ret > 0:
    print('error executing command')
    raise MakeFailure


#Each node in the dependency graph has a recipe for building that node
class DepNode:
  def __init__(self, target):
    self.srcs = list()
    self.target = target
    self.recipe = None
    self.force  = False
    self.upToDate = False
    self.mtime  = None


class DepGraph:
  def __init__(self, verbose=False):
    self.nodes = {}
    self.verbose = verbose


  def getNode(self, target):
    try:
      return self.nodes[target]
    except KeyError:
      node = DepNode(target)
      self.nodes[target] = node
      return node
      

  #add a node for target, which depends on sources and a recipe
  #for constructing target.  recipe can be none to simply add the
  #dependencies
  def add(self, target, sources, recipe=doNothing, force=False):
    node = self.getNode(target)
    for source in sources:
      snode = self.getNode(source)
      node.srcs.append(snode)
    node.force = force
    if node.recipe is None or recipe != doNothing:
      if node.recipe is not None and node.recipe != doNothing:
        print('Warning: rule for %s uses %s to override %s' % (target
          , inspect.getsource(recipe), inspect.getsource(node.recipe)))
      node.recipe = recipe


  def resetOODCache(self):
    for target, node in self.nodes.items():
      node.upToDate = False


  def isNewer(self, fn, t):
    if os.path.exists(fn) and os.path.getmtime(fn) > t:
      return True
    

  def anySourceNewer(self, node):
    if not os.path.exists(node.target):
      return True
    tThis = os.path.getmtime(node.target)
    for snode in node.srcs:
      if self.isNewer(snode.target, tThis):
        if self.verbose:
          print('%s is newer than target %s' % (snode.target, node.target))
        return True
    return False


  def runRecipe(self, node):
    if node.recipe is None:
      print('no rule to make %s' % node.target)
      raise MakeFailure
    if self.verbose:
      print('building target %s with code %s' % (node.target, inspect.getsource(node.recipe)))
    node.recipe()


  #recursively builds the specified node
  #resetOODCache must be called before the top-level call
  #returns whether we rebuilt this node (not success!)
  def make(self, node):
    if self.verbose:
      print('making %s' % node.target)
    if node.upToDate:
      return False
    #make all sources
    anySourceRebuilt = False
    for snode in node.srcs:
      if self.make(snode):
        anySourceRebuilt = True
    node.upToDate = True

    if not node.force and not anySourceRebuilt and node.recipe == doNothing:
      return False
    
    if self.anySourceNewer(node):
      self.runRecipe(node)
      return True
    else:
      if self.verbose:
        print('%s is up to date' % node.target)
      return False
    

  #makes a particular target
  def makeTarget(self, target):
    self.resetOODCache()
    node = self.nodes[target]
    didSomething = self.make(node)
    if not didSomething:
      print('nothing to do for target %s' % node.target)


################################################################################
#Now the fun stuff: regressions

depGraph  = DepGraph()

graphDir = '../test-graphs'
pgBinDir = '../PowerGraphReferenceImplementations'
pgOpts   = '--ncpus 4'
algorithms = ('connected_component', 'pagerank', 'sssp', 'bfs')
graphs = ('ak2010', 'belgium_osm', 'delaunay_n13', 'coAuthorsDBLP', 'delaunay_n21', 'webbase-1M') #soc-LiveJournal1 #kron_g500-logn21
#mpiN where N is the number of processors to use.
methods = ('mpi2',) #('nompi', 'mpi2', 'mpi1')


#how to run powergraph to make a gold file
def makeGold(bin, mtx, out, tm):
  cmd('rm -f __tmpgold.out* ;')
  cmd('{0} {pgOpts} --graph {1} --graph_opts ingress=batch --save __tmpgold.out'
    " | awk '/Finished Running/{{print $5}}' > {2}" .format(bin, mtx, tm, pgOpts=pgOpts))
  cmd('cat __tmpgold.out* | sort -n > {}'.format(out))
  cmd('rm -f __tmpgold.out* ;')


#add dependency information for all combinations for gold files
goldFiles = []
for graph, algo in cartprod(graphs, algorithms):
  bin = '{}/{}.x'.format(pgBinDir, algo)
  out = '{}.{}.gold'.format(graph, algo)
  mtx = '{0}/{1}/{1}.mtx'.format(graphDir, graph)
  tm  = '{}.{}.timing'.format(graph, algo)
  depGraph.add(out, (bin, mtx), lambda b=bin, m=mtx, o=out, t=tm: makeGold(b, m, o, t))
  goldFiles.append(out)
depGraph.add('gold', goldFiles)


#generate .deg and part.mtx files
for graph in graphs:
  deg = '{}.deg'.format(graph)
  mtx = '{0}/{1}/{1}.mtx'.format(graphDir, graph)
  bin = '../outdeg'
  depGraph.add(deg, (bin, mtx), lambda b=bin, m=mtx, d=deg: cmd('%s %s %s' % (b, m, d)))

  for method in methods:
    if method.startswith('mpi'):
      ncpus = int(method[3:])
      parts = ['{}.part.mtx_{}_{}'.format(graph, ncpus, icpu) for icpu in range(ncpus)]
      for part in parts:
        s = '%s/partition.py %s %s %s.part.mtx_%s_' % (graphDir, mtx, ncpus, graph, ncpus)
        depGraph.add(part, (mtx, '%s/partition.py' % graphDir), lambda s=s: cmd(s))
      depGraph.add('parts', parts)


#program-specific command lines
def testCmd(algo, mpirun, bin, mtx, deg, out, tm):
  def runCmd(s):
    cmd(s)
    cmd('sort -n __tmpout > {}'.format(out))
    cmd('rm -f __tmpout')
      
  if algo == 'pagerank':
    s = "{mpirun} {bin} {mtx} {deg} __tmpout | awk '/Took/{{print $2}}' > {tm}"
  elif algo == 'sssp':
    s = "{mpirun} {bin} -m {mtx} {deg} 0 __tmpout | awk '/Took/{{print $2}}' > {tm}"
  elif algo == 'bfs':
    s = "{mpirun} {bin} -m {mtx} {deg} 0 __tmpout | awk '/Took/{{print $2}}' > {tm}"
  elif algo == 'connected_component':
    s = "{mpirun} {bin} {mtx} {deg} __tmpout | awk '/Took/{{print $2}}' > {tm}"
  else:
    raise Exception('write the command line for algo %s' % algo)
  s = s.format(mpirun=mpirun, bin=bin, mtx=mtx, deg=deg, tm=tm)
  return lambda s=s: runCmd(s)  


#program-specific regression checking
def regCmd(graph, algo, out, passFile):
  if algo == 'pagerank':
    s = './checkPageRank.py {out} {graph}.{algo}.gold && touch {passFile}'
  elif algo in ('sssp', 'bfs', 'connected_component'):
    s = './checkDist.py {out} {graph}.{algo}.gold && touch {passFile}'
  s = s.format(graph=graph, algo=algo, out=out, passFile=passFile)
  return lambda s=s: cmd(s)

  
for graph, algo, method in cartprod(graphs, algorithms, methods):
  out = '{}.{}.{}.test'.format(graph, algo, method)
  deg = '{}.deg'.format(graph)
  tm  = '{}.{}.{}.timing_gpu'.format(graph, algo, method)
  if method.startswith('mpi'):
    ncpus = method[3:]
    mpirun = 'mpirun -np {}'.format(ncpus)
    mtx = '{}.part.mtx'.format(graph)
    bin = '../{}_mpi'.format(algo)
    depGraph.add(out, (bin, 'parts', deg), testCmd(algo, mpirun, bin, mtx, deg, out, tm))  
  else:
    ncpus = 0
    mpirun = ''
    mtx = '{0}/{1}/{1}.mtx'.format(graphDir, graph)
    bin = '../{}'.format(algo)
    depGraph.add(out, (bin, mtx, deg), testCmd(algo, mpirun, bin, mtx, deg, out, tm))  
  depGraph.add('test', (out,))

  passFile = '{}.{}.{}.pass'.format(graph, algo, method)
  depGraph.add(passFile, (out,), regCmd(graph, algo, out, passFile))
  depGraph.add('regress', (passFile,))



depGraph.makeTarget(sys.argv[1])