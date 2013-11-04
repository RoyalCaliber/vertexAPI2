#utility module to load a graph into memory
#this will take a TON of memory since we're using
#straight python data structures

from cmath import *
from collections import defaultdict


__MTXNumberTypes = {
  'real': float,
  'integer': int,
  'complex': complex
} 


class MTXHeader:
  def __init__(self):
    self.format    = None
    self.dataType  = None
    self.symType   = None
    self.nVertices = -1
    self.nEdges    = -1


def readMTXHeader(f):
  hdr = MTXHeader()
  line = f.readline()
  if not line.startswith( '%%MatrixMarket' ):
    raise ValueError( 'invalid header line: %s' % line )
  matrix, hdr.format, hdr.dataType, hdr.symType = line[15:].split()
  while 1:
    line = f.readline()
    if not line:
      raise IOError('mtx header is missing number of entries')
    if line[0] != '%':
      words = line.split()
      if len(words) == 3:
        hdr.nVertices, dummy, hdr.nEdges = map(int, words)
        assert dummy == hdr.nVertices
      else:
        hdr.nVertices, hdr.nEdges = map(int, words)
      break
  return hdr
  

#loads an edge list and any associated value from a text file
def readEdgeListMTX(f, header, disallowSelfLinks=True):
  haveData = (header.dataType != 'pattern')
  dataType = __MTXNumberTypes[header.dataType.lower()]
  while 1:
    line = f.readline()
    if not line:
      return
    if line.startswith('%'):
      pass
    else:
      if haveData:
        src, dst, value = line.split()
        value = dataType(value)
      else:
        src, dst = line.split()
        value = None
      #adjust for matrix format's 1-based indexing
      src = int(src)-1
      dst = int(dst)-1
      if src != dst:      
        yield src, dst, value
        if header.symType == 'symmetric':
          yield dst, src, value
        elif header.symType == 'skew-symmetric':
          yield dst, src, -value
        elif header.symType == 'hermitian':
          yield dst, src, value.conjugate()
      elif not disallowSelfLinks:
        yield src, dst, value
        
          

#converts to CSC format
#srcs[dv] is a tuple of src indices
#values[dv] is a tuple of values of the same size as srcs[dv]
def edgeListToCSC(edgeList):
  srcs = defaultdict(list)
  values = defaultdict(list)
  for edge in edgeList:
    src, dst = edge[:2]
    srcs[dst].append(src)
    if len(edge) == 3:
      values[dst].append(edge[2])
  return srcs, values


#converts to CSR format
#srcs[dv] is a tuple of src indices
#values[dv] is a tuple of values of the same size as srcs[dv]
def edgeListToCSR(edgeList):
  dsts = defaultdict(list)
  values = defaultdict(list)
  for edge in edgeList:
    src, dst = edge[:2]
    dsts[src].append(dst)
    if len(edge) == 3:
      values[src].append(edge[2])
  return dsts, values


def writeEdgeList(f, edges):
  for edge in edges:
    f.write('%d %d' % (edge[0], edge[1]))
    if len(edges) == 3:
      f.write(' %s', edge[2])
    f.write('\n')


def writeMTX(f, header, edges):
  f.write('%%MatrixMarket matrix %s %s general\n' % (header.format, header.dataType))
  f.write('-1 -1\n')
  for edge in edges:
    f.write('%d %d' % (edge[0] + 1, edge[1] + 1))
    if header.dataType != 'pattern':
      f.write(' %s\n' % edge[2])
    else:
      f.write('\n')


