#ifndef UTIL_H__
#define UTIL_H__

#include <stdint.h>
#include <vector>
#include <algorithm>

//some simple utility routines to avoid duplication across test programs

//Return the current time in microseconds from epoch.
int64_t currentTime();

//extract arguments from the command line, useful for very simple
//command lines
int parseCmdLineSimple(int argc, char **argv, const char*fmt, ...);



//Helper function for edgeListToCS*()
//this is a quick and dirty implementation because we are no concerned at this
//point with how a CSC or CSR representation is obtained.
//Return sortedIdx such that inData[sortedIdx] is sorted ascending
template<typename Int>
void indSort(int n, const Int* inData, Int* sortedIdx)
{
  typedef std::pair<Int, Int> Pair;
  
  struct PairCmp
  {
    static bool lt(const Pair &p1, const Pair &p2)
    {
      return p1.first < p2.first;
    }
  };
  
  std::vector<Pair> pairs(n);
  for( Int i = 0; i < n; ++i )
  {
    pairs[i].first  = inData[i];
    pairs[i].second = i;
  }
  
  std::sort(pairs.begin(), pairs.end(), PairCmp::lt);

  for( Int i = 0; i < n; ++i )
    sortedIdx[i] = pairs[i].second;
}


//convert a list of edges into a CSR representation of the adjacency matrix
//again, this is not intended to be efficient, since this is outside the
//scope of the present work.
//Note: offsets should have nVertices + 1 elements.
template<typename Int>
void edgeListToCSR(Int nVertices, Int nEdges
  , const Int *srcs, const Int *dsts
  , Int *offsets, Int *outDsts, Int* sortIndices)
{
  Int* ind = sortIndices;
  indSort(nEdges, srcs, ind);
  for( Int i = 0; i < nEdges; ++i )
    outDsts[i] = dsts[ ind[i] ];
  Int count = 0;
  Int prev  = -1;
  for( Int i = 0; i < nEdges; ++i )
  {
    if( srcs[ ind[i] ] != prev )
    {
      prev = srcs[ ind[i] ];
      offsets[ count ] = i;
      ++count;
    }
  }
  //sentinel
  offsets[nVertices] = nEdges;
}


//convert a list of edges into a CSC representation of the adjacency matrix
template<typename Int>
void edgeListToCSC(Int nVertices, Int nEdges
  , const Int *srcs, const Int *dsts
  , Int *offsets, Int *outSrcs, Int* sortIndices)
{
  edgeListToCSR<Int>(nVertices, nEdges, dsts, srcs 
    , offsets, outSrcs, sortIndices);
};  


#endif
