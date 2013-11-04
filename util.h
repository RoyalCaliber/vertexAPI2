/******************************************************************************
Copyright 2013 Royal Caliber LLC. (http://www.royal-caliber.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************/

#ifndef UTIL_H__
#define UTIL_H__

#include <stdint.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <string>

//some simple utility routines to avoid duplication across test programs

//Return the current time in microseconds from epoch.
int64_t currentTime();

//extract arguments from the command line, useful for very simple
//command lines
//The format string is a sequence of one letter characters:
//s: string argument (returned via strdup, to be freed with free)
//i: integer
//f: float
//-x: a dash indicates the presence of an optional argument '-x'
//    anywhere in the command line.  The corresponding bool* argument
//    is set to 1 if the option is present
//|:  all positional arguments after the pipe symbol are optional
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
  Int* ind;
  std::vector<Int> tmpIndices;
  if( sortIndices )
    ind = sortIndices;
  else
  {
    tmpIndices.resize(nEdges);
    ind = &tmpIndices[0];
  }
  indSort(nEdges, srcs, ind);

  if( outDsts )
  {
    for( Int i = 0; i < nEdges; ++i )
      outDsts[i] = dsts[ ind[i] ];
  }

  Int curSrc = 0;
  for( Int i = 0; i < nEdges; ++i )
  {
    Int src = srcs[ind[i]];
    while( curSrc <= src )
      offsets[curSrc++] = i;
  }
  while(curSrc <= nVertices)
    offsets[curSrc++] = nEdges;
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


//add a rank/size suffix to the filename, used for loading pre-partitioned
//files from disk when running with MPI
std::string filenameSuffixMPI(const char* fn, int rank, int size);

#endif
