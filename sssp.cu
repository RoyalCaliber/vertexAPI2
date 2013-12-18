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

//Single source shortest paths using vertexAPI2

#include "util.h"
#include "graphio.h"
#include "refgas.h"
#include "gpugas.h"
#include <climits>

struct SSSP
{
  //making these typedefs rather than singleton structs
  typedef int VertexData;
  typedef int EdgeData;

  typedef int GatherResult;
  static const int maxLength = 100000;
  static const int gatherZero = INT_MAX - maxLength;

  enum {Commutative = true};


  __host__ __device__
  static int gatherReduce(const int& left, const int& right)
  {
    return min(left, right);
  }


  __host__ __device__
  static int gatherMap(const VertexData* dstDist, const VertexData *srcDist, const EdgeData* edgeLen)
  {
    return *srcDist + *edgeLen;
  }


  __host__ __device__
  static bool apply(VertexData* curDist, GatherResult dist)
  {
    bool changed = dist < *curDist;
    *curDist = min(*curDist, dist);
    return changed;
  }


  __host__ __device__
  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


template<typename Engine>
int64_t run(int nVertices, SSSP::VertexData* vertexData, int nEdges
  , SSSP::EdgeData* edgeData, const int* srcs, const int* dsts)
{
  Engine engine;
  #ifdef VERTEXAPI_USE_MPI
    engine.initMPI();
  #endif
  engine.setGraph(nVertices, vertexData, nEdges, edgeData, srcs, dsts);

  //TODO, setting all vertices to active for first step works, but it would
  //be faster to instead set to neighbors of starting vertex
  engine.setActive(0, nVertices);
  int64_t t0 = currentTime();
  engine.run();
  engine.getResults();
  int64_t t1 = currentTime();
  return t1 - t0;
}


void outputDists(int nVertices, int* dists, FILE* f = stdout)
{
  for (int i = 0; i < nVertices; ++i)
    fprintf(f, "%d %d\n", i, dists[i]);
}


int main(int argc, char** argv)
{
  #ifdef VERTEXAPI_USE_MPI
    int mpiRank = 0;
    int mpiSize = 0; //number of mpi nodes

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
  #endif

  #ifdef VERTEXAPI_USE_MPI
    #define MASTER (mpiRank == 0)
  #else
    #define MASTER (1)
  #endif
  
  char *inputFilename;
  char *outputFilename = 0;
  char *inputDegreeFilename;
  int sourceVertex;
  bool runTest;
  bool dumpResults;
  bool useMaxOutDegreeStart;
  if( !parseCmdLineSimple(argc, argv, "ssi-t-d-m|s", &inputFilename, &inputDegreeFilename
    , &sourceVertex, &runTest, &dumpResults, &useMaxOutDegreeStart, &outputFilename) )
  {
    printf("Usage: sssp [-t] [-d] [-m] inputfile inputDegrees source [outputfile]\n");
    exit(1);
  }

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  std::vector<int> edgeData;
  #ifdef VERTEXAPI_USE_MPI
    std::string tmp;
    tmp = filenameSuffixMPI(inputFilename, mpiRank, mpiSize);
    loadGraph(tmp.c_str(), nVertices, srcs, dsts, &edgeData);
  #else
    loadGraph(inputFilename, nVertices, srcs, dsts, &edgeData);
  #endif
  if( edgeData.size() == 0 )
  {
    printf("No edge data available in input file\n");
    exit(1);
  }

  //read in out degrees from file.  This is the correct number of vertices
  std::vector<int> outDegrees;
  loadData(inputDegreeFilename, outDegrees);
  nVertices = outDegrees.size();

  if( useMaxOutDegreeStart )
  {
    sourceVertex = std::max_element(outDegrees.begin(), outDegrees.end()) - outDegrees.begin();
    int maxDegree = outDegrees[sourceVertex];
    printf("using vertex %d with degree %d as source\n", sourceVertex, maxDegree);
  }
  

  //initialize vertex data
  std::vector<int> vertexData(nVertices);
  for( int i = 0; i < nVertices; ++i )
    vertexData[i] = SSSP::gatherZero;
  vertexData[sourceVertex] = 0;

  std::vector<int> refVertexData;
  if( runTest )
  {
    printf("Running reference calculation\n");
    refVertexData = vertexData;
    run< GASEngineRef<SSSP> >(nVertices, &refVertexData[0], (int)srcs.size()
      , &edgeData[0], &srcs[0], &dsts[0]);
    if( dumpResults )
    {
      printf("Reference:\n");
      outputDists(nVertices, &refVertexData[0]);
    }  
  }

  int64_t t = run< GASEngineGPU<SSSP> >(nVertices, &vertexData[0], (int)srcs.size()
    , &edgeData[0], &srcs[0], &dsts[0]);
  if( MASTER )
    printf("Took %f ms\n", t / 1000.0f);
    
  if( MASTER && dumpResults )
  {
    printf("GPU:\n");
    outputDists(nVertices, &vertexData[0]);
  }

  if( MASTER && runTest )
  {
    bool diff = false;
    for( int i = 0; i < nVertices; ++i )
    {
      if( vertexData[i] != refVertexData[i] )
      {
        printf("%d %d %d\n", i, refVertexData[i], vertexData[i]);
        diff = true;
      }
    }
    if( diff )
      return 1;
    else
      printf("No differences found\n");
  }

  if( MASTER && outputFilename )
  {
    printf("writing results to %s\n", outputFilename);
    FILE* f = fopen(outputFilename, "w");
    outputDists(nVertices, &vertexData[0], f);
    fclose(f);
  }

  free(inputFilename);
  free(outputFilename);

  #ifdef VERTEXAPI_USE_MPI
    MPI_Finalize();
  #endif

  return 0;
}
