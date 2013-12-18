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

//Connected Component using vertexAPI2

#include "util.h"
#include "graphio.h"
#include "refgas.h"
#include "gpugas.h"
#include <climits>

struct CC
{
  //making these typedefs rather than singleton structs
  typedef int VertexData;
  struct EdgeData {};

  typedef int GatherResult;
  static const int gatherZero = INT_MAX;

  enum { Commutative = true };

  __host__ __device__
  static int gatherReduce(const int& left, const int& right)
  {
    return min(left, right);
  }


  __host__ __device__
  static int gatherMap(const VertexData* dstLabel, const VertexData *srcLabel, const EdgeData* edge)
  {
    return *srcLabel;
  }


  __host__ __device__
  static bool apply(VertexData* curLabel, GatherResult label)
  {
    bool changed = label < *curLabel;
    *curLabel = min(*curLabel, label);
    return changed;
  }


  __host__ __device__
  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


template<typename Engine>
int64_t run(int nVertices, CC::VertexData* vertexData, int nEdges
       , const int* srcs, const int* dsts)
{
  Engine engine;
  #ifdef VERTEXAPI_USE_MPI
    engine.initMPI();
  #endif
  engine.setGraph(nVertices, vertexData, nEdges, 0, srcs, dsts);

  //TODO, setting all vertices to active for first step works, but it would
  //be faster to instead set to neighbors of starting vertex
  engine.setActive(0, nVertices);
  int64_t t0 = currentTime();
  engine.run();
  engine.getResults();
  int64_t t1 = currentTime();
  return t1 - t0;
}


void outputLabels(int nVertices, int* labels, FILE* f = stdout)
{
  for (int i = 0; i < nVertices; ++i)
    fprintf(f, "%d %d\n", i, labels[i]);
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
  char *inputDegreeFilename;
  char *outputFilename = 0;
  bool runTest;
  bool dumpResults;
  if( !parseCmdLineSimple(argc, argv, "ss-t-d|s", &inputFilename, &inputDegreeFilename
                        , &runTest, &dumpResults, &outputFilename) )
  {
    printf("Usage: cc [-t] [-d] inputfile source [outputfile]\n");
    exit(1);
  }

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  #ifdef VERTEXAPI_USE_MPI
    std::string tmp;
    tmp = filenameSuffixMPI(inputFilename, mpiRank, mpiSize);
    loadGraph(tmp.c_str(), nVertices, srcs, dsts);
  #else
    loadGraph(inputFilename, nVertices, srcs, dsts);
  #endif  
  printf("loaded %s with %d vertices and %zd edges\n", inputFilename, nVertices, srcs.size());

  //this has no purpose other than to figure out the number of vertices.
  //we could pass it in from the command line, but this avoids any modification
  //to the regression scripts.
  std::vector<int> outDegrees;
  loadData(inputDegreeFilename, outDegrees);
  nVertices = outDegrees.size();

  //initialize vertex data
  std::vector<int> vertexData(nVertices);
  for (int i = 0; i < nVertices; ++i)
    vertexData[i] = i;

  std::vector<int> refVertexData;
  if( runTest )
  {
    printf("Running reference calculation\n");
    refVertexData = vertexData;
    run< GASEngineRef<CC> >(nVertices, &refVertexData[0], (int)srcs.size()
                          , &srcs[0], &dsts[0]);
    if( MASTER && dumpResults )
    {
      printf("Reference:\n");
      outputLabels(nVertices, &refVertexData[0]);
    }  
  }

  int64_t t = run< GASEngineGPU<CC> >(nVertices, &vertexData[0], (int)srcs.size()
    , &srcs[0], &dsts[0]);

  if( MASTER )
    printf("Took %f ms\n", t/1000.0f);


  if( MASTER && dumpResults )
  {
    printf("GPU:\n");
    outputLabels(nVertices, &vertexData[0]);
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
    outputLabels(nVertices, &vertexData[0], f);
    fclose(f);
  }

  free(inputFilename);
  free(outputFilename);
  
  #ifdef VERTEXAPI_USE_MPI
    MPI_Finalize();
  #endif

  return 0;
}
