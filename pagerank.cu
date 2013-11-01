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

//this should come from the Makefile, putting it in here for testing
#define VERTEXAPI_USE_MPI

#include "refgas.h"
#include "gpugas.h"
#include "util.h"
#include "graphio.h"
#include <vector>


//Vertex program for Pagerank
struct PageRank
{  
  static const float pageConst = 0.15f;
  static const float tol = 0.01f;
    
  struct VertexData
  {
    float rank;
    int   numOutEdges;
  };

  struct EdgeData {};

  typedef float GatherResult;

  static const float gatherZero = 0.0f;

  enum { Commutative = true };

  __host__ __device__
  static float gatherMap(const VertexData* dst, const VertexData* src, const EdgeData* edge)
  {
    //this division is being done too many times right?
    //should just store the normalized value in apply?
    return src->rank / src->numOutEdges;
  }

  __host__ __device__
  static float gatherReduce(const float& left, const float& right)
  {
    return left + right;
  }

  __host__ __device__
  static bool apply(VertexData* vertexData, const float& gatherResult)
  {
    float newRank = pageConst + (1.0f - pageConst) * gatherResult;
    bool ret = fabs(newRank - vertexData->rank) >= tol;
    vertexData->rank = newRank;
    return ret;
  }

  __host__ __device__
  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


void outputRanks(int n, const PageRank::VertexData* vertexData, FILE* f = stdout)
{
  for( int i = 0; i < n; ++i )
  {
    fprintf(f, "%d %f\n", i, vertexData[i].rank);
  }
}


template<typename Engine>
void run(int nVertices, PageRank::VertexData* vertexData, int nEdges
  , const int* srcs, const int* dsts)
{
  Engine engine;
  engine.setGraph(nVertices, vertexData, nEdges, 0, srcs, dsts);
  #ifdef VERTEXAPI_USE_MPI
    engine.initMPI();
  #endif
  //all vertices begin active for pagerank
  engine.setActive(0, nVertices);
  int64_t t0 = currentTime();
  engine.run();
  engine.getResults();
  int64_t t1 = currentTime();
  printf("Took %f ms\n", (t1 - t0)/1000.0f);
}


int main(int argc, char **argv)
{
  int mpiRank = 0;

  #ifdef VERTEXAPI_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  #endif
  
  char* inputFilename;
  char* inputDegreeFilename;
  char* outputFilename = 0;
  bool runTest;
  bool dumpResults;
  if( !parseCmdLineSimple(argc, argv, "ss-t-d|s"
    , &inputFilename, &inputDegreeFilename, &runTest, &dumpResults, &outputFilename) )
  {
    printf("Usage: pagerank [-t] [-d] inputEdges inputDegrees [outputfile]\n");
    exit(1);
  }

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  #ifdef VERTEXAPI_USE_MPI
    char rankStr[8];
    snprintf(rankStr, sizeof(rankStr), "_%d", mpiRank);
    std::string tmp = inputFilename;
    tmp += rankStr;
    loadGraph(tmp.c_str(), nVertices, srcs, dsts);
    printf("%d: loaded %zd edges\n", mpiRank, srcs.size());
  #else
    loadGraph(inputFilename, nVertices, srcs, dsts);
    printf("loaded %s with %d vertices and %zd edges\n", inputFilename, nVertices, srcs.size());
  #endif

  //read in the out-degree for the vertices
  std::vector<int> outDegrees;
  loadData(inputDegreeFilename, outDegrees);

  //Get the actual number of vertices
  #ifdef VERTEXAPI_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nVertices, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    printf("%d: nVertices = %d\n", mpiRank, nVertices);
  #endif

  //initialize vertex data
  //convert to CSR to get the count of edges.
  std::vector<int> srcOffsets(nVertices + 1);
  std::vector<int> csrSrcs(srcs.size());
  edgeListToCSR<int>(nVertices, srcs.size(), &srcs[0], &dsts[0], &srcOffsets[0], 0, 0);
  
  std::vector<PageRank::VertexData> vertexData(nVertices);
  for( int i = 0; i < nVertices; ++i )
  {
    vertexData[i].numOutEdges = outDegrees[i]; //srcOffsets[i + 1] - srcOffsets[i];
    vertexData[i].rank = PageRank::pageConst;
  }
  
  std::vector<PageRank::VertexData> refVertexData;
  if( runTest )
  {
    printf("Running reference calculation\n");
    refVertexData = vertexData;
    run< GASEngineRef<PageRank> >(nVertices, &refVertexData[0], (int)srcs.size(), &srcs[0], &dsts[0]);
    if( dumpResults )
    {
      printf("Reference\n");
      outputRanks(nVertices, &refVertexData[0]);
    }
  }

//  run< GASEngineGPU<PageRank> >(nVertices, &vertexData[0], (int)srcs.size(), &srcs[0], &dsts[0]);
//  if( dumpResults )
//  {
//    printf("GPU:\n");
//    outputRanks(nVertices, &vertexData[0]);
//  }

  if( runTest )
  {
    const float tol = 1.0e-6f;
    bool diff = false;
    for( int i = 0; i < nVertices; ++i )
    {
      if( fabs(vertexData[i].rank - refVertexData[i].rank) > tol )
      {
        printf("%d %f %f\n", i, refVertexData[i].rank, vertexData[i].rank);
        diff = true;
      }
    }
    if( diff )
      return 1;
    else
      printf("No differences found\n");
  }

  if( outputFilename && mpiRank == 0 )
  {
    FILE* f = fopen(outputFilename, "w");
    printf("writing results to file %s\n", outputFilename);
    outputRanks(nVertices, &vertexData[0], f);
    fclose(f);
  }

  free(inputFilename);
  free(outputFilename);

  #ifdef VERTEXAPI_USE_MPI
    MPI_Finalize();
  #endif
  
  return 0;
}

