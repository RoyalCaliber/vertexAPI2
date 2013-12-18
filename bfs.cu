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

//BFS using vertexAPI2

#include "util.h"
#include "graphio.h"
#include "refgas.h"
#include "gpugas.h"


//nvcc doesn't like the __device__ variable to be a static member inside BFS
//so these are both outside.
int g_iterationCount;
__device__ __constant__ int g_iterationCountGPU;


struct BFS
{
  struct VertexData
  {
    int depth;
  };

  struct EdgeData {}; //nothing

  typedef int GatherResult;
  static const int gatherZero = INT_MAX - 1;

  enum { Commutative = true };

  __host__ __device__
  static int gatherReduce(const int& left, const int& right)
  {
    return 0; //do nothing
  }


  __host__ __device__
  static int gatherMap(const VertexData* dst, const VertexData *src, const EdgeData* edge)
  {
    return 0; //do nothing
  }


  __host__ __device__
  static bool apply(VertexData* vert, int dist)
  {
    if( vert->depth == -1 )
    {
      #ifdef __CUDA_ARCH__
        vert->depth = g_iterationCountGPU;
      #else
        vert->depth = g_iterationCount;
      #endif        
      return true;
    }
    return false;
  }


  __host__ __device__
  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


template<bool GPU>
void setIterationCount(int v)
{
  if( GPU )
    cudaMemcpyToSymbol(g_iterationCountGPU, &v, sizeof(v));
  else
    g_iterationCount = v;
}


template<typename Engine, bool GPU>
int64_t run(int nVertices, BFS::VertexData* vertexData, int nEdges
  , const int *srcs, const int *dsts, int sourceVertex)
{
  Engine engine;
  #ifdef VERTEXAPI_USE_MPI
    engine.initMPI();
  #endif
  engine.setGraph(nVertices, vertexData, nEdges, 0, &srcs[0], &dsts[0]);
  engine.setActive(sourceVertex, sourceVertex+1);
  int iter = 0;
  setIterationCount<GPU>(iter);
  int64_t t0 = currentTime();
  while( engine.countActive() )
  {
    //run apply without gather
    engine.gatherApply(false);
    engine.scatterActivate(false);
    engine.nextIter();
    setIterationCount<GPU>(++iter);
  }
  engine.getResults();
  int64_t t1 = currentTime();
  return t1 - t0;
}


void outputDepths(int nVertices, BFS::VertexData* vertexData, FILE *f = stdout)
{
  for( int i = 0; i < nVertices; ++i )
    fprintf(f, "%d %d\n", i, vertexData[i].depth);
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
  int sourceVertex;
  bool runTest;
  bool dumpResults;
  bool useMaxOutDegreeStart;
  if( !parseCmdLineSimple(argc, argv, "ssi-t-d-m|s", &inputFilename, &inputDegreeFilename
    , &sourceVertex, &runTest, &dumpResults, &useMaxOutDegreeStart, &outputFilename) )
  {
    printf("Usage: bfs [-t] [-d] [-m] inputfile inputDegrees source [outputFilename]\n");
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

  //read in out degrees from file.  This is the correct number of vertices
  std::vector<int> outDegrees;
  loadData(inputDegreeFilename, outDegrees);
  nVertices = outDegrees.size();
    
  //initialize vertex data
  std::vector<BFS::VertexData> vertexData(nVertices);
  for( int i = 0; i < nVertices; ++i )
    vertexData[i].depth = -1;

  if( useMaxOutDegreeStart )
  {
    sourceVertex = std::max_element(outDegrees.begin(), outDegrees.end()) - outDegrees.begin();
    int maxDegree = outDegrees[sourceVertex];
    printf("using vertex %d with degree %d as source\n", sourceVertex, maxDegree);
  }  
    
  std::vector<BFS::VertexData> refVertexData;
  if( runTest )
  {
    refVertexData = vertexData;
    run<GASEngineRef<BFS>, false>(nVertices, &refVertexData[0], (int)srcs.size()
      , &srcs[0], &dsts[0], sourceVertex);
    if( dumpResults )
    {
      printf("Reference:\n");
      outputDepths(nVertices, &refVertexData[0]);
    }
  }

  int64_t t = run<GASEngineGPU<BFS>, true>(nVertices, &vertexData[0], (int) srcs.size()
    , &srcs[0], &dsts[0], sourceVertex);
    
  if( MASTER )
    printf("Took %f ms\n", t/1000.0f);
    
  if( MASTER && dumpResults )
  {
    printf("GPU:\n");
    outputDepths(nVertices, &vertexData[0]);
  }

  if( MASTER && runTest )
  {
    bool diff = false;
    for( int i = 0; i < nVertices; ++i )
    {
      if( vertexData[i].depth != refVertexData[i].depth )
      {
        printf("%d %d %d\n", i, refVertexData[i].depth, vertexData[i].depth);
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
    outputDepths(nVertices, &vertexData[0], f);
    fclose(f);
  }

  free(inputFilename);
  free(outputFilename);

  #ifdef VERTEXAPI_USE_MPI
    MPI_Finalize();
  #endif
  
  return 0;
}
