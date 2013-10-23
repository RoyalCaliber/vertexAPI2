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
void run(int nVertices, BFS::VertexData* vertexData, int nEdges
  , const int *srcs, const int *dsts, int sourceVertex)
{
  Engine engine;
  engine.setGraph(nVertices, vertexData, nEdges, 0, &srcs[0], &dsts[0]);
  engine.setActive(sourceVertex, sourceVertex+1);
  int iter = 0;
  setIterationCount<GPU>(iter);  
  while( engine.countActive() )
  {
    //run apply without gather
    engine.gatherApply(false);
    engine.scatterActivate(false);
    engine.nextIter();
    setIterationCount<GPU>(++iter);
  }
  engine.getResults();
}


int main(int argc, char** argv)
{
  char *inputFilename;
  int sourceVertex;
  bool runTest;
  bool dumpResults;
  if( !parseCmdLineSimple(argc, argv, "si-t-d", &inputFilename, &sourceVertex
    , &runTest, &dumpResults) )
  {
    printf("Usage: bfs [-t] [-d] inputfile source\n");
    exit(1);
  }

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  loadGraph(inputFilename, nVertices, srcs, dsts);

  //initialize vertex data
  std::vector<BFS::VertexData> vertexData(nVertices);
  for( int i = 0; i < nVertices; ++i )
    vertexData[i].depth = -1; 

  std::vector<BFS::VertexData> refVertexData;
  if( runTest )
  {
    refVertexData = vertexData;
    run<GASEngineRef<BFS>, false>(nVertices, &refVertexData[0], (int)srcs.size()
      , &srcs[0], &dsts[0], sourceVertex);
    if( dumpResults )
    {
      printf("Reference:\n");
      for( int i = 0; i < nVertices; ++i )
        printf("%d %d\n", i, refVertexData[i].depth);
    }
  }

  run<GASEngineGPU<BFS>, true>(nVertices, &vertexData[0], (int) srcs.size()
    , &srcs[0], &dsts[0], sourceVertex);
  if( dumpResults )
  {
    printf("GPU:\n");
    for( int i = 0; i < nVertices; ++i )
      printf("%d %d\n", i, vertexData[i].depth);
  }

  if( runTest )
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

  return 0;
}
