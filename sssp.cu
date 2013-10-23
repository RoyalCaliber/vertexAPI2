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
void run(int nVertices, SSSP::VertexData* vertexData, int nEdges
  , SSSP::EdgeData* edgeData, const int* srcs, const int* dsts)
{
    Engine engine;
    engine.setGraph(nVertices, vertexData, nEdges, edgeData, srcs, dsts);

    //TODO, setting all vertices to active for first step works, but it would
    //be faster to instead set to neighbors of starting vertex
    engine.setActive(0, nVertices);
    engine.run();
    engine.getResults();
}


void outputDists(int nVertices, int* dists, FILE* f = stdout)
{
  for (int i = 0; i < nVertices; ++i)
    fprintf(f, "%d %d\n", i, dists[i]);
}


int main(int argc, char** argv)
{
  char *inputFilename;
  char *outputFilename = 0;
  int sourceVertex;
  bool runTest;
  bool dumpResults;
  if( !parseCmdLineSimple(argc, argv, "si-t-d|s", &inputFilename, &sourceVertex
    , &runTest, &dumpResults, &outputFilename) )
  {
    printf("Usage: sssp [-t] [-d] inputfile source [outputfile]\n");
    exit(1);
  }

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  std::vector<int> edgeData;
  loadGraph(inputFilename, nVertices, srcs, dsts, &edgeData);
  if( edgeData.size() == 0 )
  {
    printf("No edge data available in input file\n");
    exit(1);
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

  run< GASEngineGPU<SSSP> >(nVertices, &vertexData[0], (int)srcs.size()
    , &edgeData[0], &srcs[0], &dsts[0]);
  if( dumpResults )
  {
    printf("GPU:\n");
    outputDists(nVertices, &vertexData[0]);
  }

  if( runTest )
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

  if( outputFilename )
  {
    printf("writing results to %s\n", outputFilename);
    FILE* f = fopen(outputFilename, "w");
    outputDists(nVertices, &vertexData[0], f);
    fclose(f);
  }

  free(inputFilename);
  free(outputFilename);

  return 0;
}
