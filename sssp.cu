//Single source shortest paths using vertexAPI2

#include "util.h"
#include "graphio.h"
#include "refgas.h"
#include "gpugas.h"
#include <climits>

struct SSSP
{
  struct VertexData
  {
    int dist;
  };


  struct EdgeData
  {
    int length;
  };


  typedef int GatherResult;
  static const int gatherZero = INT_MAX - 100000; //this should be set to nVertices + 1


  __host__ __device__
  static int gatherReduce(const int& left, const int& right)
  {
    return min(left, right);
  }


  __host__ __device__
  static int gatherMap(const VertexData* dst, const VertexData *src, const EdgeData* edge)
  {
    return src->dist + edge->length;
  }


  __host__ __device__
  static bool apply(VertexData* vert, int dist)
  {
    bool changed = dist < vert->dist;
    vert->dist = min(vert->dist, dist);
    return changed;
  }


  __host__ __device__
  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


int main(int argc, char** argv)
{
  char *inputFilename;
  int sourceVertex;
  if( !parseCmdLineSimple(argc, argv, "si", &inputFilename, &sourceVertex) )
    exit(1);

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  std::vector<int> edge_data;
  loadGraph(inputFilename, nVertices, srcs, dsts, &edge_data);

  //initialize vertex data
  std::vector<SSSP::VertexData> vertexData(nVertices);

  {
    for( int i = 0; i < nVertices; ++i ) {
      if (i != sourceVertex)
        vertexData[i].dist = nVertices + 1; //larger than max diameter
      else
        vertexData[i].dist = 0;
    }

    GASEngineRef<SSSP> engine;
    engine.setGraph(nVertices, &vertexData[0], srcs.size(), (SSSP::EdgeData *)&edge_data[0], &srcs[0], &dsts[0]);

    //TODO, setting all vertices to active for first step works, but it would
    //be faster to instead set to neighbors of starting vertex
    engine.setActive(0, nVertices);
    engine.run();
    engine.getResults();

    //output distances;
    for (int i = 0; i < nVertices; ++i)
      printf("%d %d\n", i, vertexData[i].dist);
  }

   {
    for( int i = 0; i < nVertices; ++i ) {
      if (i != sourceVertex)
        vertexData[i].dist = nVertices + 1; //larger than max diameter
      else
        vertexData[i].dist = 0;
    }

    GASEngineGPU<SSSP> engine;
    engine.setGraph(nVertices, &vertexData[0], srcs.size(), (SSSP::EdgeData *)&edge_data[0], &srcs[0], &dsts[0]);

    //TODO, setting all vertices to active for first step works, but it would
    //be faster to instead set to neighbors of starting vertex
    engine.setActive(0, nVertices);
    engine.run();
    engine.getResults();

    //output distances;
    for (int i = 0; i < nVertices; ++i)
      printf("%d %d\n", i, vertexData[i].dist);
   }
}
