//Single source shortest paths using vertexAPI2

#include "util.h"
#include "graphio.h"
#include "refgas.h"

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
  static int gatherZero; //this should be set to nVertices + 1


  static int gatherReduce(const int& left, const int& right)
  {
    return min(left, right);
  }


  static int gatherMap(const VertexData* dst, const VertexData *src, const EdgeData* edge)
  {
    return src->dist + edge->length;
  }


  static bool apply(VertexData* vert, int dist)
  {
    bool changed = dist != vert->dist;
    vert->dist = dist;
    return changed;
  }


  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


//This is not a const because we set to nVertices+1 after loading the graph
int SSSP::gatherZero;


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
  loadGraph(inputFilename, nVertices, srcs, dsts);

  //initialize vertex data
  std::vector<SSSP::VertexData> vertexData(nVertices);
  for( int i = 0; i < nVertices; ++i )
    vertexData[i].dist = nVertices + 1; //larger than max diameter
  SSSP::gatherZero = nVertices + 1;

  GASEngineRef<SSSP> engine;
  engine.setGraph(nVertices, &vertexData[0], srcs.size(), 0, &srcs[0], &dsts[0]);
  engine.setActive(sourceVertex, sourceVertex+1);
  engine.run();
  engine.getResults();

  //output distances;
  for (int i = 0; i < nVertices; ++i)
    printf("%d %d\n", i, vertexData[i].dist);
}
