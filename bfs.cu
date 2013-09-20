//BFS using vertexAPI2

#include "util.h"
#include "graphio.h"
#include "refgas.h"



struct BFS
{
  //global iteration count, terrible! OK, while we still figure out
  //how to do the important stuff correctly, but correct method would
  //be to add engine context to all the GAS prototypes
  static int iterationCount;
  
  struct VertexData
  {
    int depth;
  };


  struct EdgeData {}; //nothing


  typedef int GatherResult;
  static int gatherZero; //this should be set to nVertices + 1


  static int gatherReduce(const int& left, const int& right)
  {
    return 0; //do nothing
  }


  static int gatherMap(const VertexData* dst, const VertexData *src, const EdgeData* edge)
  {
    return 0; //do nothing
  }


  static bool apply(VertexData* vert, int dist)
  {
    if( vert->depth == -1 )
    {
      vert->depth = iterationCount;
      return true;
    }
    return false;
  }


  static void scatter(const VertexData* src, const VertexData *dst, EdgeData* edge)
  {
    //nothing
  }
};


int BFS::iterationCount;
int BFS::gatherZero;


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

  BFS::gatherZero = nVertices + 1;

  //initialize vertex data
  std::vector<BFS::VertexData> vertexData;
  for( int i = 0; i < nVertices; ++i )
    vertexData[i].depth = -1; 

  GASEngineRef<BFS> engine;
  engine.setGraph(nVertices, &vertexData[0], srcs.size(), 0, &srcs[0], &dsts[0]);
  engine.setActive(sourceVertex, sourceVertex+1);
  BFS::iterationCount = 0;
  while( engine.nextIter() )
  {
    //run apply without gather
    engine.gatherApply(false);
    //no scatter
    ++BFS::iterationCount;
  }
  engine.getResults();

  //output distances;
}