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


int main(int argc, char **argv)
{
  char* inputFilename;
  if( !parseCmdLineSimple(argc, argv, "s", &inputFilename) )
    exit(1);

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  loadGraph(inputFilename, nVertices, srcs, dsts);

  //initialize vertex data
  std::vector<PageRank::VertexData> vertexData(nVertices);
  for( int i = 0; i < nVertices; ++i )
    vertexData[i].rank = PageRank::pageConst;

  //instantiate and run the engine to completion
  {
    GASEngineRef<PageRank> engine;
    engine.setGraph(nVertices, &vertexData[0], srcs.size(), 0, &srcs[0], &dsts[0]);
    //all vertices begin active for pagerank
    engine.setActive(0, nVertices);
    engine.run();
    engine.getResults();

    //output ranks.
  }

  {
    //Repeat the calculation with the GPU engine
    GASEngineGPU<PageRank> engine;
    engine.setGraph(nVertices, &vertexData[0], srcs.size(), 0, &srcs[0], &dsts[0]);
    engine.setActive(0, nVertices);
    engine.run();
    engine.getResults();

    //output ranks
  }
}

