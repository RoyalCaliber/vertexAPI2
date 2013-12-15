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
void run(int nVertices, CC::VertexData* vertexData, int nEdges
       , const int* srcs, const int* dsts)
{
  Engine engine;
  engine.setGraph(nVertices, vertexData, nEdges, 0, srcs, dsts);

  //TODO, setting all vertices to active for first step works, but it would
  //be faster to instead set to neighbors of starting vertex
  engine.setActive(0, nVertices);
  int64_t t0 = currentTime();
  engine.run();
  engine.getResults();
  int64_t t1 = currentTime();
  printf("Took %f ms\n", (t1 - t0)/1000.0f);
}


void outputLabels(int nVertices, int* labels, FILE* f = stdout)
{
  for (int i = 0; i < nVertices; ++i)
    fprintf(f, "%d %d\n", i, labels[i]);
}


int main(int argc, char** argv)
{
  char *inputFilename;
  char *outputFilename = 0;
  bool runTest;
  bool dumpResults;
  if( !parseCmdLineSimple(argc, argv, "s-t-d|s", &inputFilename
                        , &runTest, &dumpResults, &outputFilename) )
  {
    printf("Usage: cc [-t] [-d] inputfile source [outputfile]\n");
    exit(1);
  }

  //load the graph
  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  loadGraph(inputFilename, nVertices, srcs, dsts);
  printf("loaded %s with %d vertices and %zd edges\n", inputFilename, nVertices, srcs.size());

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
    if( dumpResults )
    {
      printf("Reference:\n");
      outputLabels(nVertices, &refVertexData[0]);
    }  
  }

  run< GASEngineGPU<CC> >(nVertices, &vertexData[0], (int)srcs.size()
                          , &srcs[0], &dsts[0]);
  if( dumpResults )
  {
    printf("GPU:\n");
    outputLabels(nVertices, &vertexData[0]);
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
    outputLabels(nVertices, &vertexData[0], f);
    fclose(f);
  }

  free(inputFilename);
  free(outputFilename);

  return 0;
}
