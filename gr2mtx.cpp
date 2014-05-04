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

//Utility to convert .gr to mtx for programs that can't read gr

#include "util.h"
#include "graphio.h"

int main(int argc, char **argv)
{
  char *inputFilename;
  char *outputFilename;

  if (!parseCmdLineSimple(argc, argv, "ss", &inputFilename, &outputFilename))
  {
    printf("Usage: gr2mtx input output\n");
    exit(1);
  }

  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  std::vector<int> edgeValues;
  loadGraph(inputFilename, nVertices, srcs, dsts, &edgeValues);
  printf("Read input file with %d vertices and %zd edges\n", nVertices, dsts.size());

  printf("writing output\n");
  writeGraph_mtx(outputFilename, nVertices, dsts.size()
    , &srcs[0], &dsts[0], &edgeValues[0]);
}
