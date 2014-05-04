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

//Utility to convert mtx files to .gr for faster loading.

#include "util.h"
#include "graphio.h"

int main(int argc, char **argv)
{
  char *inputFilename;
  char *outputFilename;

  if (!parseCmdLineSimple(argc, argv, "ss", &inputFilename, &outputFilename))
  {
    printf("Usage: mtx2gr input output\n");
    exit(1);
  }

  int nVertices;
  std::vector<int> srcs;
  std::vector<int> dsts;
  std::vector<int> edgeValues;
  loadGraph(inputFilename, nVertices, srcs, dsts, &edgeValues);
  printf("Read input file with %d vertices and %zd edges\n", nVertices, dsts.size());

  printf("Converting to CSR\n");
  std::vector<int> offsets(nVertices + 1);
  std::vector<int> csrDsts(dsts.size());
  std::vector<int> sortIndices(dsts.size());
  std::vector<int> sortedEdgeValues(dsts.size());
  edgeListToCSR<int>(nVertices, dsts.size(), &srcs[0], &dsts[0]
    , &offsets[0], &csrDsts[0], &sortIndices[0]);
  for (size_t i = 0; i < sortIndices.size(); ++i)
    sortedEdgeValues[i] = edgeValues[sortIndices[i]];

  printf("writing output\n");
  writeGraph_binaryCSR(outputFilename, nVertices, dsts.size()
    , &offsets[0], &csrDsts[0], &sortedEdgeValues[0]);
}
