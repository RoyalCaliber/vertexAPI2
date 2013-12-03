#include "graphio.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char **argv) {
  if (argc % 2 != 0) {
    std::cerr << "Usage: ./createCCGraph graph repetitions [graph] [repetitions] [graph] [repetitions] outputfile" << std::endl;
    exit(1);
  }

  int numGraphs = (argc - 2) / 2;

  std::vector<std::vector<int> > srcs(numGraphs);
  std::vector<std::vector<int> > dsts(numGraphs);
  std::vector<int> nVertices(numGraphs);
  std::vector<int> numReps(numGraphs);

  //for now we load all graphs into memory at once - could be changed to be less
  //memory intensive at the cost of loading the graphs twice
  long totalVertices = 0;
  long totalEdges = 0;
  for (int i = 0; i < numGraphs; ++i) {
    loadGraph_MatrixMarket(argv[2 * i + 1], nVertices[i], srcs[i], dsts[i], NULL);
    numReps[i] = atoi(argv[2 * i + 2]);
    totalVertices += nVertices[i] * numReps[i];
    totalEdges += srcs[i].size() * numReps[i];
  }

  FILE *f = fopen(argv[argc - 1], "w");

  fprintf(f, "%%%%MatrixMarket matrix coordinate Pattern symmetric\n");
  fprintf(f, "%zd %zd %zd\n", totalVertices, totalVertices, totalEdges);

  long previousVertices = 0;
  for (int g = 0; g < numGraphs; ++g) {
    for (int r = 0; r < numReps[g]; ++r) {
      for (int e = 0; e < srcs[g].size(); ++e) {
        //+1 is because the loading routine shifted down by 1, so need to shift back up
        fprintf(f, "%zd %zd\n", srcs[g][e] + previousVertices + 1, dsts[g][e] + previousVertices + 1);
      }
      previousVertices += nVertices[g];
    }
  }

  fclose(f);

  return 0;
}
