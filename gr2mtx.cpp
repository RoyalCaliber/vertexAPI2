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
