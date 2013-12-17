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

//Utility program to calculate the out-degree of each vertex
//The python script was too slow and memory hungry to work
//for the large graphs

#include "graphio.h"
#include <cstdio>
#include <cstdlib>
#include <vector>


int main( int argc, char** argv )
{
  if( argc != 3 )
  {
    printf("Usage: outdeg input.mtx output\n");
    exit(1);
  }

  const char* inputFilename  = argv[1];
  const char* outputFilename = argv[2];

  int nVertices;
  std::vector<int> srcs, dsts;
  loadGraph( inputFilename, nVertices, srcs, dsts );

  std::vector<int> count(nVertices);
  for( size_t i = 0; i < srcs.size(); ++i )
    ++count[srcs[i]];

  FILE* f = fopen( outputFilename, "w" );
  for( int i = 0; i < nVertices; ++i )
    fprintf( f, "%d\n", count[i] );
  fclose(f);
}
