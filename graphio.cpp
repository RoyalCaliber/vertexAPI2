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


#include "graphio.h"
#include <zlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstdlib>
#include <stdint.h>

using namespace std;


enum SymmetryType { stNone, stSymmetric, stSkewSymmetric, stHermitian };


static bool isBlankLine( const char* line )
{
  while( *line )
  {
    if( !isspace( *line ) )
      return false;
    ++line;
  }
  return true;
}


static gzFile openFile( const char* fname )
{
  gzFile f = gzopen( fname, "rb" );
  if( !f )
  {
    cerr << "error opening file " << fname << endl;
    exit(1);
  }
  return f;
}

//common code for some simple line-based graph formats
//This is totally hacky right now - if we really need to
//implement IO code, we need to survey the different formats
//and figure out the correct abstraction.
static int loadGraph_common( gzFile f
  , char commentChar
  , bool ignoreFirstDataLine
  , bool decrementIndices
  , bool disallowSelfLinks
  , SymmetryType symType
  , int &nVertices
  , std::vector<int> *srcs
  , std::vector<int> *dsts
  , std::vector<int> *edgeValues )
{
  char buffer[1024];
  int lineNum = 0;
  int maxVertex = 0;
  bool firstDataLine = true;
  while( !gzeof(f) )
  {
    if( !gzgets( f, buffer, sizeof(buffer) ) )
    {
      int err;
      gzerror( f, &err );
      if( err )
      {
        cerr << "gz error " << err << " at line " << lineNum << endl;
        exit(1);
      }
      else
        continue;
    }

    ++lineNum; //one-based line numbers!

    //empty line means eof
    if( buffer[0] == 0 )
      continue;

    //ignore comments and blank lines
    if( buffer[0] == commentChar || isBlankLine( buffer ) )
      continue;

    if( ignoreFirstDataLine && firstDataLine )
    {
      int nVerticesX, nVerticesY, nEdges;
      sscanf(buffer, "%d%d%d", &nVerticesX, &nVerticesY, &nEdges);
      firstDataLine = false;
      continue;
    }

    int src, dst;
    int nbytes;
    int np = sscanf( buffer, "%d%d%n", &src, &dst, &nbytes );
    if( np != 2 )
    {
      cerr << "error parsing src/dst at line " << lineNum << endl;
      exit(1);
    }

    //use 0-based indexing
    if( decrementIndices )
    {
      --src;
      --dst;
    }

    if (disallowSelfLinks && src == dst) {
      //scan for an associated edge value
      //and then ignore it
      if ( edgeValues ) {
        float edgeValue;
        sscanf( buffer + nbytes, "%f", &edgeValue );
        continue;
      }
    }
    else {
      srcs->push_back( src );
      dsts->push_back( dst );

      if( symType != stNone )
      {
        srcs->push_back( dst );
        dsts->push_back( src );
      }
    }

    if( edgeValues )
    {
      float edgeValue;
      sscanf( buffer + nbytes, "%f", &edgeValue );
      edgeValues->push_back( edgeValue );

      switch( symType )
      {
        case stNone: break; //do nothing
        case stSymmetric: edgeValues->push_back( edgeValue ); break;
        case stSkewSymmetric: edgeValues->push_back( -edgeValue ); break;
        case stHermitian: break; //unsupported
      }
    }

    int tmp = max( src, dst );
    if( tmp > maxVertex )
      maxVertex = tmp;
  }

  nVertices = maxVertex + 1;
  return 0;
}


int loadGraph_GraphLabSnap( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts )
{
  gzFile f = openFile( fname );
  int ret = loadGraph_common( f, '#', false, false, true, stNone, nVertices, &srcs, &dsts, 0 );
  gzclose(f);
  return ret;
}


int loadGraph_MatrixMarket( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts
  , std::vector<int> *edgeValues )
{
  gzFile f = openFile( fname );
  //first comment line is special
  char line[1024];
  if( !gzgets( f, line, sizeof(line) ) )
  {
    cerr << "error reading header" << endl;
    exit(1);
  }

  char* p;
  char *tok;
  const char* delim = " \n";
  strtok_r( line, delim, &p );

  //expect matrix.
  tok = strtok_r( 0, delim, &p );
  if( strcmp( tok, "matrix" ) != 0 )
  {
    cerr << "unrecognized token " << tok << " in header" << endl;
    exit(1);
  }

  //format should be coordinate
  tok = strtok_r( 0, delim, &p );
  if( strcmp( tok, "coordinate" ) != 0 )
  {
    cerr << "only coordinate format is supported" << endl;
    exit(1);
  }

  //edge data type
  tok = strtok_r( 0, delim, &p );
  if( strcmp( tok, "pattern" ) == 0 )
  {
    if( edgeValues )
    {
      cerr << "warning: graph does not have edge values" << endl;
      edgeValues = 0;
    }
  }
  else if( strcmp( tok, "complex" ) == 0 )
  {
    cerr << "complex edge values not supported" << endl;
    exit(1);
  }

  //symmetry
  tok = strtok_r( 0, delim, &p );
  SymmetryType st;
  if( strcmp( tok, "general" ) == 0 )
    st = stNone;
  else if( strcmp( tok, "symmetric" ) == 0 )
    st = stSymmetric;
  else if( strcmp( tok, "skew-symmetric" ) == 0 )
    st = stSkewSymmetric;
  else if( strcmp( tok, "hermitian" ) == 0 )
    st = stHermitian;
  else
  {
    cerr << "unrecognized symmetry type '" << tok << "'" <<  endl;
    exit(1);
  }

  int ret = loadGraph_common( f, '%', true, true, true, st, nVertices, &srcs, &dsts, edgeValues );
  gzclose(f);
  return ret;
}


//Lonestar format binary CSR
//Format: all data is little endian
//8-byte version
//8-byte unsigned nVertices
//8-byte unsigned nEdges
//8-byte * nVertices ending offsets into dsts array
//4-byte * nEdges dst indices, padded to 8-byte boundary at end
//4-byte * nEdges unsigned edge data
//This loader assumes a little endian host.
int loadGraph_binaryCSR(const char* fname
  , int &nVertices, std::vector<int> &srcs, std::vector<int> &dsts
  , std::vector<int> *edgeValues, bool expand)
{
  #define CHK_FREAD(ptr, sz, count, stream) \
    if (fread(ptr, sz, count, stream) != count) \
    { \
      cerr << "error reading lonestar binary CSR file " << fname << endl; \
      exit(1); \
    } \
  
  FILE* f = fopen(fname, "r");
  if (!f)
  {
    cerr << "unable to open file " << fname << endl;
    exit(1);
  }
  
  uint64_t version;
  CHK_FREAD(&version, 8, 1, f);
  printf("file version = %lu\n", version);

  uint64_t sizeEdgeType;
  CHK_FREAD(&sizeEdgeType, 8, 1, f);
  if (sizeEdgeType == 0 && edgeValues)
  {
    cerr << "file does not have edge values" << endl;
    exit(1);
  }
  else if (sizeEdgeType != 4)
  {
    cerr << "file edge data is " << sizeEdgeType << " bytes wide" << endl;
    exit(1);
  }

  uint64_t nVertices64;
  CHK_FREAD(&nVertices64, 8, 1, f);
  nVertices = nVertices64;
  printf("nVertices = %lu\n", nVertices64);

  uint64_t nEdges;
  CHK_FREAD(&nEdges, 8, 1, f);
  printf("nEdges = %lu\n", nEdges);

  dsts.resize(nEdges);
  if (expand)
    srcs.resize(nEdges);
  else
    srcs.resize(nVertices + 1);

  //read in the offsets
  std::vector<uint64_t> tmp(nVertices);
  CHK_FREAD(&tmp[0], 8, nVertices, f);
  printf("read in %d offsets\n", nVertices);

  //read in dst indices and consume padding if any
  CHK_FREAD(&dsts[0], 4, nEdges, f);
  printf("read in %lu dsts\n", nEdges);

  if (nEdges % 2)
  {
    uint32_t dummy;
    CHK_FREAD(&dummy, 4, 1, f);
  }

  if (expand)
  {
    int j = 0;
    for (int i = 0; i < nVertices; ++i)
    {
      for (; j < tmp[i]; ++j)
        srcs[j] = i;
    }
  }
  else
  {
    for (int i = 0; i < nVertices; ++i)
      srcs[i + 1] = tmp[i];
  }

  if (edgeValues)
  {
    edgeValues->resize(nEdges);
    CHK_FREAD(&((*edgeValues)[0]), 4, nEdges, f);
  }
  
  fclose(f);
  #undef CHK_FREAD
}


int writeGraph_binaryCSR(const char* fname
  , int nVertices, int nEdges, const int *offsets, const int* dsts
  , const int *edgeValues)
{
  FILE *f = fopen(fname, "w");
  if (!f)
  {
    cerr << "unable to write to file " << fname << endl;
    exit(1);
  }
  
  uint64_t u64;

  //version
  u64 = 1; 
  fwrite(&u64, 8, 1, f); 

  //sizeEdgeType
  u64 = edgeValues ? 4 : 0;
  fwrite(&u64, 8, 1, f);

  //nVertices
  u64 = nVertices;
  fwrite(&u64, 8, 1, f);

  //nEdges
  u64 = nEdges;
  fwrite(&u64, 8, 1, f);

  //write offsets, without first zero
  std::vector<uint64_t> tmp(nVertices);
  for (int i = 0; i < nVertices; ++i)
    tmp[i] = offsets[i + 1];
  fwrite(&tmp[0], 8, nVertices, f);

  //write dsts, add padding
  fwrite(dsts, 4, nEdges, f);
  if (nEdges % 2)
  {
    uint32_t pad = 0;
    fwrite(&pad, 4, 1, f);
  }

  //write edge values if present
  if (edgeValues)
    fwrite(edgeValues, 4, nEdges, f);
  
  fclose(f);
} 



int loadGraph( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts
  , std::vector<int> *edgeValues )
{
  const char*p = fname;
  while( *p )
    ++p;
  while( p >= fname && *p != '.' )
    --p;

  if( strcmp( p, ".gz" ) == 0 )
  {
    --p;
    while( p >= fname && *p != '.' )
      --p;
  }

  if( strncmp( p, ".edge", 5 ) == 0 )
    return loadGraph_GraphLabSnap( fname, nVertices, srcs, dsts );
  else if( strncmp( p, ".mtx", 4 ) == 0 )
    return loadGraph_MatrixMarket( fname, nVertices, srcs, dsts, edgeValues );
  else if( strncmp( p, ".gr", 3 ) == 0 )
    return loadGraph_binaryCSR( fname, nVertices, srcs, dsts, edgeValues, true );
  else
  {
    cerr << "unrecognized filetype extension " << p << endl;
    exit(1);
  }

  return 0;
}



