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
  else
  {
    cerr << "unrecognized filetype extension " << p << endl;
    exit(1);
  }

  return 0;
}



