/******************************************************************************
Copyright 2013 SYSTAP, LLC. http://www.systap.com

Written by Erich Elsen and Vishal Vaidyanathan
of Royal Caliber, LLC
Contact us at: info@royal-caliber.com

This file was taken from mpgraph v0.1 which was (partially) funded by the
DARPA XDATA program under AFRL Contract #FA8750-13-C-0002.  The file has
been modified by Royal Caliber, LLC.

Copyright 2013, 2014 Royal Caliber LLC. (http://www.royal-caliber.com)

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

#ifndef GRAPHIO_H__
#define GRAPHIO_H__

//Some utilities for loading graph data

#include <string>
#include <vector>

//Read in a snap format graph
int loadGraph_GraphLabSnap( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts );


//Read in a MatrixMarket coordinate format graph
int loadGraph_MatrixMarket( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts
  , std::vector<int> *edgeValues );


//Read in a binary CSR graph (Lonestar format)
//If expand is true, converts CSR into list of edges
//to be compatible with the other loaders, otherwise
//the argument srcs will contain nVertices + 1 offsets
int loadGraph_binaryCSR(const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts
  , std::vector<int> *edgeValues
  , bool expand = true);


//Detects the filetype from the extension
int loadGraph( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts
  , std::vector<int> *edgeValues = 0);


//write out a lonestar format binary csr file
int writeGraph_binaryCSR(const char* fname
  , int nVertices, int nEdges, const int *offsets, const int* dsts
  , const int *edgeValues);
  
int writeGraph_mtx(const char* fname, int nVertices, int nEdges
  , const int *srcs, const int *dsts, const int* edgeValues);

#endif
