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


//Detects the filetype from the extension
int loadGraph( const char* fname
  , int &nVertices
  , std::vector<int> &srcs
  , std::vector<int> &dsts
  , std::vector<int> *edgeValues = 0);

#endif
