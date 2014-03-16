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

#ifndef REFGAS_H__
#define REFGAS_H__

#include <vector>
#include <stdio.h>

#include "util.h"

//Reference implementation, useful for correctness checking
//and prototyping interfaces.
//This is not an optimized CPU implementation.


template<typename Program
  , typename Int = int32_t>
class GASEngineRef
{
  typedef typename Program::VertexData   VertexData;
  typedef typename Program::EdgeData     EdgeData;
  typedef typename Program::GatherResult GatherResult;

  Int         m_nVertices;
  Int         m_nEdges;
  VertexData *m_vertexData;
  EdgeData   *m_edgeData;

  //CSC representation for gather phase
  std::vector<Int> m_srcs;
  std::vector<Int> m_srcOffsets;
  std::vector<Int> m_edgeIndexCSC;

  //CSR representation for reduce phase
  std::vector<Int> m_dsts;
  std::vector<Int> m_dstOffsets;
  std::vector<Int> m_edgeIndexCSR;

  //doing similar to the GPU for ease of comparison
  std::vector<GatherResult> m_gatherResults;
  std::vector<Int>  m_active;
  std::vector<Int>  m_applyRet;
  std::vector<bool> m_activeFlags;

  public:
    GASEngineRef()
      : m_nVertices(0)
      , m_nEdges(0)
    {}


    ~GASEngineRef(){}

  
    //initialize the graph data structures for the GPU
    //All the graph data provided here is "owned" by the GASEngine until
    //explicitly released with getResults().  We may make a copy or we
    //may map directly into host memory
    //The Graph is provided here as an edge list.  We internally convert
    //to CSR/CSC representation.  This separates the implementation details
    //from the vertex program.  Can easily add interfaces for graphs that
    //are already in CSR or CSC format.
    //
    //This function is not optimized and at the moment, this initialization
    //is considered outside the scope of the core work on GAS.
    //We will have to revisit this assumption at some point.
    void setGraph(Int nVertices
      , VertexData* vertexData
      , Int nEdges
      , EdgeData* edgeData
      , const Int *edgeListSrcs
      , const Int *edgeListDsts)
    {
      m_nVertices  = nVertices;
      m_nEdges     = nEdges;
      m_vertexData = vertexData;
      m_edgeData   = edgeData;

      //get CSR representation for activate/scatter
      m_dstOffsets.resize(m_nVertices + 1);
      m_dsts.resize(m_nEdges);
      m_edgeIndexCSR.resize(m_nEdges);
      edgeListToCSR(m_nVertices, m_nEdges
        , edgeListSrcs, edgeListDsts
        , &m_dstOffsets[0], &m_dsts[0], &m_edgeIndexCSR[0]);

      //get CSC representation for gather/apply
      m_srcOffsets.resize(m_nVertices + 1);
      m_srcs.resize(m_nEdges);
      m_edgeIndexCSC.resize(m_nEdges);
      edgeListToCSC(m_nVertices, m_nEdges
        , edgeListSrcs, edgeListDsts
        , &m_srcOffsets[0], &m_srcs[0], &m_edgeIndexCSC[0]);

      m_active.reserve(m_nVertices);
      m_applyRet.resize(m_nVertices);
      m_activeFlags.resize(m_nVertices, false);
      m_gatherResults.resize(m_nVertices);
    }


    //This may be a slow function, so normally would only be called
    //at the end of a computation.  This does not invalidate the
    //data already in the engine, but does make sure that the host
    //data is consistent with the engine's internal data
    void getResults()
    {
      //do nothing.
    }


    //set the active flag for a range [vertexStart, vertexEnd)
    //affects only the next gather step
    void setActive(Int vertexStart, Int vertexEnd)
    {
      m_active.clear();
      for( Int i = vertexStart; i < vertexEnd; ++i )
        m_active.push_back(i);
    }


    //Return the number of active vertices in the next gather step
    Int countActive()
    {
      return m_active.size();
    }


    void gatherApply(bool haveGather=true)
    {
      for( Int i = 0; i < m_active.size(); ++i )
      {
        Int dv = m_active[i];
        GatherResult sum = Program::gatherZero;
        Int edgeStart = m_srcOffsets[dv];
        Int edgeEnd   = m_srcOffsets[dv + 1];
        for( Int ie = edgeStart; ie < edgeEnd; ++ie )
        {
          Int src = m_srcs[ie];
          GatherResult tmp = Program::gatherMap(m_vertexData + dv
            , m_vertexData + src, m_edgeData + m_edgeIndexCSC[ie]);
          sum = Program::gatherReduce(sum, tmp);
        }
        m_gatherResults[i] = sum;
      }

      //separate loop to keep bulk synchronous
      for( Int i = 0; i < m_active.size(); ++i )
      {
        Int dv = m_active[i];
        m_applyRet[i] = Program::apply(m_vertexData + dv, m_gatherResults[i]);
      }
    }


    //do the scatter operation
    void scatterActivate(bool haveScatter=true)
    {
      m_activeFlags.clear();
      m_activeFlags.resize(m_nVertices, false);
      for( Int i = 0; i < m_active.size(); ++i )
      {
        //only run scatter if the vertex has requested its nbd
        //activated for the next step.
        if( m_applyRet[i] )
        {
          Int sv = m_active[i];
          Int edgeStart = m_dstOffsets[sv];
          Int edgeEnd   = m_dstOffsets[sv + 1];
          for( Int ie = edgeStart; ie < edgeEnd; ++ie )
          {
            Int dv = m_dsts[ie];
            m_activeFlags[dv] = true;
            if( haveScatter )
            {
               Program::scatter(m_vertexData + sv, m_vertexData + dv
                , m_edgeData + m_edgeIndexCSR[ie]);
            }
          }
        }
      }
    }


    //sets up the engine for the next iteration
    //returns the number of active vertices
    Int nextIter()
    {
      m_active.clear();
      for( Int i = 0; i < m_nVertices; ++i )
      {
        if( m_activeFlags[i] )
          m_active.push_back(i);
      }
      return countActive();
    }
  

    //single entry point for the whole affair, like before.
    //Need to improve the key steps to make it more flexible.
    //Todo.
    void run()
    {
      while( countActive() )
      {
        gatherApply();
        scatterActivate();
        nextIter();
      }
    }
};  



#endif
