#ifndef REFGAS_H__
#define REFGAS_H__

#include <vector>

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
  const Int  *m_edgeListSrcs; 
  const Int  *m_edgeListDsts;

  //CSC representation for gather phase
  std::vector<Int> m_srcs;
  std::vector<Int> m_srcOffsets;
  std::vector<Int> m_edgeIndexCSC;

  //CSR representation for reduce phase
  std::vector<Int> m_dsts;
  std::vector<Int> m_dstOffsets;
  std::vector<Int> m_edgeIndexCSR;

  std::vector<bool> m_active;
  std::vector<bool> m_activeNextStep;

  public:
    GASEngineRef()
      : m_nVertices(0)
      , m_nEdges(0)
      , m_edgeListSrcs(0)
      , m_edgeListDsts(0)
    {}

  
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
      , Int *src
      , Int *dst)
    {
      m_nVertices  = nVertices;
      m_nEdges     = nEdges;
      m_vertexData = vertexData;
      m_edgeData   = edgeData;
      m_edgeListSrcs = src;
      m_edgeListDsts = dst;

      //get CSC representation for gather/apply
      m_srcOffsets.resize(m_nVertices + 1);
      m_srcs.resize(m_nEdges);
      m_edgeIndexCSC.resize(m_nEdges);
      edgeListToCSC(m_nVertices, m_nEdges
        , m_edgeListSrcs, m_edgeListDsts
        , &m_srcOffsets[0], &m_srcs[0], &m_edgeIndexCSC[0]);

      //get CSR representation for activate/scatter
      m_dstOffsets.resize(m_nVertices + 1);
      m_dsts.resize(m_nEdges);
      m_edgeIndexCSR.resize(m_nEdges);
      edgeListToCSR(m_nVertices, m_nEdges
        , m_edgeListSrcs, m_edgeListDsts
        , &m_dstOffsets[0], &m_dsts[0], &m_edgeIndexCSR[0]);
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
    void setActive(Int vertexStart, Int vertexEnd, bool value)
    {
      for( Int i = vertexStart; i < vertexEnd; ++i )
        m_activeNextStep[i] = true;
    }


    //Return the number of active vertices in the next gather step
    Int countActiveNext()
    {
      Int count = 0;
      for( Int i = 0; i < m_nVertices; ++i )
        count += m_activeNextStep[i];
      return count;
    }


    void gatherApply(bool haveGather=true)
    {
      //initialize active flags.
      m_active.swap(m_activeNextStep);
      m_activeNextStep.clear();
      m_activeNextStep.resize(m_nVertices, false);
      
      for( Int dv = 0; dv < m_nVertices; ++dv )
      {
        if( m_active[dv] )
        {
          GatherResult sum = Program::gatherZero;
          const VertexData *dstVert = m_vertexData ? m_vertexData + dv : 0;

          if( haveGather )
          {
            Int edgeStart = m_srcOffsets[dv];
            Int edgeEnd   = m_srcOffsets[dv + 1];
          
            //do gather map-reduce
            for( Int ie = edgeStart; ie < edgeEnd; ++ie )
            {
              const VertexData *srcVert = m_vertexData ? m_vertexData + m_srcs[ie] : 0;
              const EdgeData *edgeData = m_edgeData ? m_edgeData + m_edgeIndexCSC[ie] : 0;
              sum = Program::gatherReduce(sum, Program::gatherMap(dstVert, srcVert, edgeData));
            }
          }

          //do apply
          bool ret = Program::apply( const_cast<VertexData*>(dstVert), sum );

          //if the return value is true, we activate all the neighbors
          //for the next step.  Doing it here instead of in scatter to force the
          //Program to choose between activating all neighbors and no neighbors
          //which may be faster on a GPU.  Don't know yet whether this works for
          //all types of Programs we'd be interested in.
          if( ret )
          {
            Int edgeStart = m_dstOffsets[dv];
            Int edgeEnd   = m_dstOffsets[dv + 1];
            for( Int ie = edgeStart; ie < edgeEnd; ++ie )
              m_activeNextStep[ m_dsts[ie] ] = true;
          }
        }
      }
    }


    void apply()
    {
      gatherApply(false);
    }


    //do the scatter operation
    //wrote this similar to how we might do it on the GPU to help debugging
    //on the CPU there is no need to use the CSR ordering at all.
    void scatter()
    {
      for( Int sv = 0; sv < m_nVertices; ++sv )
      {
        if( m_active[sv] )
        {
          Int edgeStart = m_dstOffsets[sv];
          Int edgeEnd   = m_dstOffsets[sv + 1];
          const VertexData *srcVert = m_vertexData ? m_vertexData + sv : 0;
          for( Int ie = edgeStart; ie < edgeEnd; ++ie )
          {
            Int dv = m_dsts[ie];
            const VertexData *dstVert = m_vertexData ? m_vertexData + dv : 0;
            EdgeData* edgeData = m_edgeData ? m_edgeData + m_edgeIndexCSR[ie] : 0;
            Program::scatter(srcVert, dstVert, edgeData);
          }
        }
      }
    }


    //single entry point for the whole affair, like before.
    //special cases that don't need gather or scatter can
    //easily roll their own loop.
    void run()
    {
      while( countActiveNext() )
      {
        gatherApply();

        //can skip scatter if the algorithm has no edge data.
        if( m_edgeData )
          scatter();
      }
      getResults();
    }
};  



#endif
