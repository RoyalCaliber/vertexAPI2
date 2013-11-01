#ifndef REFGAS_H__
#define REFGAS_H__

#include <vector>
#include <stdio.h>

#ifdef VERTEXAPI_USE_MPI
#include <mpi.h>
#endif


//Reference implementation, useful for correctness checking
//and prototyping interfaces.
//This is not an optimized CPU implementation.


//Doing a first-pass MPI implementation.  This is not likely
//to be scalable to more than a few nodes, but helps us solve
//some API questions for the scalable version.


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

  #ifdef VERTEXAPI_USE_MPI
  int          m_mpiRank;
  MPI_Datatype m_mpiGatherResultType;
  MPI_Op       m_mpiReduceOp;
  #endif

  public:
    GASEngineRef()
      : m_nVertices(0)
      , m_nEdges(0)
    {}


    ~GASEngineRef(){}


    #ifdef VERTEXAPI_USE_MPI
    //wrap Program::gatherReduce for MPI
    static void mpiReduce(const GatherResult *in
      , GatherResult* inout
      , const int *len
      , MPI_Datatype *type)
    {
      for( int i = 0; i < *len; ++i )
        inout[i] = Program::gatherReduce(inout[i], in[i]);
    }


    void initMPI()
    {
      //errcheck needed
      MPI_Comm_rank(MPI_COMM_WORLD, &m_mpiRank);
      
      //This assumes the same endianness, word size and packing for all
      //processes. Otherwise we need some help from the end user in defining the
      //MPI_Datatype corresponding to Program::GatherResult
      //This means: use the same compiler, same 32/64-bitness and same architecture
      //on all processes for this to work.
      MPI_Type_contiguous(sizeof(GatherResult), MPI_CHAR, &m_mpiGatherResultType);
      MPI_Type_commit(&m_mpiGatherResultType);
      
      MPI_Op_create((MPI_User_function*) mpiReduce, Program::Commutative, &m_mpiReduceOp);
    }
    #endif

  
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
      //the temporary gather results are now nVertices sized,
      //where nVertices is the total number of vertices across all nodes
      m_gatherResults.clear();
      m_gatherResults.resize(m_nVertices, Program::gatherZero);
      
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
        m_gatherResults[dv] = sum;
      }

      #ifdef VERTEXAPI_USE_MPI
      for( Int i = 0; i < m_nVertices; ++i )
      {
        printf("%d %d %f\n", m_mpiRank, i, m_gatherResults[i]);
      }
      printf("finished pre\n");
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &m_gatherResults[0], m_nVertices
        , m_mpiGatherResultType, m_mpiReduceOp, MPI_COMM_WORLD);
      for( Int i = 0; i < m_nVertices; ++i )
      {
        printf("after %d %d %f\n", m_mpiRank, i, m_gatherResults[i]);
      }
      #endif

      //separate loop to keep bulk synchronous
      for( Int i = 0; i < m_active.size(); ++i )
      {
        Int dv = m_active[i];
        m_applyRet[i] = Program::apply(m_vertexData + dv, m_gatherResults[dv]);
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
