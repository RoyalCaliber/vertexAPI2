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

#ifndef GPUGAS_H__
#define GPUGAS_H__

/*
Second iteration of a CUDA implementation for GPUs.
The primary difference in this version as opposed to the first round is that we
maintain a compact list of active vertices as opposed to always working on the
entire graph.

There are pros and cons to using an active vertex list vs always working on
everything:

pros:
-  improved performance where the active set is much smaller than the whole graph

cons:
-  an active vertex list requires additional calculations to load balance
   properly.  For both gather and scatter, we need to dynamically figure out the
   mapping between threads and the edge(s) they are responsible for.

-  scattering with an active vertex list requires us to be able to look up the
   outgoing edges given a vertex id.  This means that in addition to the CSC
   representation used in the gather portion, we also need the CSR representation
   for the scatter.  This doubles the edge storage requirement.


Implementation Notes(VV):
-  decided to move away from thrust because thrust requires us to compose at the
   host level and there are unavoidable overheads to that approach.

-  between CUB and MGPU, MGPU offers a few key pieces that we want to use, namely
   LBS and IntervalMove.  Both CUB and MGPU have their own slightly different
   ways of doing things host-side.  Since neighter CUB nor MGPU seem to be stable
   APIs, this implementation chooses an MGPU/CUB-neutral way of doing things
   wherever possible.

-  Program::apply() now returns a boolean, indicating whether to active its entire
   neighborhood or not.
*/



#include "gpugas_kernels.cuh"
#include <vector>
#include <iterator>
#include "moderngpu.cuh"
#include "primitives/scatter_if_mgpu.h"
#include "util.h"

//using this because CUB device-wide reduce_by_key does not yet work
//and I am still working on a fused gatherMap/gatherReduce kernel.
#include "thrust/reduce.h"
#include "thrust/device_ptr.h"

//CUDA implementation of GAS API, version 2.

template<typename Program
  , typename Int = int32_t
  , bool sortEdgesForGather = true>
class GASEngineGPU
{
  //public to make nvcc happy
public:
  typedef typename Program::VertexData   VertexData;
  typedef typename Program::EdgeData     EdgeData;
  typedef typename Program::GatherResult GatherResult;

private:

  Int         m_nVertices;
  Int         m_nEdges;

  //input/output pointers to host data
  VertexData *m_vertexDataHost;
  EdgeData   *m_edgeDataHost;

  //GPU copy
  VertexData *m_vertexData;
  EdgeData   *m_edgeData;

  //CSC representation for gather phase
  //Kernel accessible data
  Int *m_srcs;
  Int *m_srcOffsets;
  Int *m_edgeIndexCSC;

  //CSR representation for reduce phase
  Int *m_dsts;
  Int *m_dstOffsets;
  Int *m_edgeIndexCSR;

  //Active vertex lists
  Int *m_active;
  Int  m_nActive;
  Int *m_activeNext;
  Int  m_nActiveNext;
  Int *m_applyRet; //set of vertices whose neighborhood will be active next
  char *m_activeFlags;

  //some temporaries that are needed for LBS
  Int *m_edgeCountScan;

  //mapped memory to avoid explicit copy of reduced value back to host memory
  Int *m_hostMappedValue;
  Int *m_deviceMappedValue;

  //counter and list for small sized scatter / activation
  Int *m_edgeOutputCounter;
  Int *m_outputEdgeList;

  //These go away once gatherMap/gatherReduce/apply are fused
  GatherResult *m_gatherMapTmp;  //store results of gatherMap()
  GatherResult *m_gatherTmp;     //store results of gatherReduce()
  Int          *m_gatherDstsTmp; //keys for reduce_by_key in gatherReduce

  //Preprocessed data for speeding up reduce_by_key when all vertices are active
  std::auto_ptr<mgpu::ReduceByKeyPreprocessData> preprocessData;
  bool preComputed;

  //MGPU context
  mgpu::ContextPtr m_mgpuContext;

  //convenience
  void errorCheck(cudaError_t err, const char* file, int line)
  {
    if( err != cudaSuccess )
    {
      printf("%s(%d): cuda error %d (%s)\n", file, line, err, cudaGetErrorString(err));
      abort();
    }
  }

  //use only for debugging kernels
  //this slows stuff down a LOT
  void syncAndErrorCheck(const char* file, int line)
  {
    cudaThreadSynchronize();
    errorCheck(cudaGetLastError(), file, line);
  }

  //this is undefined at the end of this template definition
  #define CHECK(X) errorCheck(X, __FILE__, __LINE__)
  #define SYNC_CHECK() syncAndErrorCheck(__FILE__, __LINE__)

  template<typename T>
  void gpuAlloc(T* &p, Int n)
  {
    CHECK( cudaMalloc(&p, sizeof(T) * n) );
  }

  template<typename T>
  void copyToGPU(T* dst, const T* src, Int n)
  {
    CHECK( cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice) );
  }

  template<typename T>
  void copyToHost(T* dst, const T* src, Int n)
  {
    //error check please!
    CHECK( cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost) );
  }

  void gpuFree(void *ptr)
  {
    if( ptr )
      CHECK( cudaFree(ptr) );
  }


  dim3 calcGridDim(Int n)
  {
    if (n < 65536)
      return dim3(n, 1, 1);
    else {
      int side1 = static_cast<int>(sqrt((double)n));
      int side2 = static_cast<int>(ceil((double)n / side1));
      return dim3(side2, side1, 1);
    }
  }


  Int divRoundUp(Int x, Int y)
  {
    return (x + y - 1) / y;
  }


  //for debugging
  template<typename T>
  void printGPUArray(T* ptr, int n)
  {
    std::vector<T> tmp(n);
    copyToHost(&tmp[0], ptr, n);
    for( Int i = 0; i < n; ++i )
      std::cout << i << " " << tmp[i] << std::endl;
  }

  public:
    GASEngineGPU()
      : m_nVertices(0)
      , m_nEdges(0)
      , m_vertexDataHost(0)
      , m_edgeDataHost(0)
      , m_vertexData(0)
      , m_edgeData(0)
      , m_srcs(0)
      , m_srcOffsets(0)
      , m_edgeIndexCSC(0)
      , m_dsts(0)
      , m_dstOffsets(0)
      , m_edgeIndexCSR(0)
      , m_active(0)
      , m_nActive(0)
      , m_activeNext(0)
      , m_nActiveNext(0)
      , m_applyRet(0)
      , m_activeFlags(0)
      , m_edgeCountScan(0)
      , m_gatherMapTmp(0)
      , m_gatherTmp(0)
      , m_gatherDstsTmp(0)
      , m_edgeOutputCounter(0)
      , m_outputEdgeList(0)
      , preComputed(false)
    {
      m_mgpuContext = mgpu::CreateCudaDevice(1);
    }


    ~GASEngineGPU()
    {
      gpuFree(m_vertexData);
      gpuFree(m_edgeData);
      gpuFree(m_srcs);
      gpuFree(m_srcOffsets);
      gpuFree(m_edgeIndexCSC);
      gpuFree(m_dsts);
      gpuFree(m_dstOffsets);
      gpuFree(m_edgeIndexCSR);
      gpuFree(m_active);
      gpuFree(m_activeNext);
      gpuFree(m_applyRet);
      gpuFree(m_activeFlags);
      gpuFree(m_edgeCountScan);
      gpuFree(m_gatherMapTmp);
      gpuFree(m_gatherTmp);
      gpuFree(m_gatherDstsTmp);
      gpuFree(m_edgeOutputCounter);
      gpuFree(m_outputEdgeList);
      cudaFreeHost(m_hostMappedValue);
      //don't we need to explicitly clean up m_mgpuContext?
    }


    void initMPI()
    {
      //do nothing right now
    }


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
      m_vertexDataHost = vertexData;
      m_edgeDataHost   = edgeData;

      //allocate copy of vertex and edge data on GPU
      if( m_vertexDataHost )
      {
        gpuAlloc(m_vertexData, m_nVertices);
        copyToGPU(m_vertexData, m_vertexDataHost, m_nVertices);
      }

      //allocate CSR and CSC edges
      gpuAlloc(m_srcOffsets, m_nVertices + 1);
      gpuAlloc(m_dstOffsets, m_nVertices + 1);
      gpuAlloc(m_srcs, m_nEdges);
      gpuAlloc(m_dsts, m_nEdges);

      //These edges are not needed when there is no edge data
      //We only need one of these since we can sort the edgeData directly into
      //either CSR or CSC.  But the memory and performance overhead of these
      //arrays needs to be discussed.
      if( sortEdgesForGather )
        gpuAlloc(m_edgeIndexCSR, m_nEdges);
      else
        gpuAlloc(m_edgeIndexCSC, m_nEdges);

      //these are pretty big temporaries, but we're assuming 'unlimited'
      //host memory for now.
      std::vector<Int> tmpOffsets(m_nVertices + 1);
      std::vector<Int> tmpVerts(m_nEdges);
      std::vector<Int> tmpEdgeIndex(m_nEdges);
      std::vector<EdgeData> sortedEdgeData(m_nEdges);

      //get CSC representation for gather/apply
      edgeListToCSC(m_nVertices, m_nEdges
        , edgeListSrcs, edgeListDsts
        , &tmpOffsets[0], &tmpVerts[0], &tmpEdgeIndex[0]);

      //sort edge data into CSC order to avoid an indirected read in gather
      if( sortEdgesForGather )
      {
        for(size_t i = 0; i < m_nEdges; ++i)
          sortedEdgeData[i] = m_edgeDataHost[ tmpEdgeIndex[i] ];
      }
      else
        copyToGPU(m_edgeIndexCSC, &tmpEdgeIndex[0], m_nEdges);

      copyToGPU(m_srcOffsets, &tmpOffsets[0], m_nVertices + 1);
      copyToGPU(m_srcs, &tmpVerts[0], m_nEdges);

      //get CSR representation for activate/scatter
      edgeListToCSR(m_nVertices, m_nEdges
        , edgeListSrcs, edgeListDsts
        , &tmpOffsets[0], &tmpVerts[0], &tmpEdgeIndex[0]);

      //sort edge data into CSR order to avoid an indirected write in scatter
      if( !sortEdgesForGather )
      {
        for(size_t i = 0; i < m_nEdges; ++i)
          sortedEdgeData[i] = m_edgeDataHost[ tmpEdgeIndex[i] ];
      }
      else
        copyToGPU(m_edgeIndexCSR, &tmpEdgeIndex[0], m_nEdges);

      copyToGPU(m_dstOffsets, &tmpOffsets[0], m_nVertices + 1);
      copyToGPU(m_dsts, &tmpVerts[0], m_nEdges);

      if( m_edgeDataHost )
      {
        gpuAlloc(m_edgeData, m_nEdges);
        copyToGPU(m_edgeData, &sortedEdgeData[0], m_nEdges);
      }

      //allocate active lists
      gpuAlloc(m_active, m_nVertices);
      gpuAlloc(m_activeNext, m_nVertices);

      //allocate temporaries for current multi-part gather kernels
      gpuAlloc(m_applyRet, m_nVertices);
      gpuAlloc(m_activeFlags, m_nVertices);
      gpuAlloc(m_edgeCountScan, m_nVertices);
      //have to allocate extra for faked incoming edges when there are
      //no incoming edges
      gpuAlloc(m_gatherMapTmp, m_nEdges + m_nVertices);
      gpuAlloc(m_gatherTmp, m_nVertices);
      gpuAlloc(m_gatherDstsTmp, m_nEdges + m_nVertices);

      //allocated mapped memory
      cudaMallocHost(&m_hostMappedValue, sizeof(Int), cudaHostAllocMapped );
      cudaHostGetDevicePointer(&m_deviceMappedValue, m_hostMappedValue, 0);

      //allocate for small sized list
      gpuAlloc(m_edgeOutputCounter, 1);
      if (m_nEdges < 50000) {
        gpuAlloc(m_outputEdgeList, m_nEdges);
      }
      else {
        gpuAlloc(m_outputEdgeList, m_nEdges / 100 + 1);
      }
    }


    //This may be a slow function, so normally would only be called
    //at the end of a computation.  This does not invalidate the
    //data already in the engine, but does make sure that the host
    //data is consistent with the engine's internal data
    void getResults()
    {
      if( m_vertexDataHost )
        copyToHost(m_vertexDataHost, m_vertexData, m_nVertices);
      if( m_edgeDataHost )
      {
        //unsort the edge data - todo
        copyToHost(m_edgeDataHost, m_edgeData, m_nEdges);
      }
    }


    //set the active flag for a range [vertexStart, vertexEnd)
    void setActive(Int vertexStart, Int vertexEnd)
    {
      m_nActive = vertexEnd - vertexStart;
      const int nThreadsPerBlock = 128;
      const int nBlocks = divRoundUp(m_nActive, nThreadsPerBlock);
      dim3 grid = calcGridDim(nBlocks);
      GPUGASKernels::kRange<<<grid, nThreadsPerBlock>>>(vertexStart, vertexEnd, m_active);
    }


    //Return the number of active vertices in the next gather step
    Int countActive()
    {
      return m_nActive;
    }


    //nvcc will not let this struct be private. Why?
    //MGPU-specific, equivalent to a thrust transform_iterator
    //customize scan to use edge counts from either offset differences or
    //a separately stored edge count array given a vertex list
    struct EdgeCountIterator : public std::iterator<std::input_iterator_tag, Int>
    {
      Int *m_offsets;
      Int *m_active;

      __host__ __device__
      EdgeCountIterator(Int *offsets, Int *active) : m_offsets(offsets), m_active(active) {};

      __device__
      Int operator[](Int i) const
      {
        Int active = m_active[i];
        return max(m_offsets[active + 1] - m_offsets[active], 1);
      }

      __device__
      EdgeCountIterator operator +(Int i) const
      {
        return EdgeCountIterator(m_offsets, m_active + i);
      }
    };

    //this one checks if predicate is false and outputs zero if so
    //used in the current impl for scatter, this will go away.
    struct PredicatedEdgeCountIterator : public std::iterator<std::input_iterator_tag, Int>
    {
      Int *m_offsets;
      Int *m_active;
      Int *m_predicates;

      __host__ __device__
      PredicatedEdgeCountIterator(Int *offsets, Int *active, Int * predicates) : m_offsets(offsets), m_active(active), m_predicates(predicates) {};

      __device__
      Int operator[](Int i) const
      {
        Int active = m_active[i];
        return m_predicates[i] ? m_offsets[active + 1] - m_offsets[active] : 0;
      }

      __device__
      PredicatedEdgeCountIterator operator +(Int i) const
      {
        return PredicatedEdgeCountIterator(m_offsets, m_active + i, m_predicates + i);
      }
    };

    //nvcc, why can't this struct by private?
    //wrap Program::gatherReduce for use with thrust
    struct ThrustReduceWrapper : std::binary_function<GatherResult, GatherResult, GatherResult>
    {
      __device__ GatherResult operator()(const GatherResult &left, const GatherResult &right)
      {
        return Program::gatherReduce(left, right);
      }
    };


    void gatherApply(bool haveGather=true)
    {
      if( haveGather )
      {
        //first scan the numbers of edges from the active list
        EdgeCountIterator ecIterator(m_srcOffsets, m_active);
        mgpu::Scan<mgpu::MgpuScanTypeExc, EdgeCountIterator, Int, mgpu::plus<Int>, Int*>(ecIterator
          , m_nActive
          , 0
          , mgpu::plus<int>()
          , m_deviceMappedValue
          , (Int *)NULL
          , m_edgeCountScan
          , *m_mgpuContext);
        cudaDeviceSynchronize();
        const int nThreadsPerBlock = 128;
        Int nActiveEdges = *m_hostMappedValue;

        MGPU_MEM(int) partitions = mgpu::MergePathPartitions<mgpu::MgpuBoundsUpper>
          (mgpu::counting_iterator<int>(0), nActiveEdges, m_edgeCountScan, m_nActive
          , nThreadsPerBlock, 0, mgpu::less<int>(), *m_mgpuContext);


        Int nBlocks = MGPU_DIV_UP(nActiveEdges + m_nActive, nThreadsPerBlock);


        dim3 grid = calcGridDim(nBlocks);
        GPUGASKernels::kGatherMap<Program, Int, nThreadsPerBlock, !sortEdgesForGather>
          <<<grid, nThreadsPerBlock>>>
          ( m_nActive
          , m_active
          , nBlocks
          , nActiveEdges
          , m_edgeCountScan
          , partitions->get()
          , m_srcOffsets
          , m_srcs
          , m_vertexData
          , m_edgeData
          , m_edgeIndexCSC
          , m_gatherDstsTmp
          , m_gatherMapTmp );
        SYNC_CHECK();


        if (m_nActive == m_nVertices && !preComputed) {
          mgpu::ReduceByKeyPreprocess<GatherResult>(nActiveEdges
                                                  , m_gatherDstsTmp
                                                  , (Int *)NULL
                                                  , mgpu::equal_to<Int>()
                                                  , NULL
                                                  , NULL
                                                  , &preprocessData
                                                  , *m_mgpuContext);

          preComputed = true;
        }


        if (m_nActive == m_nVertices) {
          mgpu::ReduceByKeyApply(*preprocessData
                               , m_gatherMapTmp
                               , Program::gatherZero
                               , ThrustReduceWrapper()
                               , m_gatherTmp
                               , *m_mgpuContext);
        }
        else {
          mgpu::ReduceByKey(m_gatherDstsTmp
                          , m_gatherMapTmp
                          , nActiveEdges
                          , Program::gatherZero
                          , ThrustReduceWrapper()
                          , mgpu::equal_to<Int>()
                          , (Int *)NULL
                          , m_gatherTmp
                          , NULL
                          , NULL
                          , *m_mgpuContext);
        }
        SYNC_CHECK();
      }

      //Now run the apply kernel
      {
        const int nThreadsPerBlock = 128;
        const int nBlocks = divRoundUp(m_nActive, nThreadsPerBlock);
        dim3 grid = calcGridDim(nBlocks);
        GPUGASKernels::kApply<Program, Int><<<grid, nThreadsPerBlock>>>
          (m_nActive, m_active, m_gatherTmp, m_vertexData, m_applyRet);
        SYNC_CHECK();
      }
    }


    //helper types for scatterActivate that should be private if nvcc would allow it
    //ActivateGatherIterator does an extra dereference: iter[x] = offsets[active[x]]
    struct ActivateGatherIterator : public std::iterator<std::input_iterator_tag, Int>
    {
      Int *m_offsets;
      Int *m_active;

      __host__ __device__
      ActivateGatherIterator(Int* offsets, Int* active)
        : m_offsets(offsets)
        , m_active(active)
      {};

      __device__
      Int operator [](Int i)
      {
        return m_offsets[ m_active[i] ];
      }

      __device__
      ActivateGatherIterator operator +(Int i) const
      {
        return ActivateGatherIterator(m_offsets, m_active + i);
      }
    };

    //"ActivateOutputIterator[i] = dst" effectively does m_flags[dst] = true
    //This does not work because DeviceScatter in moderngpu is written assuming
    //a T* rather than OutputIterator .
    struct ActivateOutputIterator
    {
      char* m_flags;

      __host__ __device__
      ActivateOutputIterator(char* flags) : m_flags(flags) {}

      __device__
      ActivateOutputIterator& operator[](Int i)
      {
        return *this;
      }

      __device__
      void operator =(Int dst)
      {
        m_flags[dst] = true;
      }

      __device__
      ActivateOutputIterator operator +(Int i)
      {
        return ActivateOutputIterator(m_flags);
      }
    };

    struct ActivateOutputIteratorSmallSize
    {
      int* m_count;
      Int* m_list;

      __host__ __device__
      ActivateOutputIteratorSmallSize(int* count, Int *list) : m_count(count), m_list(list) {}

      __device__
      ActivateOutputIteratorSmallSize& operator[](Int i)
      {
        return *this;
      }

      __device__
      void operator =(Int dst)
      {
        int pos = atomicAdd(m_count, 1);
        m_list[pos] = dst;
      }

      __device__
      ActivateOutputIteratorSmallSize operator +(Int i)
      {
        return ActivateOutputIteratorSmallSize(m_count, m_list);
      }
    };

    struct ListToHeadFlagsIterator : public std::iterator<std::input_iterator_tag, Int>
    {
      int *m_list;
      int m_offset;

      __host__ __device__
      ListToHeadFlagsIterator(int *list) : m_list(list), m_offset(0) {}

      __host__ __device__
      ListToHeadFlagsIterator(int *list, int offset) : m_list(list), m_offset(offset) {}

      __device__
      int operator[](int i) {
        if (m_offset == 0 && i == 0)
          return 1;
        else {
          return m_list[m_offset + i] != m_list[m_offset + i - 1];
        }
      }

      __device__
      ListToHeadFlagsIterator operator+(int i) const
      {
        return ListToHeadFlagsIterator(m_list, m_offset + i);
      }
    };

    struct ListOutputIterator : public std::iterator<std::output_iterator_tag, Int>
    {
      int* m_inputlist;
      int* m_outputlist;
      int m_offset;

      __host__ __device__
      ListOutputIterator(int *inputlist, int *outputlist) : m_inputlist(inputlist), m_outputlist(outputlist), m_offset(0) {}

      __host__ __device__
      ListOutputIterator(int *inputlist, int *outputlist, int offset) : m_inputlist(inputlist), m_outputlist(outputlist), m_offset(offset) {}

      __host__ __device__
      ListOutputIterator operator[](Int i) const
      {
        return ListOutputIterator(m_inputlist, m_outputlist, m_offset + i);
      }

      __device__
      void operator =(Int dst)
      {
        if (m_offset == 0) {
          m_outputlist[dst] = m_inputlist[0];
        }
        else {
          if (m_inputlist[m_offset] != m_inputlist[m_offset - 1]) {
            m_outputlist[dst] = m_inputlist[m_offset];
          }
        }
      }

      __device__
      ListOutputIterator operator +(Int i) const
      {
        return ListOutputIterator(m_inputlist, m_outputlist, m_offset + i);
      }
    };

    //not writing a custom kernel for this until we get kGather right because
    //it actually shares a lot with this kernel.
    //this version only does activate, does not actually invoke Program::scatter,
    //this will let us improve the gather kernel and then factor it into something
    //we can use for both gather and scatter.
    void scatterActivate(bool haveScatter=true)
    {
      //counts = m_applyRet ? outEdgeCount[ m_active ] : 0
      //first scan the numbers of edges from the active list
      PredicatedEdgeCountIterator ecIterator(m_dstOffsets, m_active, m_applyRet);
      mgpu::Scan<mgpu::MgpuScanTypeExc, PredicatedEdgeCountIterator, Int, mgpu::plus<Int>, Int*>(ecIterator
        , m_nActive
        , 0
        , mgpu::plus<Int>()
        , m_deviceMappedValue
        , (Int *)NULL
        , m_edgeCountScan
        , *m_mgpuContext);
      cudaDeviceSynchronize();
      Int nActiveEdges = *m_hostMappedValue;
      SYNC_CHECK();

      if (nActiveEdges == 0) {
        m_nActive = 0;
      }
      //100 is an empirically chosen value that seems to give good performance
      else if (nActiveEdges > m_nEdges / 100) {
        //Gathers the dst vertex ids from m_dsts and writes a true for each
        //dst vertex into m_activeFlags
        CHECK( cudaMemset(m_activeFlags, 0, sizeof(char) * m_nVertices) );

        IntervalGather(nActiveEdges
          , ActivateGatherIterator(m_dstOffsets, m_active)
          , m_edgeCountScan
          , m_nActive
          , m_dsts
          , ActivateOutputIterator(m_activeFlags)
          , *m_mgpuContext);
        SYNC_CHECK();

        //convert m_activeFlags to new active compact list in m_active
        //set m_nActive to the number of active vertices
        scatter_if_inputloc_twophase(m_nVertices,
                                     m_activeFlags,
                                     m_active,
                                     m_deviceMappedValue,
                                     m_mgpuContext);
        SYNC_CHECK();
        m_nActive = *m_hostMappedValue;
      }
      else {
        //we have a small number of edges, so just output into a list
        //with atomics, sort and then extract unique values
        CHECK( cudaMemset(m_edgeOutputCounter, 0, sizeof(int) ) );

        IntervalGather(nActiveEdges
          , ActivateGatherIterator(m_dstOffsets, m_active)
          , m_edgeCountScan
          , m_nActive
          , m_dsts
          , ActivateOutputIteratorSmallSize(m_edgeOutputCounter, m_outputEdgeList)
          , *m_mgpuContext);
        SYNC_CHECK();

        mgpu::MergesortKeys(m_outputEdgeList, nActiveEdges, mgpu::less<Int>(), *m_mgpuContext);
        SYNC_CHECK();

        mgpu::Scan<mgpu::MgpuScanTypeExc, ListToHeadFlagsIterator, Int, mgpu::plus<Int>, ListOutputIterator>(
            ListToHeadFlagsIterator(m_outputEdgeList)
          , nActiveEdges
          , 0
          , mgpu::plus<Int>()
          , m_deviceMappedValue
          , (Int *)NULL
          , ListOutputIterator(m_outputEdgeList, m_active)
          , *m_mgpuContext);


        cudaDeviceSynchronize();
        m_nActive = *m_hostMappedValue;
        SYNC_CHECK();
      }
    }


    Int nextIter()
    {
      //nothing to be done here.
      return m_nActive;
    }


    //single entry point for the whole affair, like before.
    //special cases that don't need gather or scatter can
    //easily roll their own loop.
    void run()
    {
      while( countActive() )
      {
        gatherApply();
        scatterActivate();
        nextIter();
      }
    }

    //remove macro clutter
    #undef CHECK
    #undef SYNC_CHECK
};



#endif
