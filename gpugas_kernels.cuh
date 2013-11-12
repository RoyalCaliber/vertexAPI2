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

#ifndef GPUGAS_KERNELS_CUH__
#define GPUGAS_KERNELS_CUH__

//from moderngpu library - no 'mgpu' prefix on the header :(
#include "device/ctaloadbalance.cuh"
#include "cub/warp/warp_reduce.cuh"
#include "cub/block/block_scan.cuh"

//Device code for GASEngineGPU


//Ideally would have wanted these kernels to be static private member functions
//but nvcc (Cuda 5.0) does not like __global__ or __device__ code inside classes.
namespace GPUGASKernels
{


//assumes blocks are 1-D
//allows for 2D grids since 1D grids don't get us very far.
template<typename Int>
__device__ Int globalThreadId()
{
  return threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
}


//set an array to a range of values
template<typename Int>
__global__ void kRange(Int start, Int end, Int *out)
{
  Int tid = globalThreadId<Int>();
  if( tid < end - start )
    out[tid] = tid + start;
}


//Wrap up Program::gatherReduce for use with cub::Reduce
template<typename Program>
class CubReduceWrapper
{
  typedef typename Program::GatherResult GatherResult;
  public:
    __forceinline__ __device__
    GatherResult operator ()(const GatherResult &left, const GatherResult &right)
    {
      return Program::gatherReduce(left, right);
    }
};


//This version does the gatherMap only and generates keys for subsequent
//use iwth thrust reduce_by_key
template<typename Program, typename Int, int NT, int indirectedGather>
__global__ void kGatherMap(Int nActiveVertices
  , const Int *activeVertices
  , const int numBlocks
  , Int nTotalEdges
  , const Int *edgeCountScan
  , const Int *mergePathPartitions
  , const Int *srcOffsets
  , const Int *srcs
  , const typename Program::VertexData* vertexData
  , const typename Program::EdgeData*   edgeData
  , const Int* edgeIndexCSC
  , Int *globalOutputIndex
  , Int *dsts
  , typename Program::GatherResult* output)
{
  typedef typename Program::GatherResult GatherResult;
  
  //boilerplate from MGPU, VT will be 1 in this kernel until
  //a full rewrite of this kernel, so not bothering with LaunchBox
  const int VT = 1;

  //For reducing GatherResults across a warp
  typedef cub::WarpReduce<float, NT/32, 32> WarpReduce;

  //For compacting results prior to writing out
  typedef cub::BlockScan<int, NT> BlockScan;

  union Shared
  {
    Int indices[NT * (VT + 1)];
    Int dstVerts[NT * VT];
    typename BlockScan::TempStorage bsTmp;
    typename WarpReduce::TempStorage wrTmp;
    struct {
      GatherResult stagedResults[NT * VT];
      Int stagedKeys[NT * VT];
    };
  };
  __shared__ Shared shared; //so poetic!

  Int block = blockIdx.x + blockIdx.y * gridDim.x;
  Int bTid  = threadIdx.x; //tid within block

  if (block >= numBlocks)
    return;

  int4 range = mgpu::CTALoadBalance<NT, VT>(nTotalEdges, edgeCountScan
    , nActiveVertices, block, bTid, mergePathPartitions
    , shared.indices, true);

  //global index into output
  Int gTid = bTid + range.x;

  //get the count of edges this block will do
  int edgeCount = range.y - range.x;

  //get the number of dst vertices this block will do
  //int nDsts = range.w - range.z;

  int iActive[VT];
  mgpu::DeviceSharedToReg<NT, VT>(NT * VT, shared.indices, bTid, iActive);

  //each thread that is responsible for an edge should now apply Program::gatherMap
  GatherResult result = Program::gatherZero;
  Int dstVerts[VT];
  int rank = -1; //used to keep track of first edge for a given dst
  
  if( bTid < edgeCount )
  {
    rank = gTid - shared.indices[edgeCount + iActive[0] - range.z];
    
    //should we use an mgpu function for this indirected load?
    Int dst = dstVerts[0] = activeVertices[iActive[0]];
    //check if we have a vertex with no incoming edges
    //this is the matching kludge for faking the count to be 1
    Int soff = srcOffsets[dst];
    Int nEdges = srcOffsets[dst + 1] - soff;
    if( nEdges )
    {
      int iEdge = soff + rank;
      Int src = srcs[ iEdge ];
      if( indirectedGather )
        result = Program::gatherMap(vertexData + dst, vertexData + src, edgeData + edgeIndexCSC[iEdge] );
      else
        result = Program::gatherMap(vertexData + dst, vertexData + src, edgeData + iEdge );
    }

//    //write out a key and a result.
//    dsts[gTid] = dstVerts[0];
//    output[gTid] = result;      

  }

  //start of the sequence at beginning of a warp and when rank == 0
  const int headFlag = (rank == 0) || ( (bTid % 32) == 0 && rank != -1);

  //These can be removed if we don't reuse the same shared memory.
    __syncthreads();
  
  //Do a warp-reduce-by-key using cub
  result = WarpReduce(shared.wrTmp).template HeadSegmentedReduce
    <CubReduceWrapper<Program>, int>(result, headFlag, CubReduceWrapper<Program>());
    
  //this sync not needed if we can have separate smem temp areas for
  //WarpReduce and BlockScan
  __syncthreads();
  
//  //Now do a block-wide compaction of results
  int outIdx;
  BlockScan(shared.bsTmp).ExclusiveSum((int)headFlag, outIdx);

  __syncthreads();

  //let all threads know how many values we have
  //and also get the gmem output location using atomicAdd
  __shared__ int nOutputs;
  __shared__ Int gOutIdx;
  if( bTid == NT - 1 )
  {
    nOutputs = outIdx + headFlag;
    gOutIdx  = atomicAdd(globalOutputIndex, (Int) nOutputs);
  }

  __syncthreads();

  //stage values in smem
  if( headFlag )
  {
    shared.stagedResults[outIdx] = result;
    shared.stagedKeys[outIdx]    = dstVerts[0];
  }
  
  __syncthreads();

  //Now write out staged values
  if( bTid < nOutputs )
  {
    output[gOutIdx + bTid] = shared.stagedResults[bTid];
    dsts[gOutIdx + bTid]   = shared.stagedKeys[bTid];
  }
}


//Run Program::apply and store which vertices are to activate their neighbors
//we can either return a compact list or a set of flags.  To be decided
template<typename Program, typename Int>
__global__ void kApply(Int nActiveVertices
  , const Int *activeVertices
  , const typename Program::GatherResult *gatherResults
  , typename Program::VertexData *vertexData
  , Int *retFlags)
{
  Int tid = globalThreadId<Int>();
  if( tid >= nActiveVertices )
    return;
  int vid = activeVertices[tid];
  retFlags[tid] = Program::apply(vertexData ? vertexData + vid : 0, gatherResults[tid]);
}



} //end namespace GPUGASKernels

#endif
