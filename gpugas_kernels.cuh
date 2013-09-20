#ifndef GPUGAS_KERNELS_CUH__
#define GPUGAS_KERNELS_CUH__

//from moderngpu library
#include "device/ctaloadbalance.cuh"

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
  return threadIdx.x + blockDim.x
    * (blockIdx.x + blockIdx.y * gridDim.y + blockIdx.z * gridDim.y * gridDim.z);
}


//set an array to a range of values
template<typename Int>
__global__ void kRange(Int start, Int end, Int *out)
{
  Int tid = globalThreadId<Int>();
  if( tid < end - start )
    out[tid] = tid + start;
}


//This version does the gatherMap only and generates keys for subsequent
//use iwth thrust reduce_by_key
template<typename Program, typename Int, int NT>
__global__ void kGatherMap(Int nActiveVertices
  , const Int *activeVertices
  , Int nTotalEdges
  , const Int *edgeCountScan
  , const Int *mergePathPartitions
  , const Int *srcOffsets
  , const Int *srcs
  , const typename Program::VertexData* vertexData
  , const typename Program::EdgeData*   edgeData
  , Int *dsts
  , typename Program::GatherResult* output)
{
  //boilerplate from MGPU, VT will be 1 in this kernel until
  //a full rewrite of this kernel, so not bothering with LaunchBox
  const int VT = 1;
  
  union Shared
  {
    Int indices[NT * (VT + 1)];
    Int dstVerts[NT * VT];
  };
  __shared__ Shared shared; //so poetic!

  Int block = blockIdx.x + blockIdx.y * gridDim.y;
  Int bTid  = threadIdx.x; //tid within block

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

  //get the incoming edge index for this dstVertex
  int iEdge;
  if( bTid < edgeCount )
    iEdge = gTid - shared.indices[edgeCount + iActive[0]];
  
  //each thread that is responsible for an edge should now apply Program::gatherMap
  typename Program::GatherResult result;
  Int dstVerts[VT];
  if( bTid < edgeCount )
  {
    //should we use an mgpu function for this indirected load?
    Int dst = dstVerts[0] = activeVertices[iActive[0]];
    iEdge  += srcOffsets[dst];
    Int src = srcs[ iEdge ];
    result = Program::gatherMap(vertexData + dst, vertexData + src, edgeData + iEdge );
  }

  //write out a key and a result.
  //Next we will be adding a blockwide or atleast a warpwide reduction here.
  if( bTid < edgeCount )
  {
    dsts[gTid] = dstVerts[0];
    output[gTid] = result;
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
