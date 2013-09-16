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
__global__ void kRange(Int start, Int end, Int incr, Int *out)
{
  Int tid = globalThreadId<Int>();
  if( tid < end - start )
    out[tid] = (tid + start) * incr;
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
  , Int *dst
  , typename Program::GatherResult* output)
{
  //boilerplate from MGPU, VT will be 1 in this kernel until
  //a full rewrite of this kernel, so not bothering with LaunchBox
  const int VT = 1;
  
  union Shared
  {
    Int indices[NT * (VT + 1)];    
  };
  __shared__ Shared shared; //so poetic!

  Int block = blockIdx.x + blockIdx.y * gridDim.y;

  int4 range = mgpu::CTALoadBalance<NT, VT>(nTotalEdges, edgeCountScan
    , nActiveVertices, block, threadIdx.x, mergePathPartitions
    , shared.indices, true);

  //should use a variant of DeviceMemToMemLoop here for getting dstVid
  //iSrcEdge and vertexData[dst]
  const int bTid = threadIdx.x;
  const Int tid  = globalThreadId<Int>();
  Int iActive    = shared.indices[bTid];
  Int dstVid     = activeVertices[iActive];
  Int iSrcEdge   = tid - edgeCountScan[iActive];
  Int srcVid     = srcs[ srcOffsets[dstVid] + iSrcEdge ];

  //this mapping is not quite right - since there are nActiveEdges + nActiveVertices
  //threads.
  output[tid] = Program::gatherMap(vertexData + dstVid, vertexData + srcVid
    , 0); //need indirect address to edge data here
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
  if( tid > nActiveVertices )
    return;
  int vid = activeVertices[tid];
  retFlags[tid] = Program::apply(vertexData ? vertexData + vid : 0, gatherResults[tid]);  
}



} //end namespace GPUGASKernels

#endif
