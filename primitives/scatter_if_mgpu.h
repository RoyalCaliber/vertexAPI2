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

#ifndef SCATTER_IF_H__
#define SCATTER_IF_H__

#include "moderngpu.cuh"

template<typename InputIt, typename PredicateIt, typename OutputIt>
__global__
void scatterKernel(InputIt in,
                   int N,
                   PredicateIt pred,
                   int *d_map,
                   OutputIt output)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    if (pred[i]) {
      output[d_map[i]] = in[i];
    }
  }
}

template<typename PredType>
struct scatterIterator {
  PredType m_pred;
  int*  m_dst;
  int   m_index;

  __host__ __device__
  scatterIterator(PredType pred, int *dst) : m_pred(pred), m_dst(dst), m_index(0) {}

  __host__ __device__
  scatterIterator(PredType pred, int *dst, int index) : m_pred(pred), m_dst(dst), m_index(index) {}

  __host__ __device__
  scatterIterator operator[](int i) const
  {
    return scatterIterator(m_pred, m_dst, m_index + i);
  }

  __host__ __device__
  void operator =(int dst)
  {
    if (m_pred[m_index]) {
      m_dst[dst] = m_index;
    }
  }

  __host__ __device__
  scatterIterator operator +(int i) const
  {
    return scatterIterator(m_pred, m_dst, m_index + i);
  }
};

//one pass version using scatter iterator
//unfortunately, 64-bit linux nvcc generates crappy
//code for now.  Will be almost 50% faster once the bug is fixed.
//Currently much slower when compiling for 64 bit.
template<typename PredIt, typename OutputIt>
int scatter_if_inputloc_onephase(int num,
                             PredIt pred_begin,
                             OutputIt    output_begin,
                             mgpu::ContextPtr mgpuContext) {

  int total;
  mgpu::Scan<mgpu::MgpuScanTypeExc>(pred_begin, num, output_begin, mgpu::ScanOpAdd(),
                                    &total, true, *mgpuContext);
  return total;
}

//two pass version which requires memory allocation
//for intermediate scan result
//(allocation should be mostly hidden by mgpu caching)
//and writing scan result out to memory
//and then reading it back in
template<typename InputIt, typename PredIt, typename OutputIt>
int scatter_if_general_twophase(InputIt     input_begin,
                             int num,
                             PredIt pred_begin,
                             OutputIt    output_begin,
                             mgpu::ContextPtr mgpuContext) {

  int total;

  MGPU_MEM(int) d_map = mgpuContext->Malloc<int>(num);

  mgpu::Scan<mgpu::MgpuScanTypeExc>(pred_begin, num, d_map->get(), mgpu::ScanOpAdd(),
                                    &total, true, *mgpuContext);

  const int numThreads = 192;
  const int numBlocks = min((num + numThreads - 1) / numThreads, 256);

  scatterKernel<<<numBlocks, numThreads>>>(input_begin, num,
                                           pred_begin,
                                           d_map->get(),
                                           output_begin);

  return total;
}

template<typename PredIt, typename OutputIt>
int scatter_if_inputloc_twophase(int num,
                                 PredIt pred_begin,
                                 OutputIt    output_begin,
                                 mgpu::ContextPtr mgpuContext) {

  int total;

  MGPU_MEM(int) d_map = mgpuContext->Malloc<int>(num);

  mgpu::Scan<mgpu::MgpuScanTypeExc>(pred_begin, num, d_map->get(), mgpu::ScanOpAdd(),
                                    &total, false, *mgpuContext);

  const int numThreads = 192;
  const int numBlocks = min((num + numThreads - 1) / numThreads, 256);

  mgpu::counting_iterator<int> input_begin(0);

  scatterKernel<<<numBlocks, numThreads>>>(input_begin, num,
                                           pred_begin,
                                           d_map->get(),
                                           output_begin);

  return total;
}

#endif
