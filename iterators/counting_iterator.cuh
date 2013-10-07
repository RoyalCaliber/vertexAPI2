#ifndef COUNTING_ITERATOR_H__
#define COUNTING_ITERATOR_H__

#include <iterator>

template<typename T>
struct countingIterator : public std::iterator<std::random_access_iterator_tag, T> {
  T m_base;

  __host__ __device__
  countingIterator(T base) : m_base(base) {}

  __host__ __device__
  T operator [](int i) {
    return m_base + i;
  }

  __host__ __device__
  countingIterator operator +(int i ) const {
    return countingIterator(m_base + i);
  }
};

#endif
