#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#ifdef __CUDACC__
#include <thrust/complex.h>
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
using Complex = thrust::complex<double>;
CUDA_HOST_DEVICE inline double ComplexAbs(const Complex& value) {
    return thrust::abs(value);
}
#else
#include <complex>
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_GLOBAL
using Complex = std::complex<double>;
inline double ComplexAbs(const Complex& value) {
    return std::abs(value);
}
#endif

#endif // CUDA_COMMON_H


