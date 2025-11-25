#ifndef CUDA_BESSEL_H
#define CUDA_BESSEL_H

#include "CudaCommon.h"

CUDA_HOST_DEVICE double J0(double x);
CUDA_HOST_DEVICE double _Y0(double x);
CUDA_HOST_DEVICE double N0(double x);
CUDA_HOST_DEVICE inline Complex H0_1(double x) {
    return Complex(J0(x), N0(x));
}
CUDA_HOST_DEVICE inline Complex H0_2(double x) {
    return Complex(J0(x), -N0(x));
}

#endif // CUDA_BESSEL_H


