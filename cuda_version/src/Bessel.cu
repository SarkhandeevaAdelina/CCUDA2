#include "Bessel.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CUDA_HOST_DEVICE double J0(double x) {
    const double eps = 1e-6;
    const int maxIter = 10000;

    double sum = 0.0;
    double s = -1.0;
    double k2 = 1.0;
    double xS = 1.0;
    double x2 = x * x / 4.0;
    double _1 = -1.0;
    long k = 0;

    while (fabs(s) > eps && k < maxIter) {
        sum += s;
        k++;
        _1 = -_1;
        k2 /= (k * k);
        xS *= x2;
        s = _1 * k2 * xS;
    }

    return -sum;
}

CUDA_HOST_DEVICE double _Y0(double x) {
    const double eps = 1e-6;
    const int maxIter = 10000;

    double sum = 0.0;
    double s = -1.0;
    double k2 = 1.0;
    double xS = 1.0;
    double x2 = x * x / 4.0;
    double _1 = -1.0;
    double psi = -0.57721566 + 1.0;
    long k = 0;

    while (fabs(s) > eps && k < maxIter) {
        sum += s;
        k++;
        _1 = -_1;
        k2 /= (k * k);
        xS *= x2;
        psi += 1.0 / k;
        s = _1 * k2 * xS * psi;
    }

    return sum;
}

CUDA_HOST_DEVICE double N0(double x) {
    return 2.0 / M_PI * (J0(x) * log(x / 2.0) + _Y0(x));
}


