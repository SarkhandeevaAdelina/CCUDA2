#ifndef CUDA_DIFRONLENTA_H
#define CUDA_DIFRONLENTA_H

#include "Matrix.h"
#include "Bessel.h"
#include "Chebyshev.h"
#include <vector>

struct TimingInfo {
    double matrixBuildTime;
    double solveTime;
    double totalTime;
};

class DifrOnLentaCuda {
public:
    double a;
    double b;
    double lambda;
    double theta;
    int N;
    double skinDepth;
    Complex chi;
    std::vector<Complex> y;
    TimingInfo lastTiming;

    DifrOnLentaCuda(double _a, double _b, double _lambda, double _theta, int _N, double _skinDepth = 0.0);

    int SolveDifr();

private:
    CUDA_HOST_DEVICE double ChebAB(int n, double x) const;
    CUDA_HOST_DEVICE Complex r(double t, double x) const;
    CUDA_HOST_DEVICE Complex u0(double x, double z) const;
    CUDA_HOST_DEVICE Complex f(double x) const;
    Complex CalculateChi() const;
};

#endif // CUDA_DIFRONLENTA_H


