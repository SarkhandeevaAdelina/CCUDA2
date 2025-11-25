#include "Chebyshev.h"

CUDA_HOST_DEVICE double Cheb(int n, double x) {
    double T = 0.0;
    double T0 = 1.0;
    double T1 = x;

    if (n == 0) {
        T = 1.0;
    } else if (n == 1) {
        T = x;
    } else {
        for (int i = 2; i <= n; i++) {
            T = 2.0 * x * T1 - T0;
            T0 = T1;
            T1 = T;
        }
    }

    return T;
}


