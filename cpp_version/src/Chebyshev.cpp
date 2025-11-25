#include "Chebyshev.h"

// Вычисление полинома Чебышева n-го порядка в точке x
// T_0(x) = 1
// T_1(x) = x
// T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
double Cheb(int n, double x) {
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





