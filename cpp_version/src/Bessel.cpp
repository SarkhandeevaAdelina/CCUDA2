#include "Bessel.h"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Комплексная единица i
const Complex ci(0.0, 1.0);

// Функция Бесселя первого рода нулевого порядка J0(x)
double J0(double x) {
    const double eps = 1e-4;
    const int maxIter = 10000;
    
    double sum = 0.0;
    double s = -1.0;
    double k2 = 1.0;
    double xS = 1.0;
    double x2 = x * x / 4.0;
    double _1 = -1.0;
    long k = 0;
    
    while (std::abs(s) > eps && k < maxIter) {
        sum += s;
        k++;
        _1 = -_1;
        k2 /= (k * k);
        xS *= x2;
        s = _1 * k2 * xS;
    }
    
    if (k >= maxIter) {
        std::cerr << "Warning: J0(" << x << ") did not converge in " 
                  << maxIter << " iterations" << std::endl;
    }
    
    return -sum;
}

// Вспомогательная функция для вычисления N0
double _Y0(double x) {
    const double eps = 1e-4;
    const int maxIter = 10000;
    
    double sum = 0.0;
    double s = -1.0;
    double k2 = 1.0;
    double xS = 1.0;
    double x2 = x * x / 4.0;
    double _1 = -1.0;
    double psi = -0.57721566 + 1.0; // -γ (константа Эйлера) + 1
    long k = 0;
    
    while (std::abs(s) > eps && k < maxIter) {
        sum += s;
        k++;
        _1 = -_1;
        k2 /= (k * k);
        xS *= x2;
        psi += 1.0 / k;
        s = _1 * k2 * xS * psi;
    }
    
    if (k >= maxIter) {
        std::cerr << "Warning: _Y0(" << x << ") did not converge in " 
                  << maxIter << " iterations" << std::endl;
    }
    
    return sum;
}

// Функция Бесселя второго рода нулевого порядка N0(x) = Y0(x)
double N0(double x) {
    return 2.0 / M_PI * (J0(x) * std::log(x / 2.0) + _Y0(x));
}

// Функция Ханкеля первого рода нулевого порядка
Complex H0_1(double x) {
    return Complex(J0(x), N0(x));
}

// Функция Ханкеля второго рода нулевого порядка
Complex H0_2(double x) {
    return Complex(J0(x), -N0(x));
}

