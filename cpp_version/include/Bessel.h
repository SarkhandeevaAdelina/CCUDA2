#ifndef BESSEL_H
#define BESSEL_H

#include <complex>

using Complex = std::complex<double>;

// Комплексная единица i
extern const Complex ci;

// Функция Бесселя первого рода нулевого порядка J0(x)
double J0(double x);

// Вспомогательная функция для вычисления N0
double _Y0(double x);

// Функция Бесселя второго рода нулевого порядка N0(x) = Y0(x)
double N0(double x);

// Функция Ханкеля первого рода нулевого порядка H0^(1)(x) = J0(x) + i*N0(x)
Complex H0_1(double x);

// Функция Ханкеля второго рода нулевого порядка H0^(2)(x) = J0(x) - i*N0(x)
Complex H0_2(double x);

#endif // BESSEL_H





