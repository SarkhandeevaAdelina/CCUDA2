#include "DifrOnLenta.h"
#include "Gauss.h"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DifrOnLenta::DifrOnLenta(double _a, double _b, double _lambda, double _teta, int _N, double _skinDepth)
    : a(_a), b(_b), lambda(_lambda), teta(_teta), N(_N), skinDepth(_skinDepth) {
    
    y.resize(N, Complex(0.0, 0.0));
    chi = CalculateChi();
    
    lastTiming.matrixBuildTime = 0.0;
    lastTiming.solveTime = 0.0;
    lastTiming.totalTime = 0.0;
}

// Расчет импедансного коэффициента χ для скин-эффекта
Complex DifrOnLenta::CalculateChi() {
    if (skinDepth <= 0.0) {
        // Для идеального проводника χ = 0
        return Complex(0.0, 0.0);
    }
    
    // Волновое число k = 2π/λ
    double k = 2.0 * M_PI / lambda;
    
    // Нормированный импедансный параметр
    // χ = k * δ * (1 + i) - классическая формула для скин-эффекта
    double chiReal = k * skinDepth;
    double chiImag = k * skinDepth;
    
    return Complex(chiReal, chiImag);
}

// Полином Чебышева на отрезке [a, b]
double DifrOnLenta::ChebAB(int n, double x) {
    return Cheb(n, 2.0 / (b - a) * x - (b + a) / (b - a));
}

// Ядро интегрального уравнения
Complex DifrOnLenta::r(double t, double x) {
    double k = 2.0 * M_PI / lambda;
    Complex s(J0(k * std::abs(t - x)), 0.0);
    
    if (std::abs(t - x) > 1e-8) {
        s = M_PI * ci / 2.0 * s + (s - 1.0) * std::log(k * std::abs(t - x) / 2.0);
    } else {
        s = M_PI * ci / 2.0 * s;
    }
    
    s += std::log(k / 2.0) + _Y0(k * std::abs(t - x));
    return -s;
}

// Падающее поле
Complex DifrOnLenta::u0(double x, double z) {
    double k = 2.0 * M_PI / lambda;
    return std::exp(k * std::cos(teta) * ci * x + k * std::sin(teta) * ci * z);
}

// Функция правой части
Complex DifrOnLenta::f(double x) {
    return -2.0 * M_PI * u0(x, 0.0);
}

// Решение задачи дифракции с замером времени
int DifrOnLenta::SolveDifr() {
    using namespace std::chrono;
    
    auto totalStart = high_resolution_clock::now();
    
    const int M = 20;
    CVect B(N);
    CMatr A(N);
    
    // Вычисление точек коллокации (узлы квадратуры Гаусса-Чебышева)
    std::vector<double> x_points(M);
    for (int m = 0; m < M; m++) {
        x_points[m] = (b - a) / 2.0 * std::cos((2 * m + 1) / 2.0 / M * M_PI) + (b + a) / 2.0;
    }
    
    // ========== ПОСТРОЕНИЕ МАТРИЦЫ ==========
    auto matrixStart = high_resolution_clock::now();
    
    for (int k = 0; k < N; k++) {
        // Построение строки матрицы
        for (int j = 0; j < N; j++) {
            Complex s(0.0, 0.0);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < M; n++) {
                    s += r(x_points[n], x_points[m]) * ChebAB(j, x_points[n]) * ChebAB(k, x_points[m]);
                }
            }
            s *= M_PI * M_PI / (M * M);
            A[k][j] = s;
        }
        
        // Построение вектора правой части
        Complex s(0.0, 0.0);
        for (int m = 0; m < M; m++) {
            s += f(x_points[m]) * ChebAB(k, x_points[m]);
        }
        B[k] = s * M_PI / static_cast<double>(M);
        
        // Диагональные элементы с учетом регуляризации
        Complex regularization;
        if (k == 0) {
            regularization = Complex(M_PI * M_PI * std::log(b - a), 0.0);
        } else {
            regularization = Complex(M_PI * M_PI / 2.0 / (k + 1), 0.0);
        }
        
        A[k][k] = A[k][k] + regularization;
        
        // Граничное условие Леонтовича для импедансной поверхности
        if (skinDepth > 0.0) {
            Complex impedanceCorrection = ci * chi * A[k][k];
            A[k][k] = A[k][k] + impedanceCorrection;
        }
    }
    
    auto matrixEnd = high_resolution_clock::now();
    lastTiming.matrixBuildTime = duration_cast<microseconds>(matrixEnd - matrixStart).count() / 1000.0;
    
    // ========== РЕШЕНИЕ СЛАУ ==========
    auto solveStart = high_resolution_clock::now();
    
    CVect w(N);
    int output = Gauss(A, B, w);
    
    auto solveEnd = high_resolution_clock::now();
    lastTiming.solveTime = duration_cast<microseconds>(solveEnd - solveStart).count() / 1000.0;
    
    auto totalEnd = high_resolution_clock::now();
    lastTiming.totalTime = duration_cast<microseconds>(totalEnd - totalStart).count() / 1000.0;
    
    // Копирование результата
    for (int k = 0; k < N; k++) {
        y[k] = w[k];
    }
    
    return output;
}

