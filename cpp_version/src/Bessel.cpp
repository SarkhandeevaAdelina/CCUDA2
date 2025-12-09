#include "Bessel.h"  
#include <cmath>     
#include <iostream> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Комплексная единица i (0 + 1i)
const Complex ci(0.0, 1.0);

// Функция Бесселя первого рода нулевого порядка J0(x)
// Ряд: J0(x) = Σ_{k=0}^∞ (-1)^k / (k!)^2 * (x/2)^{2k}
double J0(double x) {
    const double eps = 1e-4;        
    const int maxIter = 10000;      
    
    double sum = 0.0;               // Накопленная сумма ряда
    double s = -1.0;                // Текущий член ряда (для k=0: (-1)^0 * (x/2)^0 / (0!)^2 = 1)
    double k2 = 1.0;                // 1/(k!)^2 - обратный квадрат факториала
    double xS = 1.0;                // (x/2)^{2k} - степень аргумента
    double x2 = x * x / 4.0;        // (x/2)^2 - квадрат половинного аргумента
    double _1 = -1.0;               // (-1)^k - знакочередующийся множитель
    long k = 0;                     // Текущий индекс в ряду
    
    // Итеративное суммирование ряда до достижения точности или лимита итераций
    while (std::abs(s) > eps && k < maxIter) {
        sum += s;                   // Добавляем текущий член к сумме
        k++;                        // Увеличиваем индекс
        _1 = -_1;                   // Меняем знак: (-1)^k → (-1)^{k+1}
        k2 /= (k * k);              // Обновляем 1/(k!)^2: делим на k^2
        xS *= x2;                   // Умножаем степень: (x/2)^{2(k-1)} → (x/2)^{2k}
        s = _1 * k2 * xS;           // Вычисляем следующий член ряда
    }
    
    // Предупреждение, если ряд не сошёлся
    if (k >= maxIter) {
        std::cerr << "Warning: J0(" << x << ") did not converge in " 
                  << maxIter << " iterations" << std::endl;
    }
    
    return -sum;  // Возвращаем отрицательную сумму, т.к. начальное s = -1 (для k=0 член должен быть +1)
}

// Вспомогательная функция для вычисления части N0 (без логарифмического члена)
// Вычисляет: Ỹ0(x) = Σ_{k=0}^∞ (-1)^k * H_k / (k!)^2 * (x/2)^{2k}
// где H_k = 1 + 1/2 + ... + 1/k - гармоническое число
double _Y0(double x) {
    const double eps = 1e-4;
    const int maxIter = 10000;
    
    double sum = 0.0;
    double s = -1.0;
    double k2 = 1.0;
    double xS = 1.0;
    double x2 = x * x / 4.0;
    double _1 = -1.0;
    double psi = -0.57721566 + 1.0; // ψ(1) = -γ (константа Эйлера-Маскерони ≈ 0.5772)
    long k = 0;
    
    while (std::abs(s) > eps && k < maxIter) {
        sum += s;
        k++;
        _1 = -_1;
        k2 /= (k * k);
        xS *= x2;
        psi += 1.0 / k;            // Обновляем гармоническое число: H_k = H_{k-1} + 1/k
        s = _1 * k2 * xS * psi;    // Член ряда с гармоническим числом
    }
    
    if (k >= maxIter) {
        std::cerr << "Warning: _Y0(" << x << ") did not converge in " 
                  << maxIter << " iterations" << std::endl;
    }
    
    return sum;
}

// Функция Бесселя второго рода нулевого порядка N0(x) = Y0(x)
// Формула: Y0(x) = (2/π) [J0(x)·ln(x/2) + Ỹ0(x)]
double N0(double x) {
    return 2.0 / M_PI * (J0(x) * std::log(x / 2.0) + _Y0(x));
}

// Функция Ханкеля первого рода нулевого порядка
// H0^{(1)}(x) = J0(x) + i·N0(x)
Complex H0_1(double x) {
    return Complex(J0(x), N0(x));
}

// Функция Ханкеля второго рода нулевого порядка  
// H0^{(2)}(x) = J0(x) - i·N0(x)
Complex H0_2(double x) {
    return Complex(J0(x), -N0(x));
}
