#ifndef DIFRONLENTA_H
#define DIFRONLENTA_H

#include "Matrix.h"
#include "Bessel.h"
#include "Chebyshev.h"
#include <vector>
#include <chrono>

// Структура для хранения времени выполнения
struct TimingInfo {
    double matrixBuildTime;  // Время построения матрицы в миллисекундах
    double solveTime;        // Время решения СЛАУ в миллисекундах
    double totalTime;        // Общее время в миллисекундах
};

// Класс для решения задачи дифракции на ленте
class DifrOnLenta {
public:
    // Границы ленты
    double a, b;
    // Длина волны
    double lambda;
    // Параметр усечения
    int N;
    // Угол падения
    double teta;
    // Коэффициенты разложения по полиномам Чебышева
    std::vector<Complex> y;
    // Глубина скин-слоя
    double skinDepth;
    // Импедансный коэффициент χ (chi) для скин-эффекта
    Complex chi;
    
    // Информация о времени выполнения последнего решения
    TimingInfo lastTiming;
    
    // Конструктор
    DifrOnLenta(double _a, double _b, double _lambda, double _teta, int _N, double _skinDepth = 0.0);
    
    // Полином Чебышева на отрезке [a, b]
    double ChebAB(int n, double x);
    
    // Ядро интегрального уравнения
    Complex r(double t, double x);
    
    // Падающее поле
    Complex u0(double x, double z);
    
    // Функция правой части
    Complex f(double x);
    
    // Решение задачи дифракции
    // Возвращает 1 при успехе, 0 при ошибке
    int SolveDifr();
    
private:
    // Расчет импедансного коэффициента χ для скин-эффекта
    Complex CalculateChi();
};

#endif // DIFRONLENTA_H





