#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "DifrOnLenta.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Функция для вывода коэффициентов Чебышева
void PrintCoefficients(const DifrOnLenta& solver) {
    std::cout << "\nКоэффициенты Чебышева (" << solver.N << " шт):\n";
    std::cout << std::left << std::setw(6) << "#" 
              << std::setw(30) << "Значение" << "\n";
    
    for (int i = 0; i < solver.N; i++) {
        std::cout << std::left << std::setw(6) << (i + 1)
                  << std::fixed << std::setprecision(4)
                  << std::real(solver.y[i]) << "+" 
                  << std::imag(solver.y[i]) << "i\n";
    }
}

// Функция для запуска решения с замером времени
bool SolveAndPrint(double a, double b, double lambda, double theta, int N, double skinDepth, bool verbose = true) {
    DifrOnLenta solver(a, b, lambda, theta, N, skinDepth);
    
    int result = solver.SolveDifr();
    
    if (result != 1) {
        std::cerr << "Ошибка решения СЛАУ!\n";
        return false;
    }
    
    if (verbose) {
        std::cout << "\nПараметры:\n";
        std::cout << "a = " << a << ", b = " << b << "\n";
        std::cout << "λ = " << lambda << ", θ = " << (theta * 180.0 / M_PI) << "°\n";
        std::cout << "N = " << N << ", δ = " << skinDepth << "\n";
        std::cout << "\nИмпедансный коэффициент χ: "
                  << std::fixed << std::setprecision(6)
                  << std::real(solver.chi) << " + " 
                  << std::imag(solver.chi) << "i\n";
        
        PrintCoefficients(solver);
        
        std::cout << "\nВремя выполнения:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Построение матрицы: " << solver.lastTiming.matrixBuildTime << " мс\n";
        std::cout << "Решение СЛАУ: " << solver.lastTiming.solveTime << " мс\n";
        std::cout << "Общее: " << solver.lastTiming.totalTime << " мс\n";
    }
    
    return true;
}

int main() {
    // Установка кодировки для корректного вывода русских символов в консоль
    #ifdef _WIN32
    system("chcp 65001 > nul");
    #endif
    
    std::cout << "\nРешатель дифракции на ленте (CPU)\n";
    
    // Параметры по умолчанию
    const double a = -1.0;
    const double b = 1.0;
    const double lambda = 1.0;
    const double theta = M_PI / 4.0;  // 45 градусов
    const int N_default = 10;
    const double skinDepth = 0.1;
    
    // 1. Решение с параметрами по умолчанию
    std::cout << "\n--- Случай 1: Идеальный проводник (без скин-эффекта) ---\n";
    SolveAndPrint(a, b, lambda, theta, N_default, 0.0, true);

    std::cout << "\n--- Случай 2: С учетом скин-эффекта (delta = " << skinDepth << ") ---\n";
    SolveAndPrint(a, b, lambda, theta, N_default, skinDepth, true);
    
    // 2. Тестирование зависимости от N
    std::cout << "\n\nТестирование зависимости от N\n";
    
    // Расширенный набор N для сравнения с GPU
    std::vector<int> N_values = {10, 20, 50, 100, 200, 500};
    
    std::cout << "\n" << std::left 
              << std::setw(8) << "N" 
              << std::setw(22) << "Время матрицы (мс)"
              << std::setw(22) << "Время СЛАУ (мс)"
              << std::setw(18) << "Общее (мс)" << "\n";
    
    for (int N : N_values) {
        DifrOnLenta solver(a, b, lambda, theta, N, skinDepth);
        int result = solver.SolveDifr();
        
        if (result == 1) {
            std::cout << std::left << std::setw(8) << N
                      << std::fixed << std::setprecision(3)
                      << std::setw(22) << solver.lastTiming.matrixBuildTime
                      << std::setw(22) << solver.lastTiming.solveTime
                      << std::setw(18) << solver.lastTiming.totalTime << "\n";
        } else {
            std::cout << std::left << std::setw(8) << N
                      << "Ошибка решения!\n";
        }
    }
    
    std::cout << "\n\nДанные для построения графиков (CSV):\n";
    std::cout << "N,MatrixTime_ms,SolveTime_ms,TotalTime_ms\n";
    
    for (int N : N_values) {
        DifrOnLenta solver(a, b, lambda, theta, N, skinDepth);
        int result = solver.SolveDifr();
        
        if (result == 1) {
            std::cout << N << ","
                      << std::fixed << std::setprecision(6)
                      << solver.lastTiming.matrixBuildTime << ","
                      << solver.lastTiming.solveTime << ","
                      << solver.lastTiming.totalTime << "\n";
        }
    }
    
    return 0;
}
