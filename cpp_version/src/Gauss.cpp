#include "Gauss.h"
#include <cmath>

// Решение СЛАУ методом Гаусса с выбором главного элемента
int Gauss(CMatr& A, CVect& b, CVect& x) {
    int N = b.Size();
    
    // Прямой ход метода Гаусса
    for (int i = 0; i < N - 1; i++) {
        // Поиск главного элемента
        double max = std::abs(A[i][i]);
        int maxN = i;
        
        for (int k = i + 1; k < N; k++) {
            double ss = std::abs(A[k][i]);
            if (ss > max) {
                max = ss;
                maxN = k;
            }
        }
        
        // Перестановка строк
        if (maxN != i) {
            for (int k = 0; k < N; k++) {
                Complex s1 = A[i][k];
                A[i][k] = A[maxN][k];
                A[maxN][k] = s1;
            }
            Complex s1 = b[i];
            b[i] = b[maxN];
            b[maxN] = s1;
        }
        
        // Проверка на малость диагонального элемента
        Complex s = 1.0 / A[i][i];
        if (std::abs(s) < 1e-12) {
            return -1;
        }
        
        // Исключение переменной
        for (int j = i + 1; j < N; j++) {
            Complex s1 = A[j][i] * s;
            for (int k = i + 1; k < N; k++) {
                A[j][k] = A[j][k] - s1 * A[i][k];
            }
            b[j] = b[j] - b[i] * s1;
        }
    }
    
    // Проверка последнего диагонального элемента
    if (std::abs(A[N - 1][N - 1]) < 1e-12) {
        return -1;
    }
    
    // Обратный ход метода Гаусса
    x[N - 1] = b[N - 1] / A[N - 1][N - 1];
    
    for (int i = N - 2; i >= 0; i--) {
        Complex s = b[i];
        for (int j = N - 1; j > i; j--) {
            s = s - A[i][j] * x[j];
        }
        x[i] = s / A[i][i];
    }
    
    return 1;
}





