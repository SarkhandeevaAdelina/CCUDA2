#ifndef GAUSS_H
#define GAUSS_H

#include "Matrix.h"

// Решение СЛАУ методом Гаусса с выбором главного элемента
// Возвращает 1 при успехе, -1 при ошибке (вырожденная матрица)
int Gauss(CMatr& A, CVect& b, CVect& x);

#endif // GAUSS_H





