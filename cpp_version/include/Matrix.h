#ifndef MATRIX_H
#define MATRIX_H

#include <complex>
#include <vector>
#include <stdexcept>

using Complex = std::complex<double>;

// Класс для комплексного вектора
class CVect {
private:
    std::vector<Complex> v;
    int sz;

public:
    explicit CVect(int size);
    
    int Size() const { return sz; }
    
    Complex& operator[](int index);
    const Complex& operator[](int index) const;
};

// Класс для комплексной матрицы
class CMatr {
private:
    std::vector<CVect> v;
    int sz;

public:
    explicit CMatr(int size);
    
    int Size() const { return sz; }
    
    CVect& operator[](int index);
    const CVect& operator[](int index) const;
};

#endif // MATRIX_H





