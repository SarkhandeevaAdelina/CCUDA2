#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include "CudaCommon.h"
#include <vector>
#include <stdexcept>

class CVect {
private:
    std::vector<Complex> v;
    int sz;

public:
    explicit CVect(int size);

    CUDA_HOST_DEVICE inline int Size() const { return sz; }

    Complex& operator[](int index);
    const Complex& operator[](int index) const;
};

class CMatr {
private:
    std::vector<CVect> v;
    int sz;

public:
    explicit CMatr(int size);

    CUDA_HOST_DEVICE inline int Size() const { return sz; }

    CVect& operator[](int index);
    const CVect& operator[](int index) const;
};

#endif // CUDA_MATRIX_H


