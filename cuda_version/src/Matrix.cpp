#include "Matrix.h"
#include <sstream>

CVect::CVect(int size) : v(size, Complex(0.0, 0.0)), sz(size) {
    if (size <= 0) {
        throw std::invalid_argument("CVect size must be positive");
    }
}

Complex& CVect::operator[](int index) {
    if (index < 0 || index >= sz) {
        std::ostringstream oss;
        oss << "CVect index " << index << " out of range [0, " << (sz - 1) << "]";
        throw std::out_of_range(oss.str());
    }
    return v[index];
}

const Complex& CVect::operator[](int index) const {
    if (index < 0 || index >= sz) {
        std::ostringstream oss;
        oss << "CVect index " << index << " out of range [0, " << (sz - 1) << "]";
        throw std::out_of_range(oss.str());
    }
    return v[index];
}

CMatr::CMatr(int size) : sz(size) {
    if (size <= 0) {
        throw std::invalid_argument("CMatr size must be positive");
    }
    v.reserve(size);
    for (int i = 0; i < size; i++) {
        v.emplace_back(size);
    }
}

CVect& CMatr::operator[](int index) {
    if (index < 0 || index >= sz) {
        std::ostringstream oss;
        oss << "CMatr index " << index << " out of range [0, " << (sz - 1) << "]";
        throw std::out_of_range(oss.str());
    }
    return v[index];
}

const CVect& CMatr::operator[](int index) const {
    if (index < 0 || index >= sz) {
        std::ostringstream oss;
        oss << "CMatr index " << index << " out of range [0, " << (sz - 1) << "]";
        throw std::out_of_range(oss.str());
    }
    return v[index];
}


