#include "DifrOnLentaCuda.h"
#include "Gauss.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
inline void CheckCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

inline void CheckCusolver(cusolverStatus_t status, const char* message) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(message) + " failed");
    }
}

struct DeviceParams {
    double a;
    double b;
    double lambda;
    double theta;
    double skinDepth;
    Complex chi;
    int N;
};

CUDA_HOST_DEVICE inline double ChebABValue(int n, double x, double a, double b) {
    return Cheb(n, 2.0 / (b - a) * x - (b + a) / (b - a));
}

CUDA_HOST_DEVICE inline Complex RKernel(double t, double x, double lambda) {
    double k = 2.0 * M_PI / lambda;
    double delta = fabs(t - x);
    Complex result(J0(k * delta), 0.0);
    const Complex imagUnit(0.0, 1.0);

    if (delta > 1e-8) {
        result = (M_PI * imagUnit / 2.0) * result + (result - Complex(1.0, 0.0)) * log(k * delta / 2.0);
    } else {
        result = (M_PI * imagUnit / 2.0) * result;
    }

    result += Complex(log(k / 2.0) + _Y0(k * delta), 0.0);
    return -result;
}

CUDA_HOST_DEVICE inline Complex U0Kernel(double x, double z, double lambda, double theta) {
    double k = 2.0 * M_PI / lambda;
    double phase = k * (cos(theta) * x + sin(theta) * z);
    return Complex(cos(phase), sin(phase));
}

CUDA_HOST_DEVICE inline Complex FKernel(double x, double lambda, double theta) {
    return Complex(-2.0 * M_PI, 0.0) * U0Kernel(x, 0.0, lambda, theta);
}

CUDA_GLOBAL void BuildMatrixKernel(Complex* matrix, const double* xPoints, DeviceParams params, int quadPoints) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Col index (if logic corresponds to A[k][j])
    int k = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    if (j >= params.N || k >= params.N) {
        return;
    }

    Complex sum(0.0, 0.0);
    for (int m = 0; m < quadPoints; m++) {
        double xm = xPoints[m];
        double Tk = ChebABValue(k, xm, params.a, params.b);
        for (int n = 0; n < quadPoints; n++) {
            double xn = xPoints[n];
            Complex kernel = RKernel(xn, xm, params.lambda);
            double Tj = ChebABValue(j, xn, params.a, params.b);
            sum += kernel * Tj * Tk;
        }
    }

    double scale = M_PI * M_PI / (static_cast<double>(quadPoints) * quadPoints);
    sum *= scale;

    if (j == k) {
        double regReal = (k == 0)
            ? (M_PI * M_PI * log(params.b - params.a))
            : (M_PI * M_PI / (2.0 * (k + 1)));
        sum += Complex(regReal, 0.0);
        if (params.skinDepth > 0.0) {
            const Complex imagUnit(0.0, 1.0);
            sum += imagUnit * params.chi * sum;
        }
    }

    // Store in Column-Major format for cuSOLVER
    // Element A[k][j] (Row k, Col j) should be at index j * Rows + k
    // Here Rows = N.
    matrix[j * params.N + k] = sum;
}

CUDA_GLOBAL void BuildRhsKernel(Complex* rhs, const double* xPoints, DeviceParams params, int quadPoints) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= params.N) {
        return;
    }

    Complex sum(0.0, 0.0);
    for (int m = 0; m < quadPoints; m++) {
        double xm = xPoints[m];
        sum += FKernel(xm, params.lambda, params.theta) * ChebABValue(k, xm, params.a, params.b);
    }

    sum *= M_PI / static_cast<double>(quadPoints);
    rhs[k] = sum;
}
} // namespace

DifrOnLentaCuda::DifrOnLentaCuda(double _a, double _b, double _lambda, double _theta, int _N, double _skinDepth)
    : a(_a), b(_b), lambda(_lambda), theta(_theta), N(_N), skinDepth(_skinDepth) {
    if (N <= 0) {
        throw std::invalid_argument("N must be positive");
    }
    y.resize(N, Complex(0.0, 0.0));
    chi = CalculateChi();
    lastTiming = {0.0, 0.0, 0.0};
}

Complex DifrOnLentaCuda::CalculateChi() const {
    if (skinDepth <= 0.0) {
        return Complex(0.0, 0.0);
    }
    double k = 2.0 * M_PI / lambda;
    double chiValue = k * skinDepth;
    return Complex(chiValue, chiValue);
}

int DifrOnLentaCuda::SolveDifr() {
    using clock = std::chrono::high_resolution_clock;
    auto totalStart = clock::now();

    const int quadPoints = 20;
    std::vector<double> xPoints(quadPoints);
    for (int m = 0; m < quadPoints; m++) {
        xPoints[m] = (b - a) / 2.0 * cos((2 * m + 1) / 2.0 / quadPoints * M_PI) + (b + a) / 2.0;
    }

    size_t matrixSize = static_cast<size_t>(N) * N;
    
    double* d_xPoints = nullptr;
    Complex* d_matrix = nullptr;
    Complex* d_rhs = nullptr;
    int* d_ipiv = nullptr;
    int* d_info = nullptr;

    CheckCuda(cudaMalloc(&d_xPoints, sizeof(double) * quadPoints), "cudaMalloc xPoints");
    CheckCuda(cudaMalloc(&d_matrix, sizeof(Complex) * matrixSize), "cudaMalloc matrix");
    CheckCuda(cudaMalloc(&d_rhs, sizeof(Complex) * N), "cudaMalloc rhs");
    CheckCuda(cudaMalloc(&d_ipiv, sizeof(int) * N), "cudaMalloc ipiv");
    CheckCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc info");

    CheckCuda(cudaMemcpy(d_xPoints, xPoints.data(), sizeof(double) * quadPoints, cudaMemcpyHostToDevice), "cudaMemcpy xPoints");

    DeviceParams params{a, b, lambda, theta, skinDepth, chi, N};

    // --- MATRIX BUILD ---
    cudaEvent_t matrixStart, matrixEnd;
    CheckCuda(cudaEventCreate(&matrixStart), "cudaEventCreate matrixStart");
    CheckCuda(cudaEventCreate(&matrixEnd), "cudaEventCreate matrixEnd");
    CheckCuda(cudaEventRecord(matrixStart), "cudaEventRecord matrixStart");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    BuildMatrixKernel<<<grid, block>>>(d_matrix, d_xPoints, params, quadPoints);
    CheckCuda(cudaGetLastError(), "BuildMatrixKernel launch");

    dim3 rhsBlock(256);
    dim3 rhsGrid((N + rhsBlock.x - 1) / rhsBlock.x);
    BuildRhsKernel<<<rhsGrid, rhsBlock>>>(d_rhs, d_xPoints, params, quadPoints);
    CheckCuda(cudaGetLastError(), "BuildRhsKernel launch");

    CheckCuda(cudaEventRecord(matrixEnd), "cudaEventRecord matrixEnd");
    CheckCuda(cudaEventSynchronize(matrixEnd), "cudaEventSynchronize matrixEnd");

    float matrixMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&matrixMs, matrixStart, matrixEnd), "cudaEventElapsedTime");
    lastTiming.matrixBuildTime = matrixMs;

    // --- SOLVE WITH cuSOLVER ---
    auto solveStart = clock::now();
    
    cusolverDnHandle_t cusolverH = nullptr;
    CheckCusolver(cusolverDnCreate(&cusolverH), "cusolverDnCreate");

    int workSize = 0;
    // Calculate workspace size for Zgetrf
    CheckCusolver(cusolverDnZgetrf_bufferSize(
        cusolverH,
        N,
        N,
        reinterpret_cast<cuDoubleComplex*>(d_matrix),
        N, // LDA
        &workSize
    ), "cusolverDnZgetrf_bufferSize");

    Complex* d_work = nullptr;
    CheckCuda(cudaMalloc(&d_work, sizeof(Complex) * workSize), "cudaMalloc work");

    // LU Factorization
    CheckCusolver(cusolverDnZgetrf(
        cusolverH,
        N,
        N,
        reinterpret_cast<cuDoubleComplex*>(d_matrix),
        N,
        reinterpret_cast<cuDoubleComplex*>(d_work),
        d_ipiv,
        d_info
    ), "cusolverDnZgetrf");

    // Check for singularity
    int h_info = 0;
    CheckCuda(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy info");
    if (h_info != 0) {
        // Cleanup
        cudaFree(d_xPoints); cudaFree(d_matrix); cudaFree(d_rhs); 
        cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
        return -1; // Singular or Error
    }

    // Solve A * x = b (Result stored in d_rhs)
    CheckCusolver(cusolverDnZgetrs(
        cusolverH,
        CUBLAS_OP_N, // No transpose (A is already factorized)
        N,
        1, // nrhs
        reinterpret_cast<cuDoubleComplex*>(d_matrix),
        N, // LDA
        d_ipiv,
        reinterpret_cast<cuDoubleComplex*>(d_rhs),
        N, // LDB
        d_info
    ), "cusolverDnZgetrs");

    auto solveEnd = clock::now();
    lastTiming.solveTime = std::chrono::duration_cast<std::chrono::microseconds>(solveEnd - solveStart).count() / 1000.0;

    // Copy result back
    CheckCuda(cudaMemcpy(y.data(), d_rhs, sizeof(Complex) * N, cudaMemcpyDeviceToHost), "cudaMemcpy result D2H");

    auto totalEnd = clock::now();
    lastTiming.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart).count() / 1000.0;

    // Cleanup
    CheckCuda(cudaEventDestroy(matrixStart), "cudaEventDestroy matrixStart");
    CheckCuda(cudaEventDestroy(matrixEnd), "cudaEventDestroy matrixEnd");
    CheckCuda(cudaFree(d_xPoints), "cudaFree xPoints");
    CheckCuda(cudaFree(d_matrix), "cudaFree matrix");
    CheckCuda(cudaFree(d_rhs), "cudaFree rhs");
    CheckCuda(cudaFree(d_ipiv), "cudaFree ipiv");
    CheckCuda(cudaFree(d_info), "cudaFree info");
    CheckCuda(cudaFree(d_work), "cudaFree work");
    CheckCusolver(cusolverDnDestroy(cusolverH), "cusolverDnDestroy");

    return 1; // Success
}

CUDA_HOST_DEVICE double DifrOnLentaCuda::ChebAB(int n, double x) const {
    return ChebABValue(n, x, a, b);
}

CUDA_HOST_DEVICE Complex DifrOnLentaCuda::r(double t, double x) const {
    return RKernel(t, x, lambda);
}

CUDA_HOST_DEVICE Complex DifrOnLentaCuda::u0(double x, double z) const {
    return U0Kernel(x, z, lambda, theta);
}

CUDA_HOST_DEVICE Complex DifrOnLentaCuda::f(double x) const {
    return FKernel(x, lambda, theta);
}
