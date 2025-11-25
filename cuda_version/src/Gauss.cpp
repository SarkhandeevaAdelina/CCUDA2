#include "Gauss.h"

int Gauss(CMatr& A, CVect& b, CVect& x) {
    int N = b.Size();

    for (int i = 0; i < N - 1; i++) {
        double maxVal = ComplexAbs(A[i][i]);
        int maxN = i;

        for (int k = i + 1; k < N; k++) {
            double ss = ComplexAbs(A[k][i]);
            if (ss > maxVal) {
                maxVal = ss;
                maxN = k;
            }
        }

        if (maxN != i) {
            for (int k = 0; k < N; k++) {
                Complex temp = A[i][k];
                A[i][k] = A[maxN][k];
                A[maxN][k] = temp;
            }
            Complex temp = b[i];
            b[i] = b[maxN];
            b[maxN] = temp;
        }

        Complex diag = A[i][i];
        double absDiag = ComplexAbs(diag);
        if (absDiag < 1e-12) {
            return -1;
        }
        Complex invDiag = Complex(1.0, 0.0) / diag;

        for (int j = i + 1; j < N; j++) {
            Complex factor = A[j][i] * invDiag;
            for (int k = i + 1; k < N; k++) {
                A[j][k] = A[j][k] - factor * A[i][k];
            }
            b[j] = b[j] - b[i] * factor;
        }
    }

    if (ComplexAbs(A[N - 1][N - 1]) < 1e-12) {
        return -1;
    }

    x[N - 1] = b[N - 1] / A[N - 1][N - 1];

    for (int i = N - 2; i >= 0; i--) {
        Complex sum = b[i];
        for (int j = N - 1; j > i; j--) {
            sum = sum - A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return 1;
}


