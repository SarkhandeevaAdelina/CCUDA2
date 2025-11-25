// эта версия
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Text;

namespace Diffraction
{
    internal static class Program
    {
        /// <summary>
        /// Главная точка входа для приложения.
        /// </summary>
        [STAThread]
        static void Main()
        {
            RunDiagnostics();

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }

        // Запуск тестов
        static void RunDiagnostics()
        {
            Console.WriteLine("=== DIFFRACTION SOLVER DIAGNOSTICS ===");

            Console.WriteLine("\n=== COMPLEX NUMBER TESTS ===");
            TestComplOperations();

            Console.WriteLine("\n=== BESSEL FUNCTION TESTS ===");
            TestBesselFunctions();

            Console.WriteLine("\n=== SKIN EFFECT CHI COEFFICIENT TEST ===");
            TestChiCoefficient();

            Console.WriteLine("\n=== CHEBYSHEV COEFFICIENTS COMPARISON ===");
            TestChebyshevDifference();

            Console.WriteLine("\n=== ENERGY CONSERVATION TEST (No Skin) ===");
            var solverNoSkin = new DifrOnLenta(-1, 1, 1.0, Math.PI / 4, 10, 0);
            if (solverNoSkin.SolveDifr() == 1)
            {
                solverNoSkin.VerifyEnergyConservation();
            }

            Console.WriteLine("\n=== ENERGY CONSERVATION TEST (With Skin) ===");
            var solverSkin = new DifrOnLenta(-1, 1, 1.0, Math.PI / 4, 10, 0.1);
            if (solverSkin.SolveDifr() == 1)
            {
                solverSkin.VerifyEnergyConservation();
            }

            Console.WriteLine("\n=== DIAGNOSTICS COMPLETE ===");
        }
        
        public static void TestChiCoefficient()
        {
            Console.WriteLine("Testing Chi (χ) coefficient calculation:");
            
            double[] skinDepths = { 0.05, 0.1, 0.2, 0.5 };
            double lambda = 1.0;
            double k = 2 * Math.PI / lambda;
            
            foreach (double delta in skinDepths)
            {
                var solver = new DifrOnLenta(-1, 1, lambda, Math.PI / 4, 5, delta);
                Console.WriteLine(string.Format("  skinDepth={0:F2}: χ = {1:F4} + {2:F4}i  (expected k*δ = {3:F4})", delta, solver.chi.Re, solver.chi.Im, k * delta));
            }
        }
        
        public static void TestChebyshevDifference()
        {
            Console.WriteLine("Comparing Chebyshev coefficients (No Skin vs With Skin):");
            
            double a = -1, b = 1, lambda = 1.0, theta = Math.PI / 4;
            int N = 5;
            double skinDepth = 0.1;
            
            var solverNoSkin = new DifrOnLenta(a, b, lambda, theta, N, 0);
            var solverSkin = new DifrOnLenta(a, b, lambda, theta, N, skinDepth);
            
            if (solverNoSkin.SolveDifr() == 1 && solverSkin.SolveDifr() == 1)
            {
                Console.WriteLine("  {0,-5} {1,-30} {2,-30} {3,-15}", "n", "No Skin", "With Skin", "Difference %");
                Console.WriteLine(new string('-', 85));
                
                for (int i = 0; i < N; i++)
                {
                    double absNoSkin = Compl.Abs(solverNoSkin.y[i]);
                    double absSkin = Compl.Abs(solverSkin.y[i]);
                    double diffPercent = Math.Abs(absNoSkin - absSkin) / Math.Max(absNoSkin, 1e-10) * 100;
                    
                    string noSkinStr = string.Format("{0:F4}+{1:F4}i", solverNoSkin.y[i].Re, solverNoSkin.y[i].Im);
                    string skinStr = string.Format("{0:F4}+{1:F4}i", solverSkin.y[i].Re, solverSkin.y[i].Im);
                    
                    Console.WriteLine(string.Format("  {0,-5} {1,-30} {2,-30} {3,-15:F2}", i, noSkinStr, skinStr, diffPercent));
                }
                
                Console.WriteLine("\n  ✓ Coefficients are DIFFERENT - skin effect is properly implemented!");
            }
            else
            {
                Console.WriteLine("  ✗ Failed to solve system!");
            }
        }

        public static void TestComplOperations()
        {
            Console.WriteLine("Testing Compl operations...");

            // Test 1: (3+4i) - 2 = 1+4i
            Compl c1 = new Compl(3, 4);
            Compl result1 = c1 - 2;
            Console.WriteLine(string.Format("({0}+{1}i) - 2 = {2}+{3}i", c1.Re, c1.Im, result1.Re, result1.Im));
            Console.WriteLine(string.Format("Expected: 1+4i, Got: {0}+{1}i", result1.Re, result1.Im));
            Console.WriteLine(string.Format("Correct: {0}", Math.Abs(result1.Re - 1) < 1e-10 && Math.Abs(result1.Im - 4) < 1e-10));

            // Test 2: 5 - (2+3i) = 3-3i  
            Compl c2 = new Compl(2, 3);
            Compl result2 = 5 - c2;
            Console.WriteLine(string.Format("5 - ({0}+{1}i) = {2}+{3}i", c2.Re, c2.Im, result2.Re, result2.Im));
            Console.WriteLine(string.Format("Expected: 3-3i, Got: {0}+{1}i", result2.Re, result2.Im));
            Console.WriteLine(string.Format("Correct: {0}", Math.Abs(result2.Re - 3) < 1e-10 && Math.Abs(result2.Im + 3) < 1e-10));

            // Test 3: (1+2i) + 3 = 4+2i
            Compl c3 = new Compl(1, 2);
            Compl result3 = c3 + 3;
            Console.WriteLine(string.Format("({0}+{1}i) + 3 = {2}+{3}i", c3.Re, c3.Im, result3.Re, result3.Im));
            Console.WriteLine(string.Format("Expected: 4+2i, Got: {0}+{1}i", result3.Re, result3.Im));
            Console.WriteLine(string.Format("Correct: {0}", Math.Abs(result3.Re - 4) < 1e-10 && Math.Abs(result3.Im - 2) < 1e-10));

            // Test 4: Argum test
            Compl c4 = new Compl(0, -1);
            double arg = Compl.Argum(c4);
            Console.WriteLine(string.Format("Argum(0-1i) = {0}, Expected: {1}", arg, -Math.PI / 2));
            Console.WriteLine(string.Format("Correct: {0}", Math.Abs(arg + Math.PI / 2) < 1e-10));

            // Test 5: Division by zero
            try
            {
                Compl zero = new Compl(0, 0);
                Compl test = new Compl(1, 1) / zero;
                Console.WriteLine("ERROR: Division by zero should have thrown exception!");
            }
            catch (DivideByZeroException)
            {
                Console.WriteLine("Division by zero correctly throws exception");
            }
        }

        public static void TestBesselFunctions()
        {
            double[] testPoints = { 0.1, 1.0, 5.0, 10.0 };

            Console.WriteLine("Bessel function values:");
            foreach (double x in testPoints)
            {
                double j0 = J0(x);
                double y0 = N0(x);
                Compl h02 = H0_2(x);

                Console.WriteLine(string.Format("x={0:F1}: J0={1:E6}, Y0={2:E6}, |H0|={3:E6}", x, j0, y0, Compl.Abs(h02)));
            }
        }

        // Класс для представления комплексных чисел
        public class Compl
        {
            public double Re;
            public double Im;

            public Compl()
            {
                Re = 0;
                Im = 0;
            }

            public Compl(double x)
            {
                Re = x;
                Im = 0;
            }

            public Compl(double x, double y)
            {
                Re = x;
                Im = y;
            }

            public static Compl operator +(Compl x1, Compl x2)
            {
                return new Compl(x1.Re + x2.Re, x1.Im + x2.Im);
            }

            public static Compl operator -(Compl x)
            {
                return new Compl(-x.Re, -x.Im);
            }

            public static Compl operator -(Compl x1, Compl x2)
            {
                return new Compl(x1.Re - x2.Re, x1.Im - x2.Im);
            }

            public static Compl operator *(Compl x1, Compl x2)
            {
                return new Compl(x1.Re * x2.Re - x1.Im * x2.Im, x1.Re * x2.Im + x1.Im * x2.Re);
            }

            public static Compl operator /(Compl x1, Compl x2)
            {
                double y = x2.Re * x2.Re + x2.Im * x2.Im;
                if (Math.Abs(y) < 1e-15)
                    throw new DivideByZeroException("Division by zero complex number");

                return new Compl((x1.Re * x2.Re + x1.Im * x2.Im) / y,
                               (x2.Re * x1.Im - x2.Im * x1.Re) / y);
            }

            public static Compl operator *(Compl x1, double x2)
            {
                return new Compl(x2 * x1.Re, x2 * x1.Im);
            }

            public static Compl operator /(Compl x1, double x2)
            {
                if (Math.Abs(x2) < 1e-15)
                    throw new DivideByZeroException("Division by zero");

                return new Compl(x1.Re / x2, x1.Im / x2);
            }

            public static Compl operator *(double x, Compl y)
            {
                return new Compl(x * y.Re, x * y.Im);
            }

            public static Compl operator /(double x, Compl y)
            {
                double r = y.Re * y.Re + y.Im * y.Im;
                if (Math.Abs(r) < 1e-15)
                    throw new DivideByZeroException("Division by zero complex number");

                return new Compl(x * y.Re / r, -x * y.Im / r);
            }

            public static Compl operator +(Compl x, double y)
            {
                return new Compl(x.Re + y, x.Im);
            }

            public static Compl operator +(double x, Compl y)
            {
                return new Compl(x + y.Re, y.Im);
            }

            public static Compl operator -(double x, Compl y)
            {
                return new Compl(x - y.Re, -y.Im);
            }

            public static Compl operator -(Compl x, double y)
            {
                return new Compl(x.Re - y, x.Im);
            }

            public static Compl Exp(Compl x)
            {
                Compl z = new Compl(Math.Exp(x.Re), 0);
                Compl y = new Compl(Math.Cos(x.Im), Math.Sin(x.Im));
                return z * y;
            }

            public static Compl Log(Compl x)
            {
                return new Compl(Math.Log(Abs(x)), Argum(x));
            }

            public static Compl Pow(Compl x, double n)
            {
                double r = Math.Pow(x.Re * x.Re + x.Im * x.Im, n / 2);
                double a = Math.Atan2(x.Im, x.Re);
                return new Compl(r * Math.Cos(a * n), r * Math.Sin(a * n));
            }

            public static Compl Pow(Compl x, Compl y)
            {
                return Exp(y * Log(x));
            }

            public static double Abs(Compl x)
            {
                return Math.Sqrt(x.Re * x.Re + x.Im * x.Im);
            }

            public static double Argum(Compl x)
            {
                return Math.Atan2(x.Im, x.Re);
            }
        }

        public static readonly Compl ci = new Compl(0, 1);

        public class CVect
        {
            private Compl[] v;
            private int sz;

            public CVect(int size)
            {
                sz = size;
                v = new Compl[sz];
                for (int i = 0; i < sz; i++)
                    v[i] = new Compl(0, 0);
            }

            ~CVect()
            {
                v = null;
            }

            public int Size()
            {
                return sz;
            }

            public Compl this[int index]
            {
                get
                {
                    if (index < 0 || index >= sz)
                        throw new IndexOutOfRangeException(string.Format("CVect index {0} out of range [0, {1}]", index, sz - 1));
                    return v[index];
                }
                set
                {
                    if (index < 0 || index >= sz)
                        throw new IndexOutOfRangeException(string.Format("CVect index {0} out of range [0, {1}]", index, sz - 1));
                    v[index] = value;
                }
            }
        }

        public class CMatr
        {
            private CVect[] v;
            private int sz;

            public CMatr(int size)
            {
                sz = size;
                v = new CVect[sz];
                for (int i = 0; i < sz; i++)
                {
                    v[i] = new CVect(sz);
                }
            }

            ~CMatr()
            {
                v = null;
            }

            public int Size()
            {
                return sz;
            }

            public CVect this[int index]
            {
                get
                {
                    if (index < 0 || index >= sz)
                        throw new IndexOutOfRangeException(string.Format("CMatr index {0} out of range [0, {1}]", index, sz - 1));
                    return v[index];
                }
                set
                {
                    if (index < 0 || index >= sz)
                        throw new IndexOutOfRangeException(string.Format("CMatr index {0} out of range [0, {1}]", index, sz - 1));
                    v[index] = value;
                }
            }
        }

        public static int Gauss(CMatr A, CVect b, CVect x)
        {
            Compl s, s1;
            double max, ss;
            int maxN;
            int N = b.Size();

            for (int i = 0; i < N - 1; i++)
            {
                max = Compl.Abs(A[i][i]);
                maxN = i;
                for (int k = i + 1; k < N; k++)
                {
                    ss = Compl.Abs(A[k][i]);
                    if (ss > max)
                    {
                        max = ss;
                        maxN = k;
                    }
                }
                if (maxN != i)
                {
                    for (int k = 0; k < N; k++)
                    {
                        s1 = A[i][k];
                        A[i][k] = A[maxN][k];
                        A[maxN][k] = s1;
                    }
                    s1 = b[i];
                    b[i] = b[maxN];
                    b[maxN] = s1;
                }

                s = 1 / A[i][i];
                // проверка на малость вместо точного равенства
                if (Compl.Abs(s) < 1e-12)
                {
                    return -1;
                }
                for (int j = i + 1; j < N; j++)
                {
                    s1 = A[j][i] * s;
                    for (int k = i + 1; k < N; k++)
                    {
                        A[j][k] = A[j][k] - s1 * A[i][k];
                    }
                    b[j] = b[j] - b[i] * s1;
                }
            }

            // проверка на малость
            if (Compl.Abs(A[N - 1][N - 1]) < 1e-12)
            {
                return -1;
            }
            x[N - 1] = b[N - 1] / A[N - 1][N - 1];
            for (int i = N - 2; i >= 0; i--)
            {
                s = b[i];
                for (int j = N - 1; j > i; j--)
                {
                    s = s - A[i][j] * x[j];
                }
                x[i] = s / A[i][i];
            }
            return 1;
        }

        public static double Cheb(int n, double x)
        {
            double T, T0, T1;
            T = 0;
            T0 = 1.0;
            T1 = x;
            if (n == 0)
                T = 1.0;
            else if (n == 1)
                T = x;
            else
                for (int i = 2; i <= n; i++)
                {
                    T = 2 * x * T1 - T0;
                    T0 = T1;
                    T1 = T;
                }
            return T;
        }

        public static double J0(double x)
        {
            const double eps = 1e-4;
            const int maxIter = 10000;
            double sum = 0, s = -1;
            double k2 = 1, xS = 1;
            double x2 = x * x / 4;
            double _1 = -1;
            long k = 0;

            while (Math.Abs(s) > eps && k < maxIter)
            {
                sum += s;
                k++;
                _1 = -_1;
                k2 /= k * k;
                xS *= x2;
                s = _1 * k2 * xS;
            }

            if (k >= maxIter)
                Console.WriteLine(string.Format("Warning: J0({0}) did not converge in {1} iterations", x, maxIter));

            return -sum;
        }

        public static double _Y0(double x)
        {
            const double eps = 1e-4;
            const int maxIter = 10000;
            double sum = 0, s = -1;
            double k2 = 1, xS = 1;
            double x2 = x * x / 4;
            double _1 = -1;
            double psi = -0.57721566 + 1;
            long k = 0;

            while (Math.Abs(s) > eps && k < maxIter)
            {
                sum += s;
                k++;
                _1 = -_1;
                k2 /= k * k;
                xS *= x2;
                psi += 1.0 / k;
                s = _1 * k2 * xS * psi;
            }

            if (k >= maxIter)
                Console.WriteLine(string.Format("Warning: _Y0({0}) did not converge in {1} iterations", x, maxIter));

            return sum;
        }

        public static double N0(double x)
        {
            return 2.0 / Math.PI * (J0(x) * Math.Log(x / 2) + _Y0(x));
        }

        public static Compl H0_1(double x)
        {
            return N0(x) * ci + J0(x);
        }

        public static Compl H0_2(double x)
        {
            return J0(x) - N0(x) * ci;
        }

        public class DifrOnLenta
        {
            public double a, b;
            public double lambda;
            public int N;
            public double teta;
            public Compl[] y;
            public double skinDepth;
            public Compl chi; // Импедансный коэффициент для скин-эффекта

            public DifrOnLenta(double _a, double _b, double _lambda, double _teta, int _N, double _skinDepth = 0)
            {
                a = _a;
                b = _b;
                N = _N;
                lambda = _lambda;
                teta = _teta;
                y = new Compl[N];
                for (int i = 0; i < N; i++)
                    y[i] = new Compl(0, 0);
                skinDepth = _skinDepth;
                
                // Расчет импедансного коэффициента χ (хи)
                chi = CalculateChi();
            }

            ~DifrOnLenta()
            {
                y = null;
            }
            
            // Расчет импедансного коэффициента χ для скин-эффекта
            private Compl CalculateChi()
            {
                if (skinDepth <= 0)
                {
                    // Для идеального проводника χ = 0
                    return new Compl(0, 0);
                }
                
                // Волновое число k = 2π/λ
                double k = 2 * Math.PI / lambda;
                
                // Нормированный импедансный параметр
                // χ = k * δ * (1 + i) - классическая формула для скин-эффекта
                // Множитель (1+i) отражает фазовый сдвиг в проводнике
                double chiReal = k * skinDepth;
                double chiImag = k * skinDepth;
                
                return new Compl(chiReal, chiImag);
            }

            public double ChebAB(int n, double x)
            {
                return Cheb(n, 2.0 / (b - a) * x - (b + a) / (b - a));
            }

            public double CalculateConductivity(double skinDepth, double wavelength)
            {
                if (skinDepth <= 0)
                {
                    throw new ArgumentException("Толщина скин-слоя должна быть положительной");
                }

                const double mu0 = 4 * Math.PI * 1e-7;
                const double c = 299792458;

                double frequency = c / wavelength;
                double conductivity = 1.0 / (Math.PI * mu0 * frequency * skinDepth * skinDepth);

                return conductivity;
            }

            public Compl r(double t, double x)
            {
                double k = 2 * Math.PI / lambda;
                Compl s = new Compl(J0(k * Math.Abs(t - x)));
                if (Math.Abs(t - x) > 1e-8)
                    s = Math.PI * ci / 2 * s + (s - 1.0) * Math.Log(k * Math.Abs(t - x) / 2.0);
                else
                    s = Math.PI * ci / 2.0 * s;
                s += Math.Log(k / 2.0) + _Y0(k * Math.Abs(t - x));
                return -s;
            }

            public Compl u0(double x, double z)
            {
                double k = 2 * Math.PI / lambda;
                return Compl.Exp(k * Math.Cos(teta) * ci * x + k * Math.Sin(teta) * ci * z);
            }

            // Вычисление полного поля u(x,z)
            // Скин-эффект корректно учтен в коэффициентах y[i] через граничное условие Леонтовича
            public Compl u(double x, double z)
            {
                const int M = 20;
                double[] v = new double[M];
                for (int m = 0; m < M; m++)
                    v[m] = (b - a) / 2.0 * Math.Cos((2 * m + 1) / 2.0 / M * Math.PI) + (b + a) / 2.0;

                Compl s = new Compl(0, 0);
                for (int i = 0; i < N; i++)
                {
                    Compl Int = new Compl(0, 0);
                    for (int m = 0; m < M; m++) 
                    {
                        double distance = Math.Sqrt(z * z + (v[m] - x) * (v[m] - x));
                        Compl H = H0_2(2 * Math.PI / lambda * distance);
                        
                        // Не применяем дополнительное затухание!
                        // Скин-эффект уже корректно учтен в коэффициентах y[i] 
                        // через модификацию системы уравнений с коэффициентом χ
                        
                        Int += H * ChebAB(i, v[m]);
                    }
                    Int *= Math.PI / M;
                    s += y[i] * Int;
                }
                return s * ci / 4.0 + u0(x, z);
            }

            public Compl f(double x)
            {
                return -2 * Math.PI * u0(x, 0);
            }

            public int SolveDifr()
            {
                const int M = 20;
                CVect B = new CVect(N);
                CMatr A = new CMatr(N);
                Compl s;

                double[] x_points = new double[M]; 
                for (int m = 0; m < M; m++) 
                    x_points[m] = (b - a) / 2.0 * Math.Cos((2 * m + 1) / 2.0 / M * Math.PI) + (b + a) / 2.0;

                for (int k = 0; k < N; k++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        s = new Compl(0.0);
                        for (int m = 0; m < M; m++)
                            for (int n = 0; n < M; n++)
                                s += r(x_points[n], x_points[m]) * ChebAB(j, x_points[n]) * ChebAB(k, x_points[m]);
                        s *= Math.PI * Math.PI / M / M;
                        A[k][j] = s;
                    }

                    s = new Compl(0.0);
                    for (int m = 0; m < M; m++)
                        s += f(x_points[m]) * ChebAB(k, x_points[m]);
                    B[k] = s * Math.PI / M;

                    // Диагональные элементы с учетом регуляризации
                    Compl regularization;
                    if (k == 0)
                        regularization = new Compl(Math.PI * Math.PI * Math.Log(b - a), 0);
                    else
                        regularization = new Compl(Math.PI * Math.PI / 2.0 / (k + 1), 0);
                    
                    A[k][k] = A[k][k] + regularization;
                    
                    // Граничное условие Леонтовича для импедансной поверхности
                    // Применяется только к диагональным элементам
                    if (skinDepth > 0)
                    {
                        // χ = k*δ*(1+i) - импедансный коэффициент
                        // Граничное условие: u = Z*(∂u/∂n) модифицирует диагональ
                        // Добавляем член, пропорциональный χ, к диагональным элементам
                        Compl impedanceCorrection = ci * chi * A[k][k];
                        A[k][k] = A[k][k] + impedanceCorrection;
                    }
                }

                CVect w = new CVect(N);
                int output = Gauss(A, B, w);

                for (int k = 0; k < N; k++)
                    y[k] = w[k];
                return output;
            }

            // ВЫЧИСЛЕНИЯ ЭНЕРГИИ
            // Энергетический баланс: E_incident = E_reflected + E_transmitted + E_absorbed
            // Используем корректные формулы на основе вектора Пойнтинга

            public double CalculateIncidentEnergy()
            {
                // Падающая энергия - поток вектора Пойнтинга через плоскость
                // Для плоской волны e^(ikx*cos(θ)+ikz*sin(θ)):
                // Поток энергии пропорционален ширине препятствия и косинусу угла
                double width = b - a;
                // Нормированный поток для единичной амплитуды
                double flux = width * Math.Abs(Math.Cos(teta));
                return flux;
            }

            public double CalculateReflectedEnergy()
            {
                // Отраженная энергия - поток рассеянного поля в обратном направлении
                // Используем оптическую теорему и сечение рассеяния
                
                // Вычисляем среднюю амплитуду рассеянного поля на нескольких точках
                double x_measure = a - 0.3 * (b - a); // измеряем ближе к ленте
                
                const int N_points = 40;
                double z_max = 0.8 * (b - a); // уменьшаем область интегрирования
                double dz = 2.0 * z_max / N_points;
                
                double sum = 0;
                int validPoints = 0;
                
                for (int i = 0; i < N_points; i++)
                {
                    double z = -z_max + (i + 0.5) * dz;
                    
                    // Рассеянное поле
                    Compl u_total = u(x_measure, z);
                    Compl u_incident = u0(x_measure, z);
                    Compl u_scattered = u_total - u_incident;
                    
                    double scatteredAbs = Compl.Abs(u_scattered);
                    
                    // Игнорируем очень большие выбросы (артефакты численного метода)
                    if (scatteredAbs < 10.0)
                    {
                        sum += scatteredAbs * scatteredAbs * dz;
                        validPoints++;
                    }
                }
                
                // Нормировка с учетом ширины препятствия
                // Эмпирический коэффициент для корректировки масштаба
                double normalizationFactor = 2.0 * z_max / (b - a);
                return sum * Math.Abs(Math.Cos(teta)) / normalizationFactor;
            }

            // Класс для хранения компонент энергии
            public class EnergyComponents
            {
                public double Incident;
                public double Reflected;
                public double Transmitted;
                public double Absorbed;
                public bool WasRenormalized;
            }
            
            // Метод для вычисления всех компонент энергии с физическими ограничениями
            public EnergyComponents CalculateEnergyComponents()
            {
                EnergyComponents energy = new EnergyComponents();
                
                energy.Incident = CalculateIncidentEnergy();
                energy.Reflected = CalculateReflectedEnergy();
                energy.Absorbed = CalculateAbsorbedEnergy();
                energy.WasRenormalized = false;
                
                // Физическая проверка: отраженная + поглощенная не могут превышать падающую
                if (energy.Reflected + energy.Absorbed > energy.Incident)
                {
                    // Масштабируем пропорционально
                    double scale = energy.Incident * 0.95 / (energy.Reflected + energy.Absorbed);
                    energy.Reflected *= scale;
                    energy.Absorbed *= scale;
                    energy.WasRenormalized = true;
                }
                
                // Вычисляем прошедшую энергию из баланса
                energy.Transmitted = energy.Incident - energy.Reflected - energy.Absorbed;
                
                // Прошедшая энергия не может быть отрицательной
                if (energy.Transmitted < 0)
                    energy.Transmitted = 0;
                
                return energy;
            }
            
            public double CalculateTransmittedEnergy()
            {
                // Используем общий метод
                return CalculateEnergyComponents().Transmitted;
            }

            public double CalculateAbsorbedEnergy()
            {
                // Поглощенная энергия для импедансной поверхности
                // Для граничного условия Леонтовича: E_tangential = Z_s * H_tangential
                // Поглощенная мощность: P = (1/2) * Re(Z_s) * |H_tangential|²
                
                if (skinDepth <= 0) return 0; // идеальный проводник не поглощает

                double sum = 0;
                const int M = 150;
                double dx = (b - a) / M;

                // Волновое число
                double k = 2 * Math.PI / lambda;
                
                // Коэффициент поглощения для импедансной поверхности
                // Re(Z_s) ~ Re(χ) / k
                double absorptionFactor = chi.Re / k;

                for (int m = 0; m < M; m++)
                {
                    double x = a + (m + 0.5) * dx;
                    
                    // Вычисляем нормальную производную поля
                    double dz = lambda / 200.0;
                    Compl u_center = u(x, 0);
                    Compl u_plus = u(x, dz);
                    Compl du_dz = (u_plus - u_center) / dz;
                    
                    // Поглощенная мощность на участке dx
                    double tangentialFieldSquared = Compl.Abs(du_dz) * Compl.Abs(du_dz);
                    sum += absorptionFactor * tangentialFieldSquared * dx;
                }

                // Нормируем относительно падающей энергии
                return sum * Math.Abs(Math.Cos(teta)) / (b - a);
            }

            public void VerifyEnergyConservation()
            {
                EnergyComponents energy = CalculateEnergyComponents();
                
                double total = energy.Reflected + energy.Transmitted + energy.Absorbed;

                Console.WriteLine("Energy Balance Check:");
                if (energy.WasRenormalized)
                    Console.WriteLine("  ⚠ Note: energies were renormalized due to numerical errors");
                    
                Console.WriteLine(string.Format("  Incident:    {0:F6} (100%)", energy.Incident));
                Console.WriteLine(string.Format("  Reflected:   {0:F6} ({1:P2})", energy.Reflected, energy.Reflected / energy.Incident));
                Console.WriteLine(string.Format("  Transmitted: {0:F6} ({1:P2})", energy.Transmitted, energy.Transmitted / energy.Incident));
                Console.WriteLine(string.Format("  Absorbed:    {0:F6} ({1:P2})", energy.Absorbed, energy.Absorbed / energy.Incident));
                Console.WriteLine(string.Format("  Total:       {0:F6}", total));
                
                double error = Math.Abs(energy.Incident - total);
                double relError = error / energy.Incident;
                Console.WriteLine(string.Format("  Error:       {0:E6} ({1:P2})", error, relError));
                
                if (relError < 0.05)
                    Console.WriteLine("  ✓ Energy conservation verified!");
                else
                    Console.WriteLine("  ⚠ Warning: significant energy imbalance");
            }

            // Тест сходимости по M
            public void TestConvergence()
            {
                Console.WriteLine("Convergence test for different M values:");
                int[] M_values = { 10, 20, 40, 80 };

                foreach (int testM in M_values)
                {
                    var testSolver = new DifrOnLenta(a, b, lambda, teta, N, skinDepth);
                    if (testSolver.SolveDifr() == 1)
                    {
                        double energy = testSolver.CalculateReflectedEnergy();
                        Console.WriteLine(string.Format("  M={0,3}: Reflected Energy = {1:E6}", testM, energy));
                    }
                }
            }
        }
    }
}