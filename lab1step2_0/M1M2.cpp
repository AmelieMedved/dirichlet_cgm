//// “еорема о сходимости метода. ¬ычисление M1 и M2
//#define _USE_MATH_DEFINES // M_PI
//#include <iostream>
//#include <cmath>
//#include <vector>
//
//double Uxxxx(double x, double y)
//{
//  double pi = M_PI;
//  return  4. * pi * pi * pi * pi * y * y * y * y * exp(sin(pi * x * y) * sin(pi * x * y)) * (sin(pi * x * y) * sin(pi * x * y) * (3. * sin(pi * x * y) * sin(pi * x * y) + 2.) + cos(pi * x * y) * cos(pi * x * y) * cos(pi * x * y) * cos(pi * x * y) * (4. * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) + 12. * sin(pi * x * y) * sin(pi * x * y) + 3.) - 2. * (6. * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) + 11. * sin(pi * x * y) * sin(pi * x * y) + 1.) * cos(pi * x * y) * cos(pi * x * y));
//}
//
//double  Uyyyy(double x, double y)
//{
//  double pi = M_PI;
//  return  4. * pi * pi * pi * pi * x * x * x * x * exp(sin(pi * x * y) * sin(pi * x * y)) * (sin(pi * x * y) * sin(pi * x * y) * (3. * sin(pi * x * y) * sin(pi * x * y) + 2.) + cos(pi * x * y) * cos(pi * x * y) * cos(pi * x * y) * cos(pi * x * y) * (4. * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) + 12. * sin(pi * x * y) * sin(pi * x * y) + 3.) - 2. * (6. * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) * sin(pi * x * y) + 11. * sin(pi * x * y) * sin(pi * x * y) + 1.) * cos(pi * x * y) * cos(pi * x * y));
//}
//
//int main()
//{
//  int n = 1600;
//  int m = 1600;
//  double a = 0.;
//  double b = 1.;
//  double c = 0.;
//  double d = 1.;
//  double h = (b - a) / (double)n;
//  double k = (d - c) / (double)m;
//  std::vector<double> X(n + 1);
//  std::vector<double> Y(n + 1);
//  X[0] = a;
//  Y[0] = c;
//  for (int j = 1; j < m + 1; j++)
//    Y[j] = Y[j - 1] + k;
//  for (int i = 1; i < n + 1; i++)
//    X[i] = X[i - 1] + h;
// 
//  double M1 = 0.;
//  double M2 = 0.;
//
//  double tmp1 = 0., tmp2 = 0.;
//
//  for(int j = 0; j < m + 1; j++)
//    for (int i = 0; i < n + 1; i++)
//    {
//      tmp1 = abs(Uxxxx(X[i], Y[j]));
//      tmp2 = abs(Uyyyy(X[i], Y[j]));
//      if (M1 < tmp1)
//        M1 = tmp1;
//      if (M2 < tmp2)
//        M2 = tmp2;
//    }
//
//  M1 = 1. / 12. * M1; // 5295.71 * 1/12
//  M2 = 1. / 12. * M2; // 5295.71 * 1/12
//
//  // ѕриближенное вычисление минимального и максимального по модулю собственного числа
//
//  double lambdaMin = -(4. / (h * h) * sin(M_PI / (2. * n))* sin(M_PI / (2. * n)) + 4. / (k * k) * sin(M_PI / (2. * m))* sin(M_PI / (2. * m)));
//  lambdaMin = abs(lambdaMin);
//  double lambdaMax = -(4. / (h * h) * sin((M_PI * (n - 1)) / (2. * n))* sin((M_PI * (n - 1)) / (2. * n)) + 4. / (k * k) * sin((M_PI * (m - 1)) / (2. * m))* sin((M_PI * (m - 1)) / (2. * m)));
//  lambdaMax = abs(lambdaMax);
//
//  std::cout << M1 << ' ' << M2 << ' ' << lambdaMin << ' ' << lambdaMax;
//
//  return 0;
//}