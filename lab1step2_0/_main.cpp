#define _USE_MATH_DEFINES // M_PI
#include <iostream>
#include <cmath>
#include <locale>
#include <fstream>
#include <string>
#include "SparseMatrix.h"

// Модуль xlsxwriter требуется установить в корневую папку Python, а не в виртуальную среду.
//Пока эта проблема не решена. Она возникает только при запуске из решения c++. В случае автономного запуска скрипта .py такой проблемы не возникает.

// Применение итерационного метода сопряженных градиентов (СГ) позволяет решить задачу нахождения точного решения разностной схемы (РС), которое является
//приближенным решением исходной задачи (задача Дирихле для уравнения Пуассона). Метод СГ должен возвращать вектор V, элементы которого соответствтуют координатам
//на прямоугольнике в плоскости XY: V(x,y). Вектор V не содержит граничные значения, так как они известны из постановки исходной задачи: V(0, 0),...,V(n, m),...,V(0, m),...,V(n, 0).
// Таким образом можно будет построить 3-х мерный график, где оси Z соответствует значения вектора V. 

//mu1(y) - левая грань сетки.
//mu2(y) - правая грань сетки.
//mu3(x) - нижняя грань сетки.
//mu4(x) - верхняя грань сетки.

double mu1m(double y) // m - main - основная задача
{
  return sin(M_PI * y);
}

double mu2m(double y)
{
  return sin(M_PI * y);
}

double mu3m(double x)
{
  return x - x * x;
}

double mu4m(double x)
{
  return x - x * x;
}

double mu1t(double y) // t - test - тестовая задача
{
  return 1.;
}

double mu2t(double y)
{
  return exp(sin(M_PI * y)* sin(M_PI * y));
}

double mu3t(double x)
{
  return 1.;
}

double mu4t(double x)
{
  return exp(sin(M_PI * x)* sin(M_PI * x));
}

double func_test(double x, double y)
{
  //return (x * x + y * y) * exp(sin(M_PI * x * y) * sin(M_PI * x * y)) * M_PI * M_PI * (2. * cos(2. * M_PI * x * y) + sin(2. * M_PI * x * y) * sin(2. * M_PI * x * y));
  return (-1.)*(x * x + y * y) * exp(sin(M_PI * x * y) * sin(M_PI * x * y)) * M_PI * M_PI * 2. * (cos(M_PI * x * y) * cos(M_PI * x * y) + 2*sin(M_PI * x * y)*sin(M_PI * x * y)*cos(M_PI * x * y) * cos(M_PI * x * y) - sin(M_PI * x * y)*sin(M_PI * x * y));
}

double func_main(double x, double y)
{
  return  sin(M_PI * x * y) * sin(M_PI * x * y);
}

double solution_test(double x, double y)
{
  return exp(sin(M_PI * x * y) * sin(M_PI * x * y));
}

// Скалярное произведение двух векторов.
double dotProduct(std::vector<double>& a, std::vector<double>& b)
{
  if (a.size() == b.size())
  {
    double res = 0.;
    for (int i = 0; i < a.size(); i++)
      res += a[i] * b[i];
    return res;
  }
  else
    throw(-1);
}

// Норма разности двух векторов. Евклидова Норма.
double normDiffVec2(std::vector<double> &a, std::vector<double> &b) 
{
  if (a.size() == b.size())
  {
    double norm = 0.;
    std::vector<double> tmp(a.size());
    for (int i = 0; i < a.size(); i++)
      tmp[i] = a[i] - b[i];
    norm = sqrt(dotProduct(tmp, tmp));
    return norm;
  }
  else
    throw(-1);
}

// Норма разности двух векторов. l1 норма.
double normDiffVec1(std::vector<double>& a, std::vector<double>& b) 
{
  if (a.size() == b.size())
  {
    double norm = 0.;
    std::vector<double> tmp(a.size());
    for (int i = 0; i < a.size(); i++)
    {
      tmp[i] = a[i] - b[i];
      norm += abs(tmp[i]);
    }
    
    return norm;
  }
  else
    throw(-1);
}

// Норма разности двух векторов. Чебышёва норма.
double normDiffVecInfty(std::vector<double>& a, std::vector<double>& b)
{
  if (a.size() == b.size())
  {
    double norm = 0.;
    std::vector<double> tmp(a.size());
    for (int i = 0; i < a.size(); i++)
    {
      tmp[i] = a[i] - b[i];
      if(abs(tmp[i]) > norm)
        norm = abs(tmp[i]);
    }

    return norm;
  }
  else
    throw(-1);
}

// Норма l1.
double norm1(std::vector<double>& a)
{
  double norm = 0.;
  for (int i = 0; i < a.size(); i++)
    norm += abs(a[i]);

  return norm;
}

// Евклидова норма.
double norm2(std::vector<double>& a)
{
  double norm = 0;
  
  for (int i = 0; i < a.size(); i++)
    norm = sqrt(dotProduct(a, a));
  return norm;
}

// Норма Чебышёва.
double normInfty(std::vector<double>& a)
{
  double norm = 0.;
  for (int i = 0; i < a.size(); i++)
  {
    if (abs(a[i]) > norm)
      norm = abs(a[i]);
  }

  return norm;
}


// Функция вычисления вектора правой части для основной задачи.
void calcFm(std::vector<double>& F, std::vector<double>& X, std::vector<double>& Y, int& n, int& m, double& coefHor, double& coefVer) // [MAIN]
{
  // заполним вектор F:
  // движение по сетке слева направо (по оси х) - снизу вверх (по оси у)

  int stepX = 1;
  int stepY = 1;
  int j = 0;

  for (int block = 0; block < m - 1; block++)
  {
    if (block == 0) // блок первый
    {
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu3m(X[stepX]) * coefVer - mu1m(Y[stepY]) * coefHor; // первое значение первого блока
      stepX++; // движемся вдоль оси х
      j++;
      while (stepX < n - 1) // промежуточные значения первого блока
      {
        F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu3m(X[stepX])*coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu3m(X[stepX]) * coefVer - mu2m(Y[stepY]) * coefHor; // последнее значения первого блока
      stepY++;
      j++;
      continue;
    }
    if (block == m - 2) // блок последний
    {
      stepX = 1;
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu4m(X[stepX]) * coefVer - mu1m(Y[stepY]) * coefHor; // первое значение первого блока
      stepX++; // движемся вдоль оси х
      j++;
      while (stepX < n - 1) // промежуточные значения первого блока
      {
        F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu4m(X[stepX])*coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu4m(X[stepX]) * coefVer - mu2m(Y[stepY]) * coefHor; // последнее значения первого блока
    }
    else // остальные блоки
    {
      stepX = 1;
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu1m(Y[stepY]) * coefHor;
      stepX++;
      j++;
      while (stepX < n - 1)
      {
        F[j] = (-1.) * (func_main(X[stepX], Y[stepY]));
        stepX++;
        j++;
      }
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu2m(Y[stepY]) * coefHor;
      stepY++;
      j++;
    }
  } 

  std::cout << "Вектор Fm заполнен." << std::endl;
}

// Функция вычисления вектора правой части для тестовой задачи.
void calcFt(std::vector<double>& F, std::vector<double>& X, std::vector<double>& Y, int& n, int& m, double& coefHor, double& coefVer) // [MAIN]
{
  // заполним вектор F:
  // движение по сетке слева направо (по оси х) - снизу вверх (по оси у)

  int stepX = 1;
  int stepY = 1;
  int j = 0;

  for (int block = 0; block < m - 1; block++)
  {
    if (block == 0) // блок первый
    {
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu3t(X[stepX]) * coefVer - mu1t(Y[stepY]) * coefHor; // первое значение первого блока
      stepX++; // движемся вдоль оси х
      j++;
      while (stepX < n - 1) // промежуточные значения первого блока
      {
        F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu3t(X[stepX]) * coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu3t(X[stepX]) * coefVer - mu2t(Y[stepY]) * coefHor; // последнее значения первого блока
      stepY++;
      j++;
      continue;
    }
    if (block == m - 2) // блок последний
    {
      stepX = 1;
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu4t(X[stepX]) * coefVer - mu1t(Y[stepY]) * coefHor; // первое значение первого блока
      stepX++; // движемся вдоль оси х
      j++;
      while (stepX < n - 1) // промежуточные значения первого блока
      {
        F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu4t(X[stepX]) * coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu4t(X[stepX]) * coefVer - mu2t(Y[stepY]) * coefHor; // последнее значения первого блока
    }
    else // остальные блоки
    {
      stepX = 1;
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu1t(Y[stepY]) * coefHor;
      stepX++;
      j++;
      while (stepX < n - 1)
      {
        F[j] = (-1.) * (func_test(X[stepX], Y[stepY]));
        stepX++;
        j++;
      }
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu2t(Y[stepY]) * coefHor;
      stepY++;
      j++;
    }
  }

  std::cout << "Вектор Ft заполнен." << std::endl;
}

// Функция заполнения матрицы А.
void calcA(SparseMatrix &A, double h, double k, int n, int m, double coefA, double coefHor, double coefVer)
{
  // Заполним матрицу A:
  // if n > 2 && m > 2; R - row; C - column
  int stepGR = 0, stepGC = 0; // G - главная диагональ
  int stepVR = 0, stepVC = 0; // V - верхняя диагональ
  int stepNR = 0, stepNC = 0; // N - нижняя диагональ
  for (int block = 0; block < m - 1; block++)
  {
    if (block == 0)
    {
      // первый блок
      for (int i = 0; i < n - 1; i++)
      {
        if (i == 0)
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC + 1, coefHor);
        }
        else if (i == n - 2)
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC - 1, coefHor);
        }
        else
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC - 1, coefHor);
          A.set(stepGR, stepGC + 1, coefHor);
        }
        stepGR++;
        stepGC++;
        stepVC++;
        stepNR++;
      }
      // правый блок
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepVR, stepVC, coefVer);
        stepVR++;
        stepVC++;
      }
      continue;
    }
    if (block == m - 2)
    {
      // последний блок
      for (int i = 0; i < n - 1; i++)
      {
        if (i == 0)
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC + 1, coefHor);
        }
        else if (i == n - 2)
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC - 1, coefHor);
        }
        else
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC - 1, coefHor);
          A.set(stepGR, stepGC + 1, coefHor);
        }
        stepGR++;
        stepGC++;
      }
      // левый блок
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepNR, stepNC, coefVer);
        stepNR++;
        stepNC++;
      }
    }
    else
    {
      // блок на главной диагонали
      for (int i = 0; i < n - 1; i++)
      {
        if (i == 0)
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC + 1, coefHor);
        }
        else if (i == n - 2)
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC - 1, coefHor);
        }
        else
        {
          A.set(stepGR, stepGC, coefA);
          A.set(stepGR, stepGC - 1, coefHor);
          A.set(stepGR, stepGC + 1, coefHor);
        }
        stepGR++;
        stepGC++;
      }
      // блок на верхней диагонали
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepVR, stepVC, coefVer);
        stepVR++;
        stepVC++;
      }
      // блок на нижней диагонали
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepNR, stepNC, coefVer);
        stepNR++;
        stepNC++;
      }
    }
  }

  std::cout << "Матрица A заполнена" << std::endl;
}

// Матрица A называется положительно определенной (A > 0), если для любого h != 0, h in R: (Ah, h)>0.
bool positiveDefiniteABool(SparseMatrix& A, std::vector<double>& h)
{
  if (A.getCols() == h.size())
  {
    std::vector<double> tmp(h.size());
    tmp = A.dot(h);
    if (dotProduct(tmp, h) > 0.)
      return 1;
    else
      return 0;
  }
  else
    throw(-1);
}

// Метод сопряженных градиентов (СГ).
std::vector<double> conjugateGradient(SparseMatrix& A, std::vector<double>& F, double& epsMethod, int& Nmax, int& Nreal, double& epsRealMethod, double& rNorm, double& rNorm0)
{
  std::cout << "Метод СГ начал работу." << std::endl;

  // Нулевой вектор.
  std::vector<double> null(A.getRows());

  int steps = 0;
  int size = A.getRows();
  std::vector<double> Vres(size);
  std::vector<double> hprev(size); // hprev - направление на предыдущем шаге;
  std::vector<double> h(size); // h - текущее направление ;
  std::vector<double> rprev(size); // rprev - невязка на предыдущем шаге;
  std::vector<double> r(size); // r - текущая невязка;
  std::vector<double> tmp0(size);
  std::vector<double> tmp1(size);

  double a = 0.; // alpha - аргумент вершины параболы (задача минимизации);
  double b = 0.; // beta - следует из условия сопряженности h и hprev относительно A;
  double norm = 0.;

  // Первый шаг метода.
  tmp0 = A.dot(null); // A*null;
  for (int i = 0; i < size; i++)
  {
    r[i] = (-1.) * ( tmp0[i] - F[i]);
    h[i] = (-1.) * r[i];
  }

  tmp1 = A.dot(h);
  a = ((-1.) * dotProduct(h, h)) / (dotProduct(tmp1, h));
  for (int i = 0; i < size; i++)
    Vres[i] = null[i] + a * h[i];

  steps++;

  std::cout << "Номер шага: " << steps << std::endl;

  norm = normDiffVecInfty(Vres, null);
  std::cout << "Норма разности значений на текущем и предыдущем шаге: " << norm << std::endl;

  //std::cout << "Невязка: ";
  //for(int i = 0; i < size; i++)
  //  std::cout << r[i] << '|';
  //std::cout << std::endl;

  rNorm0 = normInfty(r);

  // Критерий отыскания точного решения.
  if (r == null)
  {
    rNorm = normInfty(r);
    epsRealMethod = norm;
    Nreal = steps;
    return Vres;
  }
  // Критеорий остановки по точности.
  if (norm < epsMethod)
  {
    rNorm = normInfty(r);
    epsRealMethod = norm;
    Nreal = steps;
    return Vres;
  }
  // Критерий остановки по числу шагов
  if (steps == Nmax)
  {
    rNorm = normInfty(r);
    epsRealMethod = norm;
    Nreal = steps;
    return Vres;
  }

  for (int i = 0; i < size; i++)
  {
    rprev[i] = r[i];
    hprev[i] = h[i];
  }

  // Второй шаг и далее.
  // Метод СГ должен найти точное решение СЛАУ за n шагов (Свойство 3).
  // null теперь является хранилищем предыдущего приближенного значения (для критерия остановки по точности).
  for (int s = 1; s < size; s++)
  {
    tmp0 = A.dot(Vres);
    for (int i = 0; i < size; i++)
      r[i] = tmp0[i] - F[i];

    //std::cout << "Невязка: ";
    //for (int i = 0; i < size; i++)
    //  std::cout << r[i] << '|';
    //std::cout << std::endl;

    b = dotProduct(tmp1, r) / dotProduct(tmp1, hprev);

    for (int i = 0; i < size; i++)
      h[i] = (-1.) * r[i] + b * hprev[i];

    //// Проверка: h и hprev должны быть сопряженными относительно A.
    //if (dotProduct(tmp1, hprev) * b - dotProduct(tmp1, r) != 0)
    //  throw(-1);

    tmp1 = A.dot(h);
    a = (-1.) * dotProduct(r, h) / dotProduct(tmp1, h);

    for (int i = 0; i < size; i++)
    {
      null[i] = Vres[i];
      Vres[i] = Vres[i] + a * h[i];
    }

    // Если на текущем шаге невязка равна 0, то текущее приближение является точным решением СЛАУ.

    steps++;
    std::cout << "Номер шага: " << steps << std::endl;

    norm = normDiffVecInfty(Vres, null);
    std::cout << "Норма разности значений на текущем и предыдущем шаге: " << norm << std::endl;

    // Критерий отыскания точного решения.
    if (r == null)
    {
      rNorm = normInfty(r);
      epsRealMethod = norm;
      Nreal = steps;
      return Vres;
    }
    // Критеорий остановки по точности.
    if (normDiffVecInfty(Vres, null) < epsMethod)
    {
      rNorm = normInfty(r);
      epsRealMethod = norm;
      Nreal = steps;
      return Vres;
    }
    // Критерий остановки по числу шагов
    if (steps == Nmax)
    {
      rNorm = normInfty(r);
      epsRealMethod = norm;
      Nreal = steps;
      return Vres;
    }

    for (int i = 0; i < size; i++)
    {
      rprev[i] = r[i];
      hprev[i] = h[i];
    }
  }

  rNorm = normInfty(r);
  epsRealMethod = norm;
  Nreal = steps;
  return Vres;
}

// Погрешность решения тестовой задачи
double error(std::vector<std::vector<double>> sol1, std::vector<std::vector<double>> sol2,int n, int m)
{
  int I = 0, J = 0;
  if (sol1.size()*sol1[0].size() == sol2.size()*sol2[0].size())
  {
    double eps1 = 0.;

    for (int j = 1; j < m + 1; j++)
      for (int i = 1; i < n + 1; i++)
        if (abs(sol1[i][j] - sol2[i][j]) > eps1)
        {
          eps1 = abs(sol1[i][j] - sol2[i][j]);
          I = i;
          J = j;
        }
    return eps1;
  }
  else
    throw(-1);
}

double error2(std::vector<std::vector<double>> sol1, std::vector<std::vector<double>> sol2, int n, int m)
{
  int I = 0, J = 0;
  double eps2 = 0.;

  for (int j = 0; j < m + 1; j++)
    for (int i = 0; i < n + 1; i++)
      if (abs(sol1[i][j] - sol2[2 * i][2 * j]) > eps2)
      {
        eps2 = abs(sol1[i][j] - sol2[2 * i][2 * j]);
        I = i;
        J = j;
      }
  return eps2;
}

// Итоговое заполнение решений основной и тестовой задач
void solution(std::vector<std::vector<double>>& Utest, std::vector<std::vector<double>>& Vtest, std::vector<double> X, std::vector<double> Y, int n, int m, double h, double k, std::vector<double>& V)
{
  // Подсчет на границах.
  double stepY = 0.;
  double stepX = 0.;
  for (int i = 0; i < X.size(); i++)
  {
    Utest[i][0] = mu3t(stepX); // движемся по оси X при y = 0;
    Vtest[i][0] = mu3t(stepX);

    Utest[i][m] = mu4t(stepX); // движемся по оси X при y = 1;
    Vtest[i][m] = mu4t(stepX);

    stepX += h;
  }
  for (int j = 0; j < Y.size(); j++)
  {
    Utest[0][j] = mu1t(stepY); // движемся по оси Y при x = 0;
    Vtest[0][j] = mu1t(stepY);

    Utest[n][j] = mu2t(stepY); // движемся по оси Y при x = 1;
    Vtest[n][j] = mu2t(stepY);

    stepY += k;
  }

  // Точное решение тестовой задачи.
  stepY = k;
  for (int j = 1; j < m; j++)
  {
    stepX = h;
    for (int i = 1; i < n; i++)
    {
      Utest[i][j] = solution_test(stepX, stepY);
      stepX += h;
    }
    stepY += k;
  }

  // Точное решение РС тестовой задачи
  int stepV = 0;
  for (int j = 1; j < m; j++)
    for (int i = 1; i < n; i++)
    {
      Vtest[i][j] = V[stepV];
      stepV++;
    }
}

void solution(std::vector<std::vector<double>>& Vmain, std::vector<double> X, std::vector<double> Y, int n, int m, double h, double k, std::vector<double>& V)
{
  // Подсчет на границах.
  double stepY = 0.;
  double stepX = 0.;
  for (int i = 0; i < X.size(); i++)
  {
    Vmain[i][0] = mu3m(stepX); // движемся по оси X при y = 0;
    Vmain[i][m] = mu4m(stepX); // движемся по оси X при y = 1; 

    stepX += h;
  }
  for (int j = 0; j < Y.size(); j++)
  {
    Vmain[0][j] = mu1m(stepY); // движемся по оси Y при x = 0;
    Vmain[n][j] = mu2m(stepY); // движемся по оси Y при x = 1;

    stepY += k;
  }

  // Точное решение РС тестовой задачи
  int stepV = 0;
  for (int j = 1; j < m; j++)
    for (int i = 1; i < n; i++)
    {
      Vmain[i][j] = V[stepV];
      stepV++;
    }
}

std::vector<double> maxDeviation(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b, std::vector<double>& X, std::vector<double>& Y, int n, int m)
{
  std::vector<double> xy(2);
  double dev = 0.;
  for(int j = 1; j < m; j++)
    for (int i = 1; i < n; i++)
      if (abs(a[i][j] - b[i][j]) > dev)
      {
        dev = abs(a[i][j] - b[i][j]);
        xy[0] = i;
        xy[1] = j;
      }
  return xy;
}

std::vector<double> maxDeviation2(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b, std::vector<double>& X, std::vector<double>& Y, int n, int m)
{
  std::vector<double> xy(2);
  double dev = 0.;
  for (int j = 1; j < m; j++)
    for (int i = 1; i < n; i++)
      if (abs(a[i][j] - b[2*i][2*j]) > dev)
      {
        dev = abs(a[i][j] - b[2*i][2*j]);
        xy[0] = i;
        xy[1] = j;
      }
  return xy;
}


int main()
{
  std::setlocale(LC_ALL, "RUS");

  int n = 0; // число разбиений по x;
  int m = 0; // число разбиений по y;
  int Nmax = 0; // число шагов;
  int NrealTest = 0;
  int NrealMain = 0;
  int NrealMain2 = 0;
  double eps = 0.5 * (1 / pow(10.0, 6.0)); // погрешность решения тестовой задачи;
  double eps2 = 0.;
  double epsRealMethodTest = 0.;
  double epsRealMethodMain = 0.;
  double epsRealMethodMain2 = 0.;
  double rNormTest = 0.; // Норма невязки; r - residual (невязка).
  double rNormMain = 0.;
  double rNormMain2 = 0.;
  double rNorm0Test = 0.;
  double rNorm0Main = 0.;
  double rNorm0Main2 = 0.;
  std::vector<double> xyTest;
  std::vector<double> xyMain;
  int deg = 1; // точность (для критерия остановки по точности)

  std::cout << "Введите число разбиений по оси x." << '\n';
  std::cin >> n;
  std::cout << "Введите число разбиений по оси y." << '\n';
  std::cin >> m;
  std::cout << "Введите число шагов метода." << '\n';
  std::cin >> Nmax;
  std::cout << "Критерий остановки по точности 0.5*10^(-x) (введите целое x)." << '\n';
  std::cin >> deg;

  double epsMethod = 0.5 * pow(10, (-1) * deg);

  double a = 0., b = 1., c = 0., d = 1.;

  double h = (b - a) / (double)n; // шаг по x;
  double k = (d - c) / (double)m; // шаг по y;

  std::vector<double> X(n + 1); // узлы по оси x;
  std::vector<double> Y(m + 1); // узлы по оси y;

  // заполним векторы X и Y:
  X[0] = 0.;
  Y[0] = 0.;
  for (int i = 1; i < n + 1; i++)
    X[i] = X[i - 1] + h;
  for (int j = 1; j < m + 1; j++)
    Y[j] = Y[j - 1] + k;

  double coefHor = 1. / (h * h);
  double coefVer = 1. / (k * k); // коэффициенты: A, 1/h^2 и 1/k^2 (см. теорию)
  double coefA = (-2.) * (coefHor + coefVer);

  int rows = (n - 1) * (m - 1), cols = (n - 1) * (m - 1); // Число строк и столбцов матрицы A.
  SparseMatrix A(rows, cols);
  calcA(A, h, k, n, m, coefA, coefHor, coefVer);

  std::vector<double> Fm((n - 1) * (m - 1)); // вектор известных значений СЛАУ (правая часть) [MAIN]
  std::vector<double> Ft((n - 1) * (m - 1)); // вектор известных значений СЛАУ (правая часть) [TEST]

  calcFt(Ft, X, Y, n, m, coefHor, coefVer);
  calcFm(Fm, X, Y, n, m, coefHor, coefVer);

  if (positiveDefiniteABool(A, Fm) == 1)
  {
    // Матрица A > 0, можем применять метод СГ.
  }
  else
  {
    // Поменять знак левой и правой части.
    A.ChangeSign();
    for (int i = 0; i < A.getRows(); i++)
    {
      Fm[i] = (-1.) * Fm[i];
      Ft[i] = (-1.) * Ft[i];
    }
  }

  //// ТЕСТОВЫЕ ДАННЫЕ ДЛЯ метода СГ.
  //// Начальное приближение
  //std::vector<double> nullTEST(2);
  //nullTEST[0] = 0.;
  //nullTEST[1] = 1.;

  //SparseMatrix ATEST(2, 2);
  //ATEST.set(0, 0, 4.);
  //ATEST.set(1, 0, 1.);
  //ATEST.set(0, 1, 1.);
  //ATEST.set(1, 1, 10.);
  //std::vector<double> B(2);
  //B[0] = 9.;
  //B[1] = 12.;

  //conjugateGradient(ATEST, B, nullTEST, 0, epsMethod, 1000);

  //// Вывод матрицы A.
  //for (int i = 0; i < rows; i++)
  //{
  //  std::cout << '\n';
  //  for (int j = 0; j < cols; j++)
  //    std::cout << A.get(i, j) << '|';
  //}

  std::vector<double> V(A.getRows()); 
  
  std::vector<std::vector<double>> Utest((n + 1), std::vector<double>(m + 1)); // Точное решение тестовой задачи.
  std::vector<std::vector<double>> Vtest((n + 1), std::vector<double>(m + 1)); // Численное решение тестовой задачи. 
  std::vector<std::vector<double>> Vmain((n + 1), std::vector<double>(m + 1)); // Численное решение основной задачи.
  std::vector<std::vector<double>> Vmain2((2 * n + 1), std::vector<double>(2 * m + 1)); // Численное решение основной задачи с удвоенной сеткой.
  std::vector<std::vector<double>> V0((n + 1), std::vector<double>(m + 1)); // Начальное приближение
  std::vector<std::vector<double>> UdV((n + 1), std::vector<double>(m + 1)); // Utest difference Vtest
  std::vector<std::vector<double>> VdV2((n + 1), std::vector<double>(m + 1)); // Vmain difference V2main

  // Активация виртуальной среды Python.
  //system("start env\\Scripts\\activate.bat"); Не работает.
  system("cls");
  int flag = 1; // Флаг для выхода.
  int flagTask = 0; // 1 - Решена тестовая задача. 2 - Решена основная задача.
  int var;
  do
  {
    std::cout << "Что вы хотите сделать? Введите соответствующий пункт меню" << '\n';
    std::cout << "1. Решить тестовую задачу" << '\n';
    std::cout << "2. Решить основную задачу" << '\n';
    std::cout << "3. Постановка задачи" << '\n';
    std::cout << "4. Справка" << '\n';
    std::cout << "5. Таблица" << '\n';
    std::cout << "6. График" << '\n';
    std::cout << "7. Перезадать разбиение, критерии остановки по точности и числу шагов." << '\n';
    std::cout << "8. Выход" << '\n';
    std::cin >> var;
    switch (var)
    {
    case 1:
    {
      V = conjugateGradient(A, Ft, epsMethod, Nmax, NrealTest, epsRealMethodTest, rNormTest, rNorm0Test);
      solution(Utest, Vtest, X, Y, n, m, h, k, V);

      // w - write.
// Создание файла в формате csv для записи значений.
// Объект класса ofstream нужен для записи данных в файл. ifstream - для чтения. 
// Создаем таблицу для численного решения тестовой задачи.
      std::string tableVtest = "tableV.csv"; // Наименование файла с таблицей. 
      std::ofstream wDataVt; // Поток для записи. 
      wDataVt.open(tableVtest); // Открываем файл для записи.
      // Первая строка. Имена координат по оси х: x0, x1, ..., xn 
      wDataVt << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataVt << X[i];
        if (i < n)
          wDataVt << ","; // В конце строки .csv не должно быть запятой. Контролируем этот момент.
      }
      wDataVt << std::endl;
      wDataVt << "yj" << "," << "j / i" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataVt << i;
        if (i < n)
          wDataVt << ",";
      }
      wDataVt << std::endl;
      for (int j = 0; j < m + 1; j++)
      {
        wDataVt << Y[j] << "," << j << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataVt << Vtest[i][j];
          if (i < n)
            wDataVt << ",";
        }
        wDataVt << std::endl;
      }
      system("python csv_to_xlsx_V.py");
      wDataVt.close();

      // Создаем таблицу для точного решения тестовой задачи.
      std::string tableUtest = "tableUtest.csv";
      std::ofstream wDataUt;
      wDataUt.open(tableUtest);
      wDataUt << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataUt << X[i];
        if (i < n)
          wDataUt << ","; // В конце строки .csv не должно быть запятой. Контролируем этот момент.
      }
      wDataUt << std::endl;
      wDataUt << "yj" << "," << "j / i" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataUt << i;
        if (i < n)
          wDataUt << ",";
      }
      wDataUt << std::endl;
      for (int j = 0; j < m + 1; j++)
      {
        wDataUt << Y[j] << "," << j << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataUt << Utest[i][j];
          if (i < n)
            wDataUt << ",";
        }
        wDataUt << std::endl;
      }
      system("python csv_to_xlsx_Utest.py");
      wDataUt.close();

      // Заполнение разности точного и численного решения.
      for(int j = 0; j < m+1; j++)
        for (int i = 0; i < n + 1; i++)
          UdV[i][j] = Utest[i][j] - Vtest[i][j];

      // Запись в .csv и .xlsx разности точного и численного решения.
      std::string tableUdV = "tableUdV.csv";
      std::ofstream wDataUdV;
      wDataUdV.open(tableUdV);
      wDataUdV << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataUdV << X[i];
        if (i < n)
          wDataUdV << ","; // В конце строки .csv не должно быть запятой. Контролируем этот момент.
      }
      wDataUdV << std::endl;
      wDataUdV << "yj" << "," << "j / i" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataUdV << i;
        if (i < n)
          wDataUdV << ",";
      }
      wDataUdV << std::endl;
      for (int j = 0; j < m + 1; j++)
      {
        wDataUdV << Y[j] << "," << j << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataUdV << UdV[i][j];
          if (i < n)
            wDataUdV << ",";
        }
        wDataUdV << std::endl;
      }
      system("python csv_to_xlsx_UdV.py");
      wDataUdV.close();
 
      flagTask = 1;

      //// Вывод Utest.
      //std::cout << '\n';
      //for (int j = 1; j < m; j++)
      //{
      //  std::cout << '\n';
      //  for (int i = 1; i < n; i++)
      //    std::cout << Utest[i][j] << '|'; // Выводим в обратном порядке относительно оси X, относительно нашего представления на бумаге.
      //}
      //// Вывод Vtest.
      //std::cout << '\n';
      //for (int j = 1; j < m; j++)
      //{
      //  std::cout << '\n';
      //  for (int i = 1; i < n; i++)
      //    std::cout << Vtest[i][j] << '|'; // Выводим в обратном порядке относительно оси X, относительно нашего представления на бумаге.
      //}
      //std::cout << '\n';

      //// Сравнение Utest и Vtest поэлементно во внутренних узлах.
      //std::cout << '\n';
      //for (int j = 1; j < m; j++)
      //{
      //  std::cout << '\n';
      //  for (int i = 1; i < n; i++)
      //    std::cout << Utest[i][j] << '|' << Vtest[i][j] << '\n'; // Выводим в обратном порядке относительно оси X, относительно нашего представления на бумаге.
      //}
      //std::cout << '\n';

      system("cls");

      break;
    }
    case 2:
    {
      V = conjugateGradient(A, Fm, epsMethod, Nmax, NrealMain, epsRealMethodMain, rNormMain, rNorm0Main);
      solution(Vmain, X, Y, n, m, h, k, V);

      // Решение для задачи c разбиением 2*n и 2*m.
      int n2 = 2 * n; // 2n
      int m2 = 2 * m; // 2m

      double h2 = (b - a) / (double)n2; 
      double k2 = (d - c) / (double)m2;

      std::vector<double> X2(n2 + 1);
      std::vector<double> Y2(m2 + 1);
      X2[0] = 0.;
      Y2[0] = 0.;
      for (int i = 1; i < n2 + 1; i++)
        X2[i] = X2[i - 1] + h2;
      for (int j = 1; j < m2 + 1; j++)
        Y2[j] = Y2[j - 1] + k2;

      double coefHor2 = 1. / (h2 * h2);
      double coefVer2 = 1. / (k2 * k2);
      double coefA2 = (-2.) * (coefHor2 + coefVer2);

      std::vector<double> Fm2((n2 - 1) * (m2 - 1)); // вектор известных значений СЛАУ (правая часть) [MAIN]

      int rows2 = (n2 - 1) * (m2 - 1), cols2 = (n2 - 1) * (m2 - 1); // Число строк и столбцов матрицы A.

      SparseMatrix A2(rows2, cols2);

      calcA(A2, h2, k2, n2, m2, coefA2, coefHor2, coefVer2);

      std::vector<double> V2(A2.getRows());

      std::vector<std::vector<double>> V2main((n2 + 1), std::vector<double>(m2 + 1)); // Численное решение основной задачи.

      calcFm(Fm2, X2, Y2, n2, m2, coefHor2, coefVer2);

      V2 = conjugateGradient(A2, Fm2, epsMethod, Nmax, NrealMain2, epsRealMethodMain2, rNormMain2, rNorm0Main2);

      solution(V2main, X2, Y2, n2, m2, h2, k2, V2);

      eps2 = error2(Vmain, V2main, n, m);

      xyMain = maxDeviation2(Vmain, V2main, X, Y, n, m);

      std::string tableV = "tableV.csv";
      std::ofstream wDataV;
      wDataV.open(tableV);
      wDataV << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataV << X[i];
        if (i < n)
          wDataV << ",";
      }
      wDataV << std::endl;
      wDataV << "yj" << "," << "j / i" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataV << i;
        if (i < n)
          wDataV << ",";
      }
      wDataV << std::endl;
      for (int j = 0; j < m + 1; j++)
      {
        wDataV << Y[j] << "," << j << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataV << Vmain[i][j];
          if (i < n)
            wDataV << ",";
        }
        wDataV << std::endl;
      }
      system("python csv_to_xlsx_V.py");
      wDataV.close();

      std::string tableV2 = "tableV2.csv";
      std::ofstream wDataV2;
      wDataV2.open(tableV2);
      wDataV2 << "," << "xi" << ",";
      for (int i = 0; i < 2*n + 1; i++)
      {
        wDataV2 << X2[i];
        if (i < 2*n)
          wDataV2 << ","; // В конце строки .csv не должно быть запятой. Контролируем этот момент.
      }
      wDataV2 << std::endl;
      wDataV2 << "yj" << "," << "j / i" << ",";
      for (int i = 0; i < 2*n + 1; i++)
      {
        wDataV2 << i;
        if (i < 2*n)
          wDataV2 << ",";
      }
      wDataV2 << std::endl;
      for (int j = 0; j < 2*m + 1; j++)
      {
        wDataV2 << Y2[j] << "," << j << ",";
        for (int i = 0; i < 2*n + 1; i++)
        {
          wDataV2 << V2main[i][j];
          if (i < 2*n)
            wDataV2 << ",";
        }
        wDataV2 << std::endl;
      }
      system("python csv_to_xlsx_V2.py");
      wDataV2.close();

      // Заполнение разности точного и численного решения.
      for (int j = 0; j < m + 1; j++)
        for (int i = 0; i < n + 1; i++)
          VdV2[i][j] = Vmain[i][j] - V2main[2*i][2*j];

      // Запись в .csv и .xlsx разности точного и численного решения.
      std::string tableVdV2 = "tableVdV2.csv";
      std::ofstream wDataVdV2;
      wDataVdV2.open(tableVdV2);
      wDataVdV2 << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataVdV2 << X[i];
        if (i < n)
          wDataVdV2 << ","; // В конце строки .csv не должно быть запятой. Контролируем этот момент.
      }
      wDataVdV2 << std::endl;
      wDataVdV2 << "yj" << "," << "j / i" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataVdV2 << i;
        if (i < n)
          wDataVdV2 << ",";
      }
      wDataVdV2 << std::endl;
      for (int j = 0; j < m + 1; j++)
      {
        wDataVdV2 << Y[j] << "," << j << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataVdV2 << VdV2[i][j];
          if (i < n)
            wDataVdV2 << ",";
        }
        wDataVdV2 << std::endl;
      }
      system("python csv_to_xlsx_VdV2.py");
      wDataVdV2.close();

      V2main.clear();

      flagTask = 2;

      system("cls");

      break;
    }
    case 3:
    {
      system("start Task.png");
      system("cls");
      break;
    }
    case 4:
    {
      system("cls");
      if (flagTask == 0)
      {
        std::cout << "Создана сетка с числом разбиений по x n = " << n << " и числом разбиений по y m = " << m << '.' << std::endl;
        std::cout << "Выберете, какую задачу решить: тестовую или основную.";
      }
      else if (flagTask == 1)
      {
        xyTest = maxDeviation(Utest, Vtest, X, Y, n, m);
        std::cout << "Для решения тестовой задачи использованы сетка с числом разбиений по x n = " << n << " и числом разбиений по y m = " << m << ',' << std::endl;
        std::cout << "метод споряженных градиентов, параметры метода отсутствуют," << std::endl;
        std::cout << "критерии остановки по точности eps_(met) = " << epsMethod << " и по числу итераций Nmax = " << Nmax << '.' << std::endl;
        std::cout << "На решение схемы (СЛАУ) затрачено итераций N = " << NrealTest << " и достигнута точность итерационного метода eps^(N) = " << epsRealMethodTest << '.' << std::endl;
        std::cout << "Схема (СЛАУ) решена с невязкой ||R^(N)|| = " << rNormTest << ", для невязки СЛАУ использована норма Чебышёв ." << std::endl;
        std::cout << "Тестовая задача должна быть решена в погрешностью не более " << eps << "; задача решена с погрешностью eps_1 = " << error(Utest, Vtest, n, m) << '.' << std::endl;
        std::cout << "Максимальное отклонение точного и численного решений наблюдается в узле x = " << X[xyTest[0]] << "; y = " << Y[xyTest[1]] << '.' << std::endl;
        std::cout << "В качестве начального приближения использовано нулевое приближение." << std::endl;
        std::cout << "Невязка СЛАУ на начальном приближении ||R^(0)|| = " << rNorm0Test << ". Для невязки СЛАУ использована норма Чебышёва." << std::endl;
      }
      else if (flagTask == 2)
      {
        std::cout << "Для решения основной задачи использована сетка с числом разбиений по x n = " << n << " и числом разбиений по y m = " << m << ',' << std::endl;
        std::cout << "метод сопряженных градиентов, параметры метода отсутствуют," << std::endl;
        std::cout << "критерии остановки по точности eps_(met) = " << epsMethod << " и по числу итераций Nmax = " << Nmax << '.' << std::endl;
        std::cout << "На решение схемы (СЛАУ) затрачено итераций N = " << NrealMain << " и достигнута точность итерационного метода eps^(N) = " << epsRealMethodMain << '.' << std::endl;
        std::cout << "Схема (СЛАУ) решена с невязкой ||R^(N)|| = " << rNormMain << ", для невязки использована норма Чебышёва ." << std::endl;
        std::cout << "В качестве начального приближения на основной сетке использовано нулевое приближение." << std::endl;
        std::cout << "На основной сетке невязка СЛАУ на начальном приближении ||R^(0)|| = " << rNorm0Main << ". Для невязки СЛАУ использована норма Чебышёва." << std::endl;
        std::cout << '\n';
        std::cout << "Для контроля точности использована сетка с половинным шагом," << std::endl;
        std::cout << "метод сопряженных градиентов, параметры метода отсутствуют," << std::endl;
        std::cout << "критерии остановки по точности eps_(met-2) = " << epsMethod << " и по числу итераций Nmax-2 = " << Nmax << '.' << std::endl;
        std::cout << "На решение задачи (СЛАУ) затрачено итераций N2 = " << NrealMain2 << " и достигнута точность итерационного метода eps^(N2) = " << epsRealMethodMain2 << '.' << std::endl;
        std::cout << "Схема (СЛАУ) на сетке с половинным шагом решена с невязкой ||R^(N2)|| = " << rNormMain2 << ", для невязки СЛАУ использована норма Чебышёва." << std::endl;
        std::cout << "Основная задача должна быть решена с точностью не хуже чем eps = " << eps << "; задача решена с точностью eps_2 = " << eps2 << '.' << std::endl;
        std::cout << "Максимальное отклонение численных решений на основной сетке и сетке с половинным шагом наблюдается в узле x = " << X[xyMain[0]] << "; y = " << Y[xyMain[1]] << '.' << std::endl;
        std::cout << "В качестве начального приближения на сетке с половинным шагом использовано нулевое приближение." << std::endl;
        std::cout << "На сетке с половинным шагом невязка СЛАУ на начальном приближении ||R^(0)|| = " << rNorm0Main2 << ". Для невязки СЛАУ использована норма Чебышёва." << std::endl;
      }

      break;
    }
    case 5:
    {
      if (flagTask == 0)
      {
        system("cls");
        std::cout << "Требуется решить тестовую (1.) или основную задачу (2.)" << std::endl;
      }
      if (flagTask == 1)
      {
        system("start tableUtest.xlsx");
        system("start tableV.xlsx");
        system("start tableUdV.xlsx");
        system("cls");
      }
      if (flagTask == 2)
      {
        system("start tableV.xlsx");
        system("start tableV2.xlsx");
        system("start tableVdV2.xlsx");
        system("cls");
      }

      break;
    }
    case 6:
    {
      if (flagTask == 0)
      {
        system("cls");
        std::cout << "Требуется решить тестовую (1.) или основную задачу (2.)" << std::endl;
      }
      if (flagTask == 1)
      {
        std::string tableV0 = "tableV0.csv";
        std::ofstream wDataV0;
        wDataV0.open(tableV0);
        wDataV0 << "," << "xi" << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataV0 << "x" << i;
          if (i < n)
            wDataV0 << ","; // В конце строки .csv не должно быть запятой. Контролируем этот момент.
        }
        wDataV0 << std::endl;
        wDataV0 << "yj" << "," << "j / i" << ",";
        for (int i = 0; i < n + 1; i++)
        {
          wDataV0 << i;
          if (i < n)
            wDataV0 << ",";
        }
        wDataV0 << std::endl;
        for (int j = 0; j < m + 1; j++)
        {
          wDataV0 << "y" << j << "," << j << ",";
          for (int i = 0; i < n + 1; i++)
          {
            wDataV0 << V0[i][j];
            if (i < n)
              wDataV0 << ",";
          }
          wDataV0 << std::endl;
        }
        wDataV0.close();

        system("python graphicTest2.py");
        system("cls");
      }
      if (flagTask == 2)
      {
        system("python graphicMain2.py");
        //system("cls");
      }

      break;
    }
    case 7:
    {
      system("cls");
      std::cout << "Введите число разбиений по оси x." << '\n';
      std::cin >> n;
      std::cout << "Введите число разбиений по оси y." << '\n';
      std::cin >> m;
      std::cout << "Введите число шагов метода." << '\n';
      std::cin >> Nmax;
      std::cout << "Критерий остановки по точности 0.5*10^(-x) (введите целое x)." << '\n';
      std::cin >> deg;

      X.clear();
      Y.clear();
      Fm.clear();
      Ft.clear();
      V.clear();
      Vtest.clear();
      Vmain.clear();
      Utest.clear();
      V0.clear();
      UdV.clear();

      epsMethod = 0.5 * pow(10, (-1) * deg);
      h = (b - a) / (double)n;
      k = (d - c) / (double)m;
      X.resize(n + 1);
      Y.resize(m + 1);
      X[0] = 0.;
      Y[0] = 0.;
      for (int i = 1; i < n + 1; i++)
        X[i] = X[i - 1] + h;
      for (int j = 1; j < m + 1; j++)
        Y[j] = Y[j - 1] + k;
      coefHor = 1. / (h * h);
      coefVer = 1. / (k * k);
      coefA = (-2.) * (coefHor + coefVer);
      rows = (n - 1) * (m - 1), cols = (n - 1) * (m - 1);
      A.resize(rows, cols);
      calcA(A, h, k, n, m, coefA, coefHor, coefVer);
      Fm.resize((n - 1)* (m - 1));
      Ft.resize((n - 1)* (m - 1));
      calcFt(Ft, X, Y, n, m, coefHor, coefVer);
      calcFm(Fm, X, Y, n, m, coefHor, coefVer);
      //if (positiveDefiniteABool(A, Fm) == 1)
      //{
      //  // Матрица A > 0, можем применять метод СГ.
      //}
      //else
      //{
      //  // Поменять знак левой и правой части.
      //  A.ChangeSign();
      //  for (int i = 0; i < A.getRows(); i++)
      //  {
      //    Fm[i] = (-1.) * Fm[i];
      //    Ft[i] = (-1.) * Ft[i];
      //  }
      //}
      V.resize(A.getRows());
      Utest.resize((n + 1), std::vector<double>(m + 1));
      Vtest.resize((n + 1), std::vector<double>(m + 1));
      Vmain.resize((n + 1), std::vector<double>(m + 1));
      V0.resize((n + 1), std::vector<double>(m + 1));
      UdV.resize((n + 1), std::vector<double>(m + 1));
      system("cls");

      break;
    }
    case 8:
    {
      flag = 0;

      break;
    }
    default:
    {
      std::cout << "Выберете предложенные пункты меню";

      break;
    }
    }
  } while (flag == 1);

  // Деактивация виртуальной среды Python.
  //system("deactivate"); // Не работает.

  return 0;
}