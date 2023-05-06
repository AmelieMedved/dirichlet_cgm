#define _USE_MATH_DEFINES // M_PI
#include <iostream>
#include <cmath>
#include <locale>
#include <fstream>
#include <string>
#include "SparseMatrix.h"

// ������ xlsxwriter ��������� ���������� � �������� ����� Python, � �� � ����������� �����.
//���� ��� �������� �� ������. ��� ��������� ������ ��� ������� �� ������� c++. � ������ ����������� ������� ������� .py ����� �������� �� ���������.

// ���������� ������������� ������ ����������� ���������� (��) ��������� ������ ������ ���������� ������� ������� ���������� ����� (��), ������� ��������
//������������ �������� �������� ������ (������ ������� ��� ��������� ��������). ����� �� ������ ���������� ������ V, �������� �������� �������������� �����������
//�� �������������� � ��������� XY: V(x,y). ������ V �� �������� ��������� ��������, ��� ��� ��� �������� �� ���������� �������� ������: V(0, 0),...,V(n, m),...,V(0, m),...,V(n, 0).
// ����� ������� ����� ����� ��������� 3-� ������ ������, ��� ��� Z ������������� �������� ������� V. 

//mu1(y) - ����� ����� �����.
//mu2(y) - ������ ����� �����.
//mu3(x) - ������ ����� �����.
//mu4(x) - ������� ����� �����.

double mu1m(double y) // m - main - �������� ������
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

double mu1t(double y) // t - test - �������� ������
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

// ��������� ������������ ���� ��������.
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

// ����� �������� ���� ��������. ��������� �����.
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

// ����� �������� ���� ��������. l1 �����.
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

// ����� �������� ���� ��������. �������� �����.
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

// ����� l1.
double norm1(std::vector<double>& a)
{
  double norm = 0.;
  for (int i = 0; i < a.size(); i++)
    norm += abs(a[i]);

  return norm;
}

// ��������� �����.
double norm2(std::vector<double>& a)
{
  double norm = 0;
  
  for (int i = 0; i < a.size(); i++)
    norm = sqrt(dotProduct(a, a));
  return norm;
}

// ����� ��������.
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


// ������� ���������� ������� ������ ����� ��� �������� ������.
void calcFm(std::vector<double>& F, std::vector<double>& X, std::vector<double>& Y, int& n, int& m, double& coefHor, double& coefVer) // [MAIN]
{
  // �������� ������ F:
  // �������� �� ����� ����� ������� (�� ��� �) - ����� ����� (�� ��� �)

  int stepX = 1;
  int stepY = 1;
  int j = 0;

  for (int block = 0; block < m - 1; block++)
  {
    if (block == 0) // ���� ������
    {
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu3m(X[stepX]) * coefVer - mu1m(Y[stepY]) * coefHor; // ������ �������� ������� �����
      stepX++; // �������� ����� ��� �
      j++;
      while (stepX < n - 1) // ������������� �������� ������� �����
      {
        F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu3m(X[stepX])*coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu3m(X[stepX]) * coefVer - mu2m(Y[stepY]) * coefHor; // ��������� �������� ������� �����
      stepY++;
      j++;
      continue;
    }
    if (block == m - 2) // ���� ���������
    {
      stepX = 1;
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu4m(X[stepX]) * coefVer - mu1m(Y[stepY]) * coefHor; // ������ �������� ������� �����
      stepX++; // �������� ����� ��� �
      j++;
      while (stepX < n - 1) // ������������� �������� ������� �����
      {
        F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu4m(X[stepX])*coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_main(X[stepX], Y[stepY])) - mu4m(X[stepX]) * coefVer - mu2m(Y[stepY]) * coefHor; // ��������� �������� ������� �����
    }
    else // ��������� �����
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

  std::cout << "������ Fm ��������." << std::endl;
}

// ������� ���������� ������� ������ ����� ��� �������� ������.
void calcFt(std::vector<double>& F, std::vector<double>& X, std::vector<double>& Y, int& n, int& m, double& coefHor, double& coefVer) // [MAIN]
{
  // �������� ������ F:
  // �������� �� ����� ����� ������� (�� ��� �) - ����� ����� (�� ��� �)

  int stepX = 1;
  int stepY = 1;
  int j = 0;

  for (int block = 0; block < m - 1; block++)
  {
    if (block == 0) // ���� ������
    {
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu3t(X[stepX]) * coefVer - mu1t(Y[stepY]) * coefHor; // ������ �������� ������� �����
      stepX++; // �������� ����� ��� �
      j++;
      while (stepX < n - 1) // ������������� �������� ������� �����
      {
        F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu3t(X[stepX]) * coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu3t(X[stepX]) * coefVer - mu2t(Y[stepY]) * coefHor; // ��������� �������� ������� �����
      stepY++;
      j++;
      continue;
    }
    if (block == m - 2) // ���� ���������
    {
      stepX = 1;
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu4t(X[stepX]) * coefVer - mu1t(Y[stepY]) * coefHor; // ������ �������� ������� �����
      stepX++; // �������� ����� ��� �
      j++;
      while (stepX < n - 1) // ������������� �������� ������� �����
      {
        F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu4t(X[stepX]) * coefVer;
        j++;
        stepX++;
      }
      F[j] = (-1.) * (func_test(X[stepX], Y[stepY])) - mu4t(X[stepX]) * coefVer - mu2t(Y[stepY]) * coefHor; // ��������� �������� ������� �����
    }
    else // ��������� �����
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

  std::cout << "������ Ft ��������." << std::endl;
}

// ������� ���������� ������� �.
void calcA(SparseMatrix &A, double h, double k, int n, int m, double coefA, double coefHor, double coefVer)
{
  // �������� ������� A:
  // if n > 2 && m > 2; R - row; C - column
  int stepGR = 0, stepGC = 0; // G - ������� ���������
  int stepVR = 0, stepVC = 0; // V - ������� ���������
  int stepNR = 0, stepNC = 0; // N - ������ ���������
  for (int block = 0; block < m - 1; block++)
  {
    if (block == 0)
    {
      // ������ ����
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
      // ������ ����
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
      // ��������� ����
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
      // ����� ����
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepNR, stepNC, coefVer);
        stepNR++;
        stepNC++;
      }
    }
    else
    {
      // ���� �� ������� ���������
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
      // ���� �� ������� ���������
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepVR, stepVC, coefVer);
        stepVR++;
        stepVC++;
      }
      // ���� �� ������ ���������
      for (int i = 0; i < n - 1; i++)
      {
        A.set(stepNR, stepNC, coefVer);
        stepNR++;
        stepNC++;
      }
    }
  }

  std::cout << "������� A ���������" << std::endl;
}

// ������� A ���������� ������������ ������������ (A > 0), ���� ��� ������ h != 0, h in R: (Ah, h)>0.
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

// ����� ����������� ���������� (��).
std::vector<double> conjugateGradient(SparseMatrix& A, std::vector<double>& F, double& epsMethod, int& Nmax, int& Nreal, double& epsRealMethod, double& rNorm, double& rNorm0)
{
  std::cout << "����� �� ����� ������." << std::endl;

  // ������� ������.
  std::vector<double> null(A.getRows());

  int steps = 0;
  int size = A.getRows();
  std::vector<double> Vres(size);
  std::vector<double> hprev(size); // hprev - ����������� �� ���������� ����;
  std::vector<double> h(size); // h - ������� ����������� ;
  std::vector<double> rprev(size); // rprev - ������� �� ���������� ����;
  std::vector<double> r(size); // r - ������� �������;
  std::vector<double> tmp0(size);
  std::vector<double> tmp1(size);

  double a = 0.; // alpha - �������� ������� �������� (������ �����������);
  double b = 0.; // beta - ������� �� ������� ������������� h � hprev ������������ A;
  double norm = 0.;

  // ������ ��� ������.
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

  std::cout << "����� ����: " << steps << std::endl;

  norm = normDiffVecInfty(Vres, null);
  std::cout << "����� �������� �������� �� ������� � ���������� ����: " << norm << std::endl;

  //std::cout << "�������: ";
  //for(int i = 0; i < size; i++)
  //  std::cout << r[i] << '|';
  //std::cout << std::endl;

  rNorm0 = normInfty(r);

  // �������� ��������� ������� �������.
  if (r == null)
  {
    rNorm = normInfty(r);
    epsRealMethod = norm;
    Nreal = steps;
    return Vres;
  }
  // ��������� ��������� �� ��������.
  if (norm < epsMethod)
  {
    rNorm = normInfty(r);
    epsRealMethod = norm;
    Nreal = steps;
    return Vres;
  }
  // �������� ��������� �� ����� �����
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

  // ������ ��� � �����.
  // ����� �� ������ ����� ������ ������� ���� �� n ����� (�������� 3).
  // null ������ �������� ���������� ����������� ������������� �������� (��� �������� ��������� �� ��������).
  for (int s = 1; s < size; s++)
  {
    tmp0 = A.dot(Vres);
    for (int i = 0; i < size; i++)
      r[i] = tmp0[i] - F[i];

    //std::cout << "�������: ";
    //for (int i = 0; i < size; i++)
    //  std::cout << r[i] << '|';
    //std::cout << std::endl;

    b = dotProduct(tmp1, r) / dotProduct(tmp1, hprev);

    for (int i = 0; i < size; i++)
      h[i] = (-1.) * r[i] + b * hprev[i];

    //// ��������: h � hprev ������ ���� ������������ ������������ A.
    //if (dotProduct(tmp1, hprev) * b - dotProduct(tmp1, r) != 0)
    //  throw(-1);

    tmp1 = A.dot(h);
    a = (-1.) * dotProduct(r, h) / dotProduct(tmp1, h);

    for (int i = 0; i < size; i++)
    {
      null[i] = Vres[i];
      Vres[i] = Vres[i] + a * h[i];
    }

    // ���� �� ������� ���� ������� ����� 0, �� ������� ����������� �������� ������ �������� ����.

    steps++;
    std::cout << "����� ����: " << steps << std::endl;

    norm = normDiffVecInfty(Vres, null);
    std::cout << "����� �������� �������� �� ������� � ���������� ����: " << norm << std::endl;

    // �������� ��������� ������� �������.
    if (r == null)
    {
      rNorm = normInfty(r);
      epsRealMethod = norm;
      Nreal = steps;
      return Vres;
    }
    // ��������� ��������� �� ��������.
    if (normDiffVecInfty(Vres, null) < epsMethod)
    {
      rNorm = normInfty(r);
      epsRealMethod = norm;
      Nreal = steps;
      return Vres;
    }
    // �������� ��������� �� ����� �����
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

// ����������� ������� �������� ������
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

// �������� ���������� ������� �������� � �������� �����
void solution(std::vector<std::vector<double>>& Utest, std::vector<std::vector<double>>& Vtest, std::vector<double> X, std::vector<double> Y, int n, int m, double h, double k, std::vector<double>& V)
{
  // ������� �� ��������.
  double stepY = 0.;
  double stepX = 0.;
  for (int i = 0; i < X.size(); i++)
  {
    Utest[i][0] = mu3t(stepX); // �������� �� ��� X ��� y = 0;
    Vtest[i][0] = mu3t(stepX);

    Utest[i][m] = mu4t(stepX); // �������� �� ��� X ��� y = 1;
    Vtest[i][m] = mu4t(stepX);

    stepX += h;
  }
  for (int j = 0; j < Y.size(); j++)
  {
    Utest[0][j] = mu1t(stepY); // �������� �� ��� Y ��� x = 0;
    Vtest[0][j] = mu1t(stepY);

    Utest[n][j] = mu2t(stepY); // �������� �� ��� Y ��� x = 1;
    Vtest[n][j] = mu2t(stepY);

    stepY += k;
  }

  // ������ ������� �������� ������.
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

  // ������ ������� �� �������� ������
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
  // ������� �� ��������.
  double stepY = 0.;
  double stepX = 0.;
  for (int i = 0; i < X.size(); i++)
  {
    Vmain[i][0] = mu3m(stepX); // �������� �� ��� X ��� y = 0;
    Vmain[i][m] = mu4m(stepX); // �������� �� ��� X ��� y = 1; 

    stepX += h;
  }
  for (int j = 0; j < Y.size(); j++)
  {
    Vmain[0][j] = mu1m(stepY); // �������� �� ��� Y ��� x = 0;
    Vmain[n][j] = mu2m(stepY); // �������� �� ��� Y ��� x = 1;

    stepY += k;
  }

  // ������ ������� �� �������� ������
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

  int n = 0; // ����� ��������� �� x;
  int m = 0; // ����� ��������� �� y;
  int Nmax = 0; // ����� �����;
  int NrealTest = 0;
  int NrealMain = 0;
  int NrealMain2 = 0;
  double eps = 0.5 * (1 / pow(10.0, 6.0)); // ����������� ������� �������� ������;
  double eps2 = 0.;
  double epsRealMethodTest = 0.;
  double epsRealMethodMain = 0.;
  double epsRealMethodMain2 = 0.;
  double rNormTest = 0.; // ����� �������; r - residual (�������).
  double rNormMain = 0.;
  double rNormMain2 = 0.;
  double rNorm0Test = 0.;
  double rNorm0Main = 0.;
  double rNorm0Main2 = 0.;
  std::vector<double> xyTest;
  std::vector<double> xyMain;
  int deg = 1; // �������� (��� �������� ��������� �� ��������)

  std::cout << "������� ����� ��������� �� ��� x." << '\n';
  std::cin >> n;
  std::cout << "������� ����� ��������� �� ��� y." << '\n';
  std::cin >> m;
  std::cout << "������� ����� ����� ������." << '\n';
  std::cin >> Nmax;
  std::cout << "�������� ��������� �� �������� 0.5*10^(-x) (������� ����� x)." << '\n';
  std::cin >> deg;

  double epsMethod = 0.5 * pow(10, (-1) * deg);

  double a = 0., b = 1., c = 0., d = 1.;

  double h = (b - a) / (double)n; // ��� �� x;
  double k = (d - c) / (double)m; // ��� �� y;

  std::vector<double> X(n + 1); // ���� �� ��� x;
  std::vector<double> Y(m + 1); // ���� �� ��� y;

  // �������� ������� X � Y:
  X[0] = 0.;
  Y[0] = 0.;
  for (int i = 1; i < n + 1; i++)
    X[i] = X[i - 1] + h;
  for (int j = 1; j < m + 1; j++)
    Y[j] = Y[j - 1] + k;

  double coefHor = 1. / (h * h);
  double coefVer = 1. / (k * k); // ������������: A, 1/h^2 � 1/k^2 (��. ������)
  double coefA = (-2.) * (coefHor + coefVer);

  int rows = (n - 1) * (m - 1), cols = (n - 1) * (m - 1); // ����� ����� � �������� ������� A.
  SparseMatrix A(rows, cols);
  calcA(A, h, k, n, m, coefA, coefHor, coefVer);

  std::vector<double> Fm((n - 1) * (m - 1)); // ������ ��������� �������� ���� (������ �����) [MAIN]
  std::vector<double> Ft((n - 1) * (m - 1)); // ������ ��������� �������� ���� (������ �����) [TEST]

  calcFt(Ft, X, Y, n, m, coefHor, coefVer);
  calcFm(Fm, X, Y, n, m, coefHor, coefVer);

  if (positiveDefiniteABool(A, Fm) == 1)
  {
    // ������� A > 0, ����� ��������� ����� ��.
  }
  else
  {
    // �������� ���� ����� � ������ �����.
    A.ChangeSign();
    for (int i = 0; i < A.getRows(); i++)
    {
      Fm[i] = (-1.) * Fm[i];
      Ft[i] = (-1.) * Ft[i];
    }
  }

  //// �������� ������ ��� ������ ��.
  //// ��������� �����������
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

  //// ����� ������� A.
  //for (int i = 0; i < rows; i++)
  //{
  //  std::cout << '\n';
  //  for (int j = 0; j < cols; j++)
  //    std::cout << A.get(i, j) << '|';
  //}

  std::vector<double> V(A.getRows()); 
  
  std::vector<std::vector<double>> Utest((n + 1), std::vector<double>(m + 1)); // ������ ������� �������� ������.
  std::vector<std::vector<double>> Vtest((n + 1), std::vector<double>(m + 1)); // ��������� ������� �������� ������. 
  std::vector<std::vector<double>> Vmain((n + 1), std::vector<double>(m + 1)); // ��������� ������� �������� ������.
  std::vector<std::vector<double>> Vmain2((2 * n + 1), std::vector<double>(2 * m + 1)); // ��������� ������� �������� ������ � ��������� ������.
  std::vector<std::vector<double>> V0((n + 1), std::vector<double>(m + 1)); // ��������� �����������
  std::vector<std::vector<double>> UdV((n + 1), std::vector<double>(m + 1)); // Utest difference Vtest
  std::vector<std::vector<double>> VdV2((n + 1), std::vector<double>(m + 1)); // Vmain difference V2main

  // ��������� ����������� ����� Python.
  //system("start env\\Scripts\\activate.bat"); �� ��������.
  system("cls");
  int flag = 1; // ���� ��� ������.
  int flagTask = 0; // 1 - ������ �������� ������. 2 - ������ �������� ������.
  int var;
  do
  {
    std::cout << "��� �� ������ �������? ������� ��������������� ����� ����" << '\n';
    std::cout << "1. ������ �������� ������" << '\n';
    std::cout << "2. ������ �������� ������" << '\n';
    std::cout << "3. ���������� ������" << '\n';
    std::cout << "4. �������" << '\n';
    std::cout << "5. �������" << '\n';
    std::cout << "6. ������" << '\n';
    std::cout << "7. ���������� ���������, �������� ��������� �� �������� � ����� �����." << '\n';
    std::cout << "8. �����" << '\n';
    std::cin >> var;
    switch (var)
    {
    case 1:
    {
      V = conjugateGradient(A, Ft, epsMethod, Nmax, NrealTest, epsRealMethodTest, rNormTest, rNorm0Test);
      solution(Utest, Vtest, X, Y, n, m, h, k, V);

      // w - write.
// �������� ����� � ������� csv ��� ������ ��������.
// ������ ������ ofstream ����� ��� ������ ������ � ����. ifstream - ��� ������. 
// ������� ������� ��� ���������� ������� �������� ������.
      std::string tableVtest = "tableV.csv"; // ������������ ����� � ��������. 
      std::ofstream wDataVt; // ����� ��� ������. 
      wDataVt.open(tableVtest); // ��������� ���� ��� ������.
      // ������ ������. ����� ��������� �� ��� �: x0, x1, ..., xn 
      wDataVt << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataVt << X[i];
        if (i < n)
          wDataVt << ","; // � ����� ������ .csv �� ������ ���� �������. ������������ ���� ������.
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

      // ������� ������� ��� ������� ������� �������� ������.
      std::string tableUtest = "tableUtest.csv";
      std::ofstream wDataUt;
      wDataUt.open(tableUtest);
      wDataUt << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataUt << X[i];
        if (i < n)
          wDataUt << ","; // � ����� ������ .csv �� ������ ���� �������. ������������ ���� ������.
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

      // ���������� �������� ������� � ���������� �������.
      for(int j = 0; j < m+1; j++)
        for (int i = 0; i < n + 1; i++)
          UdV[i][j] = Utest[i][j] - Vtest[i][j];

      // ������ � .csv � .xlsx �������� ������� � ���������� �������.
      std::string tableUdV = "tableUdV.csv";
      std::ofstream wDataUdV;
      wDataUdV.open(tableUdV);
      wDataUdV << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataUdV << X[i];
        if (i < n)
          wDataUdV << ","; // � ����� ������ .csv �� ������ ���� �������. ������������ ���� ������.
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

      //// ����� Utest.
      //std::cout << '\n';
      //for (int j = 1; j < m; j++)
      //{
      //  std::cout << '\n';
      //  for (int i = 1; i < n; i++)
      //    std::cout << Utest[i][j] << '|'; // ������� � �������� ������� ������������ ��� X, ������������ ������ ������������� �� ������.
      //}
      //// ����� Vtest.
      //std::cout << '\n';
      //for (int j = 1; j < m; j++)
      //{
      //  std::cout << '\n';
      //  for (int i = 1; i < n; i++)
      //    std::cout << Vtest[i][j] << '|'; // ������� � �������� ������� ������������ ��� X, ������������ ������ ������������� �� ������.
      //}
      //std::cout << '\n';

      //// ��������� Utest � Vtest ����������� �� ���������� �����.
      //std::cout << '\n';
      //for (int j = 1; j < m; j++)
      //{
      //  std::cout << '\n';
      //  for (int i = 1; i < n; i++)
      //    std::cout << Utest[i][j] << '|' << Vtest[i][j] << '\n'; // ������� � �������� ������� ������������ ��� X, ������������ ������ ������������� �� ������.
      //}
      //std::cout << '\n';

      system("cls");

      break;
    }
    case 2:
    {
      V = conjugateGradient(A, Fm, epsMethod, Nmax, NrealMain, epsRealMethodMain, rNormMain, rNorm0Main);
      solution(Vmain, X, Y, n, m, h, k, V);

      // ������� ��� ������ c ���������� 2*n � 2*m.
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

      std::vector<double> Fm2((n2 - 1) * (m2 - 1)); // ������ ��������� �������� ���� (������ �����) [MAIN]

      int rows2 = (n2 - 1) * (m2 - 1), cols2 = (n2 - 1) * (m2 - 1); // ����� ����� � �������� ������� A.

      SparseMatrix A2(rows2, cols2);

      calcA(A2, h2, k2, n2, m2, coefA2, coefHor2, coefVer2);

      std::vector<double> V2(A2.getRows());

      std::vector<std::vector<double>> V2main((n2 + 1), std::vector<double>(m2 + 1)); // ��������� ������� �������� ������.

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
          wDataV2 << ","; // � ����� ������ .csv �� ������ ���� �������. ������������ ���� ������.
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

      // ���������� �������� ������� � ���������� �������.
      for (int j = 0; j < m + 1; j++)
        for (int i = 0; i < n + 1; i++)
          VdV2[i][j] = Vmain[i][j] - V2main[2*i][2*j];

      // ������ � .csv � .xlsx �������� ������� � ���������� �������.
      std::string tableVdV2 = "tableVdV2.csv";
      std::ofstream wDataVdV2;
      wDataVdV2.open(tableVdV2);
      wDataVdV2 << "," << "xi" << ",";
      for (int i = 0; i < n + 1; i++)
      {
        wDataVdV2 << X[i];
        if (i < n)
          wDataVdV2 << ","; // � ����� ������ .csv �� ������ ���� �������. ������������ ���� ������.
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
        std::cout << "������� ����� � ������ ��������� �� x n = " << n << " � ������ ��������� �� y m = " << m << '.' << std::endl;
        std::cout << "��������, ����� ������ ������: �������� ��� ��������.";
      }
      else if (flagTask == 1)
      {
        xyTest = maxDeviation(Utest, Vtest, X, Y, n, m);
        std::cout << "��� ������� �������� ������ ������������ ����� � ������ ��������� �� x n = " << n << " � ������ ��������� �� y m = " << m << ',' << std::endl;
        std::cout << "����� ����������� ����������, ��������� ������ �����������," << std::endl;
        std::cout << "�������� ��������� �� �������� eps_(met) = " << epsMethod << " � �� ����� �������� Nmax = " << Nmax << '.' << std::endl;
        std::cout << "�� ������� ����� (����) ��������� �������� N = " << NrealTest << " � ���������� �������� ������������� ������ eps^(N) = " << epsRealMethodTest << '.' << std::endl;
        std::cout << "����� (����) ������ � �������� ||R^(N)|| = " << rNormTest << ", ��� ������� ���� ������������ ����� ������� ." << std::endl;
        std::cout << "�������� ������ ������ ���� ������ � ������������ �� ����� " << eps << "; ������ ������ � ������������ eps_1 = " << error(Utest, Vtest, n, m) << '.' << std::endl;
        std::cout << "������������ ���������� ������� � ���������� ������� ����������� � ���� x = " << X[xyTest[0]] << "; y = " << Y[xyTest[1]] << '.' << std::endl;
        std::cout << "� �������� ���������� ����������� ������������ ������� �����������." << std::endl;
        std::cout << "������� ���� �� ��������� ����������� ||R^(0)|| = " << rNorm0Test << ". ��� ������� ���� ������������ ����� ��������." << std::endl;
      }
      else if (flagTask == 2)
      {
        std::cout << "��� ������� �������� ������ ������������ ����� � ������ ��������� �� x n = " << n << " � ������ ��������� �� y m = " << m << ',' << std::endl;
        std::cout << "����� ����������� ����������, ��������� ������ �����������," << std::endl;
        std::cout << "�������� ��������� �� �������� eps_(met) = " << epsMethod << " � �� ����� �������� Nmax = " << Nmax << '.' << std::endl;
        std::cout << "�� ������� ����� (����) ��������� �������� N = " << NrealMain << " � ���������� �������� ������������� ������ eps^(N) = " << epsRealMethodMain << '.' << std::endl;
        std::cout << "����� (����) ������ � �������� ||R^(N)|| = " << rNormMain << ", ��� ������� ������������ ����� �������� ." << std::endl;
        std::cout << "� �������� ���������� ����������� �� �������� ����� ������������ ������� �����������." << std::endl;
        std::cout << "�� �������� ����� ������� ���� �� ��������� ����������� ||R^(0)|| = " << rNorm0Main << ". ��� ������� ���� ������������ ����� ��������." << std::endl;
        std::cout << '\n';
        std::cout << "��� �������� �������� ������������ ����� � ���������� �����," << std::endl;
        std::cout << "����� ����������� ����������, ��������� ������ �����������," << std::endl;
        std::cout << "�������� ��������� �� �������� eps_(met-2) = " << epsMethod << " � �� ����� �������� Nmax-2 = " << Nmax << '.' << std::endl;
        std::cout << "�� ������� ������ (����) ��������� �������� N2 = " << NrealMain2 << " � ���������� �������� ������������� ������ eps^(N2) = " << epsRealMethodMain2 << '.' << std::endl;
        std::cout << "����� (����) �� ����� � ���������� ����� ������ � �������� ||R^(N2)|| = " << rNormMain2 << ", ��� ������� ���� ������������ ����� ��������." << std::endl;
        std::cout << "�������� ������ ������ ���� ������ � ��������� �� ���� ��� eps = " << eps << "; ������ ������ � ��������� eps_2 = " << eps2 << '.' << std::endl;
        std::cout << "������������ ���������� ��������� ������� �� �������� ����� � ����� � ���������� ����� ����������� � ���� x = " << X[xyMain[0]] << "; y = " << Y[xyMain[1]] << '.' << std::endl;
        std::cout << "� �������� ���������� ����������� �� ����� � ���������� ����� ������������ ������� �����������." << std::endl;
        std::cout << "�� ����� � ���������� ����� ������� ���� �� ��������� ����������� ||R^(0)|| = " << rNorm0Main2 << ". ��� ������� ���� ������������ ����� ��������." << std::endl;
      }

      break;
    }
    case 5:
    {
      if (flagTask == 0)
      {
        system("cls");
        std::cout << "��������� ������ �������� (1.) ��� �������� ������ (2.)" << std::endl;
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
        std::cout << "��������� ������ �������� (1.) ��� �������� ������ (2.)" << std::endl;
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
            wDataV0 << ","; // � ����� ������ .csv �� ������ ���� �������. ������������ ���� ������.
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
      std::cout << "������� ����� ��������� �� ��� x." << '\n';
      std::cin >> n;
      std::cout << "������� ����� ��������� �� ��� y." << '\n';
      std::cin >> m;
      std::cout << "������� ����� ����� ������." << '\n';
      std::cin >> Nmax;
      std::cout << "�������� ��������� �� �������� 0.5*10^(-x) (������� ����� x)." << '\n';
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
      //  // ������� A > 0, ����� ��������� ����� ��.
      //}
      //else
      //{
      //  // �������� ���� ����� � ������ �����.
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
      std::cout << "�������� ������������ ������ ����";

      break;
    }
    }
  } while (flag == 1);

  // ����������� ����������� ����� Python.
  //system("deactivate"); // �� ��������.

  return 0;
}