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
    r[i] = (-1.) * tmp0[i] + F[i];
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