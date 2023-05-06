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
    r[i] = (-1.) * tmp0[i] + F[i];
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