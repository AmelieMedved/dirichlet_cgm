#pragma once

#include <vector>

// Не используем "using namespace std;", так как это вызывает двусмысленность при определении структуры pair (содержится в пространстве имен std).

struct pair
{
  int index; // Индекс столбцовый.
  double value; // Значение в ячейке с индексом index.
};

/*
ООП:
1. Инкапсуляция.
2. Наследование.
3. Полиморфизм.
*/

// Класс - определяемый пользователем тип данных.

class SparseMatrix
{
private: // Список свойств и методов для доступа только внутри класса.
  int rows, cols;
  std::vector<std::vector<pair>> data; // Указатель тип данных vector<pair> (динамический массив векторов с элементами - парами).
public: // Список методов
  SparseMatrix(int _rows, int _cols)
  {
    rows = _rows;
    cols = _cols;
    data.resize(rows);
    for (int i = 0; i < rows; i++)
      data[i].resize(0);
  }

  SparseMatrix(SparseMatrix& a)
  {
    rows = a.rows;
    cols = a.cols;
    data.resize(rows);
    for (int i = 0; i < rows; i++)
      data[i].resize(0);

    for (int i = 0; i < rows; i++)
      for (int k = 0; k < a.data[i].size(); k++)
        this->set(i, a.data[i][k].index, a.data[i][k].value);
  }


  ~SparseMatrix()
  {
    data.clear();
    rows = 0;
    cols = 0;
  }

  void set(int i, int j, double value)
  {
    if (value != 0)
        data[i].push_back(pair{ j, value });
  }

  double get(int i, int j)
  {
    if (rows == 0 && cols == 0)
      return 0.;
    for (int k = 0; k < data[i].size(); k++)
    {
      if (data[i][k].index == j)
        return data[i][k].value;
    }
    return 0.;
  }

  SparseMatrix operator+(SparseMatrix& a)
  {
    if (rows == a.rows && cols == a.cols)
    {
      SparseMatrix b(*this);
      for (int i = 0; i < rows; i++)
        for (int k = 0; k < data[i].size(); k++)
          b.data[i][k].value += a.data[i][k].value;
      return b;
    }
  }

  SparseMatrix operator-(SparseMatrix& a)
  {
    if (rows == a.rows && cols == a.cols)
    {
      SparseMatrix b(*this);
      for (int i = 0; i < rows; i++)
        for (int k = 0; k < data[i].size(); k++)
          b.data[i][k].value -= a.data[i][k].value;
      return b;
    }
  }

  SparseMatrix operator*(SparseMatrix& a)
  {
    if (rows == a.rows && cols == a.cols)
    {
      SparseMatrix b(*this);
      for (int i = 0; i < rows; i++)
        for (int k = 0; k < data[i].size(); k++)
          b.data[i][k].value *= a.data[i][k].value;
      return b;
    }
  }

  std::vector<double> dot(std::vector<double>& v)
  {
    if (cols == v.size())
    {
      std::vector<double> b(v.size());
      for (int i = 0; i < rows; i++)
        for (int k = 0; k < data[i].size(); k++)
          b[i] += data[i][k].value * v[data[i][k].index];
      return b;
    }
    else
      throw(-1);
  }

  double dotProduct(SparseMatrix& v)
  {
    // Скалярное произведение двух векторов.
    if (rows == v.rows && cols == 1 && v.cols == 1)
    {
      double res = 0.;
      for (int j = 0; j < cols; j++)
        res += this->get(j, 0) * v.get(j, 0);
      return res;
    }
    else if (rows == 0)
    {
      double res = 0.;
      for (int j = 0; j < cols; j++)
        res += 0. * v.get(j, 0);
      return res;
    }
    else if (v.rows == 0)
    {
      double res = 0.;
      for (int j = 0; j < cols; j++)
        res += this->get(j, 0) * 0.;
      return res;
    }
    else
      throw (-1);
  }

  int getRows()
  {
    return this->rows;
  }

  int getCols()
  {
    return this->cols;
  }

  int getSize()
  {
    return this->cols * this->rows;
  }

  void ChangeSign()
  {
    for (int i = 0; i < this->getRows(); i++)
      for (int k = 0; k < this->data[i].size(); k++)
        data[i][k].value = (-1.) * data[i][k].value;
  }

  void resize(int rowsNew, int colsNew)
  {
    rows = rowsNew;
    cols = colsNew;
    data.clear();
    data.resize(rows);
    for (int i = 0; i < rows; i++)
      data[i].resize(0);
  }

};

