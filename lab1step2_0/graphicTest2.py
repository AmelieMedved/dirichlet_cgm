import pandas as pd
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

Data = [pd.read_csv("tableUtest.csv", sep = ',', index_col=0, encoding='cp1251'), 
        pd.read_csv("tableV.csv", sep = ',', index_col=0, encoding='cp1251'), 
        pd.read_csv("tableV0.csv", sep = ',', index_col=0, encoding='cp1251'),
        pd.read_csv("tableUdV.csv", sep = ',', index_col=0, encoding='cp1251')]

for i in range(4):
  Data[i].drop(Data[i].index[[0]], inplace = True)
  Data[i].drop(Data[i].columns[0], axis = 1, inplace = True)
  Data[i] = Data[i].to_numpy()
  Data[i] = Data[i].astype('float64')

n = Data[0][0].size # узлы по x
m = Data[0][:,0].size # узлы по y

fig = plt.figure(0, figsize = (7, 4))
ax = fig.add_subplot(projection = '3d')
X = np.linspace(0, 1, n)
Y = np.linspace(0, 1, m)
x, y = np.meshgrid(X, Y)
z = [Data[0], Data[1], Data[2], Data[3]]

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
ax.set_title('Точное решение задачи Дирихле для уравнения Пуассона')
ax.plot_surface(x, y, z[0], cmap=cm.coolwarm) # rstride = 1, cstride = 1

fig = plt.figure(1, figsize = (7, 4))
ax = fig.add_subplot(projection = '3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('v(x,y)')
ax.set_title('Точное решение разностной схемы')
ax.plot_surface(x, y, z[1], cmap = cm.coolwarm) # rstride = 1, cstride = 1

fig = plt.figure(2, figsize = (7, 4))
ax = fig.add_subplot(projection = '3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('v0(x,y)')
ax.set_title('Начальное приближение')
ax.plot_surface(x, y, z[2], cmap = cm.coolwarm) # rstride = 1, cstride = 1

fig = plt.figure(3, figsize = (7, 4))
ax = fig.add_subplot(projection = '3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('v0(x,y)')
ax.set_title('Разность точного и численного решения')
ax.plot_surface(x, y, z[3], cmap = cm.coolwarm) # rstride = 1, cstride = 1

plt.show()