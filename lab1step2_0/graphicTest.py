import pandas as pd
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
Data = [pd.read_csv("tableUtest.csv", sep = ',', index_col=0, encoding='cp1251'), pd.read_csv("tableV.csv", sep = ',', index_col=0, encoding='cp1251')]
Data[0].drop(Data[0].index[[0]], inplace = True)
Data[0].drop(Data[0].columns[0], axis = 1, inplace = True)
Data[0] = Data[0].to_numpy()
Data[0] = Data[0].astype('float64')
Data[1].drop(Data[1].index[[0]], inplace = True)
Data[1].drop(Data[1].columns[0], axis = 1, inplace = True)
Data[1] = Data[1].to_numpy()
Data[1] = Data[1].astype('float64')
n = Data[0][0].size # узлы по x
m = Data[0][:,0].size # узлы по y
fig = plt.figure(figsize = (7, 4))
ax1 = fig.add_subplot(1, 2, 1 ,projection = '3d')
ax2 = fig.add_subplot(1, 2, 2,projection = '3d')
X = np.linspace(0, 1, n)
Y = np.linspace(0, 1, m)
x, y = np.meshgrid(X, Y)
z = [Data[0], Data[1]]
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x,y)')
ax1.set_title('Точное решение задачи Дирихле для уравнения Пуассона')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('v(x,y)')
ax2.set_title('Точное решение разностной схемы')
#ax1.plot_surface(x, y, z[0], rstride = 1, cstride = 1, cmap=cm.coolwarm)
#ax2.plot_surface(x, y, z[1], rstride = 1, cstride = 1, cmap = cm.coolwarm)
ax1.plot_surface(x, y, z[0], cmap=cm.coolwarm)
ax2.plot_surface(x, y, z[1], cmap = cm.coolwarm)
plt.show()