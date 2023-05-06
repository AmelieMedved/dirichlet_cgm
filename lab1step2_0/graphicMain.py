import pandas as pd
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
Data = pd.read_csv("tableV.csv", sep = ',', index_col=0, encoding='cp1251')
Data.drop(Data.index[[0]], inplace = True)
Data.drop(Data.columns[0], axis = 1, inplace = True)
Data = Data.to_numpy()
Data = Data.astype('float64')
n = Data[0].size # узлы по x
m = Data[:,0].size # узлы по y
fig = plt.figure(figsize = (7, 4))
ax = fig.add_subplot(projection = '3d')
X = np.linspace(0, 1, n)
Y = np.linspace(0, 1, m)
x, y = np.meshgrid(X, Y)
z = Data;
ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('v(x,y)')
ax.set_title('Точное решение разностной схемы')
plt.show()