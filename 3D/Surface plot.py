# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure(figsize=(16,12))
ax = fig.gca(projection="3d")

# 准备数据
x = np.arange(-5, 5, 0.25)    #生成[-5,5]间隔0.25的数列，间隔越小，曲面越平滑
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x,y)  #格点矩阵,原来的x行向量向下复制len(y)次，形成len(y)*len(x)的矩阵，即为新的x矩阵；原来的y列向量向右复制len(x)次，形成len(y)*len(x)的矩阵，即为新的y矩阵；新的x矩阵和新的y矩阵shape相同
r = np.sqrt(x ** 2 + y ** 2)
z = np.sin(r)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)  # cmap指color map 图形的颜色系　冷色暖色

# 自定义z轴
ax.set_zlim(-1, 1)
ax.zaxis.set_major_locator(LinearLocator(20))  # z轴网格线的疏密，刻度的疏密，20表示刻度的个数
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # 将z的value字符串转为float，保留2位小数

#设置坐标轴的label和标题
ax.set_xlabel('x',size=15)
ax.set_ylabel('y',size=15)
ax.set_zlabel('z',size=15)
ax.set_title("Surface plot", weight='bold', size=20)

#添加右侧的色卡条
fig.colorbar(surf, shrink=0.6, aspect=8)  # shrink表示整体收缩比例，aspect仅对bar的宽度有影响，aspect值越大，bar越窄
ax.view_init(elev=45, azim=45)  # 调整坐标轴的显示角度
plt.show()