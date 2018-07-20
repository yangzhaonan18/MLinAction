# coding=utf-8
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['legend.fontsize'] = 20  # mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_params_from_file()函数从文件加载配置信息

font = {
    'color': 'b',
    'style': 'oblique',
    'size': 20,
    'weight': 'bold'
}
fig = plt.figure(figsize=(16, 12))  #参数为图片大小
ax = fig.gca(projection='3d')  # get current axes，且坐标轴是3d的

# 准备数据
theta = np.linspace(-8 * np.pi, 8 * np.pi, 100)  # 生成等差数列，[-8π,8π]，个数为100
z = np.linspace(-2, 2, 100)  # [-2,2]容量为100的等差数列，这里的数量必须与theta保持一致，因为下面要做对应元素的运算
r = z ** 2 + 1
x = r * np.sin(theta)  # [-5,5]
y = r * np.cos(theta)  # [-5,5]
ax.set_xlabel("X", fontdict=font)
ax.set_ylabel("Y", fontdict=font)
ax.set_zlabel("Z", fontdict=font)
ax.set_title("Line Plot", alpha=0.5, fontdict=font) #alpha参数指透明度transparent
ax.plot(x, y, z, label='parametric curve')
ax.legend(loc='upper right') #legend的位置可选：upper right/left/center,lower right/left/center,right,left,center,best等等

plt.show()