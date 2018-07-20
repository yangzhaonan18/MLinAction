# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 12)) #一整幅图的长和宽
ax1 = fig.gca(projection="3d")

# 在x轴和y轴画sin函数
x = np.linspace(0, 1, 100) #等分成100份
y = np.sin(2 * np.pi * x) + 1  # 2*π*x∈[0,2π] y属于[0,2] sin()使用弧度计算一个周期是0到2pi
ax1.plot(x, y, zs=0, zdir='z', label="sin curve in (x,y)") # 如何设置label的大小？？

colors = ('r', 'g', 'b', 'k')
x = np.random.sample(20 * len(colors))
y = np.random.sample(20 * len(colors))
c_list = []
for c in colors:
    c_list.extend([c] * 20)  # 比如，[colors[0]*5]的结果是['r','r','r','r','r']，是个list
ax1.scatter(x, ys=y, zs = 0, zdir = 'y', c = c_list, label = u"scatter points in (x,z)")

ax1.legend()#绘制图右上角的两个label名称
ax1.set_xlim(0, 1)#坐标轴的范围
ax1.set_ylim(0, 2)
ax1.set_zlim(0, 1)
ax1.set_xlabel("X",size = 80)#坐标轴的名称和字体的大小
ax1.set_ylabel("Y",size = 80)
ax1.set_zlabel("Z",size = 80)

ax1.view_init(elev=45, azim=45)  # 调整坐标轴的显示角度
plt.show()