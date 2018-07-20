# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection="3d")
a = zip(['r', 'g', 'b', 'k'], [30, 20, 10, 0])
for c, z in a:
    xs = np.arange(20)  # [0,20)之间的自然数,共20个
    ys = np.random.rand(20)  # 生成20个[0,1]之间的随机数
    cs = [c] * len(xs)  # 生成颜色列表
    ax.bar(xs, ys, z, zdir='x', color=cs, alpha=0.8)  # 以zdir='x'，指定z的方向为x轴，那么x轴取值为[30,20,10,0]
#   ax1.bar(xs, ys, z, zdir='y', color=cs, alpha=0.8)
#   ax1.bar(xs, ys, z, zdir='z', color=cs, alpha=0.8)
ax.set_xlabel('X',size = 80)
ax.set_ylabel('Y',size = 80)
ax.set_zlabel('Z',size = 80)
ax.set_title(u'Bar plot　柱状图', size=45, weight='bold')
ax.view_init(elev=45, azim=45)

plt.show()