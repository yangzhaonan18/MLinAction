# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)       #测试数据
print X, "\n#"*2,Y, "\n#"*2, Z
#cset = ax1.contour(X, Y, Z, cmap=cm.coolwarm)  #color map选用的是coolwarm
cset = ax1.contour(X, Y, Z , extend3d=True, cmap=cm.coolwarm)
ax1.set_title("Contour plot", color='b', weight='bold', size=25)
plt.show()



# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(figsize=(16, 12))
ax2 = fig.gca(projection="3d")  # get current axis
X, Y, Z = axes3d.get_test_data(0.001)  #测试数据

ax2.plot_surface(X, Y, Z, rstride=4, cstride= 4, alpha=0.5,cmap = cm.coolwarm)
cset = ax2.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax2.contour(X, Y, Z, zdir="x", offset=-40, cmap=cm.coolwarm) #沿x轴方向投影（将ｘ轴作为新坐标系的ｚ轴，在新坐标系的xy平面作图。）
cset = ax2.contour(X, Y, Z, zdir="y", offset=-40, cmap=cm.coolwarm)

ax2.set_xlabel('X',size = 50)
ax2.set_xlim(-40, 40)
ax2.set_ylabel('Y',size = 50)
ax2.set_ylim(-40, 40)
ax2.set_zlabel('Z',size = 50)
ax2.set_zlim(-100, 100)
ax2.set_title('Contour plot', alpha=0.5, color='g', weight='bold', size=30)
ax2.view_init(20,20)

plt.show()

