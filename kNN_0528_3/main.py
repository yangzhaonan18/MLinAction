# coding=utf-8
'''
Time:2018 05 28  mon 23:40
/home/yzn/PycharmProjects/kNN_0528_3/venv/bin/python /home/yzn/PycharmProjects/kNN_0528_3/main.py
Process finished with exit code 0
'''
#程序说明：
#标题：使用 Matplotlib 创建散点图  2.2.2 page 23
#内容：从文档中读取数据，存放方矩阵中，最后画scatter散点图
#时间：2018年5月28日 23：40  星期一    6月6日添加备注

import kNN
# 缺失下面这句将会提示：NameError: name 'array' is not defined
from numpy import array

import matplotlib.pyplot as plt
fig = plt.figure()    #画图
ax = fig.add_subplot(111)   #这里的(111) 可以写成(1,1,1) 表示将整个画图界面划分成1行1列，用其中的第1个（从左到右从上到下数的第1个），
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
print "#"*50
print type(datingLabels)
print "#"*50
print type(datingDataMat)
print "#"*50
print datingDataMat
print "#"*50
print datingDataMat[:,0]

#选用第二个，第三个属性值画图时：
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])

#.scatter()中定义点的 横坐标纵坐标 属性的数目必须相同，否则会提示 raise ValueError("x and y must be the same size")
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], s = 14.0*array(datingLabels[:]),c = array(datingLabels))

#ax.axis([-2,25,-0.2,2.0])   # size of xlabel and ylabel  前两个是X轴的范围 后两个是Y轴的范围
plt.xlabel('Percentage of Time Spent Playing Video Games',fontsize = 12)  # font：字体
plt.ylabel('Liters of Ice Cream Consumed Per Week',fontsize = 12)
plt.show()


##########程序运行过程中的测试数据如下：
print('从datingTestSet2.txt 中提取出的前三列 feature 数据是：')
print(datingDataMat)  # just for the test
print('第四列label value 是')
print(datingLabels)   # just for the test








#测试print()的用法:
print('#########')
#print("\n")
print("1：测试print()的用法\n")
#print('\n')
print('2：测试print()的用法\n')
#print "\n"
print "3：测试print()的用法\n"
#print '\n'
print '4：测试print()的用法\n'



