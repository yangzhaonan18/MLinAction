# 2018-05-29-15:00   page27
# -*- coding: UTF-8 -*-

#程序说明如下
#标题：page 27 分类器的测试程序/算法
#内容：将数据集的前面部分作为测试集，后面部分作为训练集，输出预测的结果与实际分类对比并计算预测的错误率和错误数
#时间：2018年5月29日
'''

运行结果：
Running /home/yzn/PycharmProjects/kNN_meet_0529_4/main.py
line 22,the classifier came back with: 1, the real answer is: 2
line 74,the classifier came back with: 3, the real answer is: 1
line 83,the classifier came back with: 3, the real answer is: 1
line 91,the classifier came back with: 2, the real answer is: 3
line 99,the classifier came back with: 3, the real answer is: 1
预测出错的次数是 5.0
预测集的大小是 100
the total error rate is: 5/100 = 0.050000 '''

import kNN
#from numpy import *
#import operator
kNN.datingClassTest()            #运行分类器的测试代码, 没有return()


