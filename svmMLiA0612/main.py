# coding=UTF-8
# 程序名称：第六章 支持向量机
# 程序功能：
# 1.第一个例子：读取testSet.txt文档中的信息，用简化版的SMO算法实现alpha值的优化（αi和αj的都是全部遍历的 ）
# 2.第二个例子：读取testSet.txt文档中的信息，用完整版的SMO算法实现alpha值的优化（αi只遍历值大于0的，α通过辅助函数innerL(i,oS)寻找）
# 3.第三个例子：读取testSetRBF.txt文档中的信息，先确定具体的支持向量，再使用前面找到的支持向量来计算训练数据集testSetRBF.txt和测试数据集testSetRBF2.txt的预测错误率。
# 4.第四个例子：读取digits文件中的训练数据集trainingDigits的数据，先确定具体的支持向量，再计算训练数据集和测试数据集的预测错误率
# 程序说明：
# 时间：2018年6月12日（星期二）上午
# 进度：
# 6月12日星期二看书，数学概念很难，看了smoSimple（）的程序，简单理解不知道具体的数学推到过程，下午晚上一直在看支持向量机……看不太懂
# 6月13日星期三上午将SVM数学原理又看了。能理解数学公式的推导思路，不能将程序和公式联系在一起。
# 6月19日星期二看书，看西瓜书上的数学推导。看到了6.4的软间隔与正则化。
# 6月29日星期五17:56实现三个例子的程序运行




import time
start = time.clock()
from numpy import *
lenofjing = 30  #print 中#的长度
##############      第一个例子 P96    ##########
#通过加载文档的数据计算出常数b和alpha的值

import plot
import svmMLiAPage96 #简化版的SMO程序
dataArr,labelArr = svmMLiAPage96.loadDataSet('testSet.txt')

print type(array(dataArr))
plot.plotmap(array(dataArr), array(labelArr)) #文档中的数据有两个属性，因此可以用二维图形形象的展示出来

b,alphas = svmMLiAPage96.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print "##############      第一个例子 P96    ##########"
print "\n","#" * lenofjing
print "\n常数偏移量b的值是：\n",b    # b = [[-3.81480118]]
print "\nalphas中>0的量（支持向量）是：\n",alphas[alphas>0] #alphas[alphas>0] = [[0.15169432 0.1579583  0.05515608 0.3648087 ]]
print "\n将alphas > 0 的数据和标签输出，这些数据就是支持向量:"
#[4.658191, 3.507396] -1.0
#[3.457096, -0.082216] -1.0
#[6.080573, 0.418886] 1.0
for i in range(100):
    if alphas[i]>0:
        print dataArr[i], labelArr[i]
# 第一个例子运行结果如下：
# 常数偏移量b的值是：
# [[-3.7870077]]
#
# alphas中>0的量（支持向量）是：
# [[0.05826759 0.10300911 0.19073324 0.19485838 0.15715017]]
#
# 将alphas > 0 的数据和标签输出，这些数据就是支持向量:
# [4.658191, 3.507396] -1.0
# [3.457096, -0.082216] -1.0
# [2.893743, -1.643468] -1.0
# [5.286862, -2.358286] 1.0
# [6.080573, 0.418886] 1.0



##############      第二个例子 P100    ##########

import svmMLiAPage100 #完整版的Platt SMO程序
dataArr, labelArr = svmMLiAPage100.loadDataSet('testSet.txt') #加载数据
#建立了数据结构
#P100训练（使用完整的SMO算法和核函数）
#常数b是一个具体的数值， 阿尔法α=0表示不是支持向量，α>0表示是支持向量；阿尔法α的长度和具体的数据相关，相同的数据运算出的结果也可能不相同，为什么呢？
b, alphas = svmMLiAPage100.smoP(dataArr, labelArr, 0.6, 0.001, 40) #使用数据计算出alphas和b的值
print "##############      第二个例子 P100      ##########"
print "\n常数偏移量b的值是：\n",b    # b = [[-2.89901748]]
print "\nalphas中>0的量（支持向量）是：\n",alphas[alphas>0]    #[[0.06961952 0.0169055  0.0169055  0.0272699  0.04522972 0.0272699 0.0243898  0.06140181 0.06140181]]
print "\n将alphas > 0 的数据和标签输出，这些数据就是支持向量:"
# 第二个例子运行结果如下：
# 常数偏移量b的值是：
# [[-2.89901748]]
#
# alphas中>0的量（支持向量）是：
# [[0.06961952 0.0169055  0.0169055  0.0272699  0.04522972 0.0272699
#   0.0243898  0.06140181 0.06140181]]
#
# 将alphas > 0 的数据和标签输出，这些数据就是支持向量:
# [3.542485, 1.977398] -1.0
# [2.114999, -0.004466] -1.0
# [8.127113, 1.274372] 1.0
# [4.658191, 3.507396] -1.0
# [8.197181, 1.545132] 1.0
# [7.40786, -0.121961] 1.0
# [6.960661, -0.245353] 1.0
# [6.080573, 0.418886] 1.0
# [3.107511, 0.758367] -1.0
#
# ws的值如下：
# [[ 0.65307162]
#  [-0.17196128]]
#
# 第一个数据的预测值是：
# [[-0.92555695]]
#
# 第一个数据的真实值是：
# -1.0

for i in range(100):
    if alphas[i]>0:
        print dataArr[i], labelArr[i]
ws = svmMLiAPage100.calcWs(alphas, dataArr, labelArr) #使用alpha计算ws的值,计算公式是周志华P123式子6.9
#ws权重的值如下,权重的长度和数据x的维度相同。
#[[ 0.59557521]
#[-0.25721564]]

print "\nws的值如下：\n",ws  #ws 是输入数据x的权重，因此长度和x的长度相同，在这里都是长度都是2。
datMat = mat(dataArr)
print "\n第一个数据的预测值是：\n",datMat[0]*mat(ws)+b #计算第一个数据的预测值 (使用ws和b的值计算 ) 计算公式周志华P123式子6.7 y=wT*x+b
print "\n第一个数据的真实值是：\n",labelArr[0] #读取第一个数据的真实值
#P109  测试（使用核函数）
print "\n","#" * lenofjing

print "\n####################3   第三个例子 使用核函数       #######################333333\n"
svmMLiAPage100.testRbf()
# 第三个例子运行结果如下：
# iteration number: 6
# there are 26 Support Vectors
# the training error rate is: 0.060000
# the test error rate is: 0.080000

##############      第四个例子 111 手写数字识别    ######################
print "\n##############      第四个例子 111 手写数字识别    ######################\n"
import svmMLiAPage100 #完整版的Platt SMO程序
svmMLiAPage100.testDigits()
# 第四个程序的运行结果如下：
# iteration number: 4
# there are 126 Support Vectors
# the training error rate is: 0.000000
# the test error rate is: 0.005376
print "\n","#" * lenofjing

#以下程序用于测试程序运行的时间
end = time.clock()
print "程序运行时间是:",end - start
# 程序运行结果如下：
# 程序运行时间是: 8.744524

