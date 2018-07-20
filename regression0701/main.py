# coding=UTF-8
# 程序名称：第八章  回归：预测数值型数据
# 程序功能：
# 1.第一个例子：
# 程序说明：
# 时间：2018年07月01日（星期日）15:17
# 进度：
# 7月1日：一节一代码逐步推进
# 7月2日：测试完所有代码
import  time
start = time.clock()

from numpy import *
import  regression
import  plot
lenofjing = 30

# print "######### 第一个例子：直接使用公式计算出拟合直线（一条）的系数ws的值，前提时数据矩阵可逆   #########"
xArr, yArr = regression.loadDataSet('ex0.txt') #输入数据都需要加一个常数偏量（原始的数据集是一个数据x和一个标签y,增加一个常数偏量1）
ws = regression.standRegres(xArr,yArr)  #权重的长度永远和特征输入x的长度相同，这里是2
# print "\nxArr:\n", len(xArr)
# print "\nyArr:\n", len(yArr)
# print "\nws:\n", ws
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat * ws
print "\n预测结果和真实结果之间的相关系数是：\n", corrcoef(yHat.T, yMat) # 两个参数要求都是行向量才能计算相关系数
print "\n使用拟合直线计算出的预测结果是：\n", yHat
print "\n第一个例子：拟合直线和原始数据 绘制成第一个图：\n"
plot.plot1(xMat, yMat, ws, titlename = u"第一个例子：标准线性回归 Page139")


print "########       第二例子： 局部加权线性回归      #################"
xArr, yArr = regression.loadDataSet('ex0.txt')
print "\n计算出xArr[0]的估计值为：",regression.lwlr(xArr[0], xArr, yArr, 1.0)
print "\n计算出xArr[1]的估计值为：",regression.lwlr(xArr[1], xArr, yArr, 0.001)
#print "\n计算出xArr的估计值为：",regression.lwlr(xArr, xArr, yArr, 0.003)
#
yHat = zeros(len(xArr))
# #  yHat = regression.lwlr(xArr, xArr, yArr, k= 0.01)
for i in range(len(xArr)):
    value = regression.lwlr(xArr[i], xArr, yArr, k = 0.0018) #局部加权回归 调节高斯函数中参数k的值可以改变拟合的程度（拟合线的形状k越大则近直线，k过小则过拟合）
    yHat[i] = value # 存放数据i的预测值在yHat[i]中
print "\nyHat的值是：\n", yHat
# print "###########33333333333", shape(xMat)
# print "###########33333333333", shape(yMat)
# print "###########33333333333", shape(yHat)
xMat = mat(xArr)
yMat = mat(yArr)
plot.plot2(xMat, yMat, yHat , titlename = u"第二例子： 局部加权线性回归 Page141")


print "########      第三个例子：Page145 预测鱼的年龄      #################"
abX, abY = regression.loadDataSet('abalone.txt')
#使用三个不同的k值 0.1，1，10 来预测数据集前100个数据的y值
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
print "\n局部线性回归后一百个数据集的误差情况："
print "局部线性回归，k=0.1时，前一百个数据的预测平方差和为：", regression.rssError(abY[0:99], yHat01.T)
print "局部线性回归，k=1  时，前一百个数据的预测平方差和为：", regression.rssError(abY[0:99], yHat1.T)
print "局部线性回归，k=10 时，前一百个数据的预测平方差和为：", regression.rssError(abY[0:99], yHat10.T)

#预测后一百个的y值
yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
print "\n局部线性回归后一百个数据集的误差情况："
print "局部线性回归，k=0.1时，后一百个数据的预测平方差和为：", regression.rssError(abY[100:199], yHat01.T)
print "局部线性回归，k=1  时，后一百个数据的预测平方差和为：", regression.rssError(abY[100:199], yHat1.T)
print "局部线性回归，k=10 时，后一百个数据的预测平方差和为：", regression.rssError(abY[100:199], yHat10.T)

ws = regression.standRegres(abX[0:99],abY[0:99])
yHat = mat(abX[100:199]) * ws
err = regression.rssError(abY[100:199], yHat.T.A) #标准线性回归的方法 看一下预测平方差和
print "\n使用标准线性回归方法的预测平方差和为：", err

# ########      第三个例子：Page145 预测鱼的年龄      #################
# 局部线性回归后一百个数据集的误差情况：
# 局部线性回归，k=0.1时，前一百个数据的预测平方差和为： 56.78868743048742
# 局部线性回归，k=1  时，前一百个数据的预测平方差和为： 429.8905618704059
# 局部线性回归，k=10 时，前一百个数据的预测平方差和为： 549.1181708828803
#
# 局部线性回归后一百个数据集的误差情况：
# 局部线性回归，k=0.1时，后一百个数据的预测平方差和为： 57913.51550155909
# 局部线性回归，k=1  时，后一百个数据的预测平方差和为： 573.5261441894984
# 局部线性回归，k=10 时，后一百个数据的预测平方差和为： 517.5711905381573
#
# 使用标准线性回归方法的预测平方差和为： 518.6363153245542


print "########      第四个例子：岭回归      #################"
abX, abY = regression.loadDataSet('abalone.txt')
ridgeWeights = regression.ridgeTest(abX,abY)
plot.plot3(ridgeWeights, titlename = u"第四个例子：岭回归求系数 系数ridgeWeights[0:30]的图形(30条) Page147")
print "ridgeWeights[0]:\n", ridgeWeights[0]

print "########      第五个例子：Page149 前向逐步线性回归      #################"
xArr, yArr = regression.loadDataSet('abalone.txt')
stageWiseWeights = regression.stageWise(xArr, yArr, 0.01, 200)   # 前向逐步线性回归的方法计算系数
plot.plot3(mat(stageWiseWeights), titlename =u'第五个例子：Page149 前向逐步线性回归  系数stageWiseWeights的图形 Page149')


xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regression.regularize(xMat)
yM = mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T) # 标准线性回归的方法计算系数
print " weights.T:", weights.T
plot.plot3(weights, titlename = u"标准线性回归 Page149 ")



end = time.clock()
print "\n这个程序运行的时间是：", end - start, "秒"

