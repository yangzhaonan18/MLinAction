# coding=UTF-8
#时间：2018年6月8日 下午
#名称：逻辑回归分类（监督分类算法） P80
#

from numpy import *
import matplotlib.pyplot as plt


#从文件件中读取数据 存放在两个矩阵中（数据 数据标签）          #读取数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines(): #按行读取
        lineArr = line.strip().split()  #将一行的三个数据生成一个列表
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])   #将常数1 列表的前两个数[0][1] 添加到数据矩阵中
        labelMat.append(int(lineArr[2]))    #将列表的最后一个数[2],添加到 标签列表中
    return dataMat,labelMat #返回 列表的列表（矩阵形式） 行列表

#计算西格玛函数的值
def sigmoid(inX):
   # inX = array(inX, dtype=float64) 这条语句加上去没有用
    return 1.0/(1+exp(longfloat(-inX)))    #输入inX是通常是一个列向量（列向量的每一个元素代表一个样本输入z）  
    # #overflow encountered in exp
'''
def sigmoid(inX):
    numlen = len(inX)
    for i in range(numlen):
        if inX[i] >= 0:
            inX[i] = 1.0/(1+exp(-inX[i]))  #输入inX是通常是一个列向量（列向量的每一个元素代表一个样本输入z）
        else:
            inX[i] = exp(-inX[i]) / (1 + exp(-inX[i]))
    return inX
'''

#111111111111111111111111111111111111111111111111111              #例子1：梯度上升算法的例子
#梯度上升算法（h和error都是向量，有矩阵的转置过程）求最佳参数/权重
def gradAscent(dataMatIn,classLabels):  #数据矩阵  数据类的列向量
    dataMatrix = mat(dataMatIn) #将数组装换成矩阵
    labelMat = mat(classLabels).transpose() #将行向量 转换成 列矩阵/向量
    m,n = shape(dataMatrix) #行m：数据个数，列n：表示每一个数据的属性数量
    alpha = 0.0001   #步长/学习速率
    maxCycles = 50000 #循环次数
    weights = ones((n,1))   #初始化 每个权重值为1  这里的权重也可以任意设置，只要迭代的次数足够多都能找到最佳权重
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)    #计算出每一个样本的 西格玛函数值（预测值）
        error = (labelMat - h)  #计算实际值与预测值之间的误差
        weights = weights + alpha * dataMatrix.transpose()*error ######梯度上升算法 迭代计算权重########上升求最大值 下降求最小值
    return weights  #返回500词迭代后的权重（列向量）
#'''

##2222222222222222222222222222222222222222222222222              #:例子2：随机梯度上升算法的例子
#改进点：
#1.每一次使用一个点的信息而不是整个数据集（降低时间复杂度）
#2.步长alpha是越来越小的而不是固定的常数（缓解数据波动或高频波动）
#3.随机选取样本来更新回归系数（减少周期性的波动）

#随机梯度上升算法（h和error都是数值，变量都是数组不是矩阵）求最佳参数/权重
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):    #修改过书上的程序，使算法在整个数据集上运行150次
        dataIndex = range(m)
        for i in range(m):   #运行一遍所有的数据，每一次只用一个数据
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration（迭代）, 但永远不会减到0.0001，保证新数据有一定的影响，
                                          # （alpha越来越小，即先粗调后微调）
            randIndex = int(random.uniform(0,len(dataIndex)))   #每一次的for都会将每个数据使用，每次使用一个数据。随机体现在
                                                                #数据使用的顺序上，程序运行多次时，每一次数据的使用顺序不相同，
            #                                                   #就会导致权重的计算结果不同，程序的其他部分是都相同。
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#画出决策边界（逻辑回归最佳拟合直线）  画图程序：利用数据画点 利用权重画直线   #作图
def plotBestFit(weights):
    #import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)    #将列表转换成数组
    n = shape(dataArr)[0]   #数据的行数 即样本数量
    xcord1 = []; ycord1 = []    #列用于存放点的坐标（两种类的点分开存放，两个属性x y的值也分开存放）
    xcord2 = []; ycord2 = []
    for i in range(n):  #将数据信息提取出来
        if int(labelMat[i]) == 1:   #将类为1的数据的两个属性值分别存放在 xcord1和ycord1列表中
            xcord1.append(dataArr[i,1]) # x1
            ycord1.append(dataArr[i,2]) # X2
        else:                       #将类为0的数据的两个属性值分别存放在 xcord2和ycord2列表中
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)   #只绘制一个图像
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')   #绘制类1的点 形状设置成方形  散点图
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')    #绘制类0的点 默认形状是 圆形 散点图
    x = arange(-3.5, 3.0, 0.1)  #绘制直线上的点的x坐标的范围 步长为0.1 （下一步就是 已知直线方程和直线上的点x坐标，求y坐标）
    y = (-weights[0]-weights[1]*x)/weights[2] ######将 z=w0*x0+w1*x1+w2*x2=0 设置为两个类别的分界线（西格玛函数图像决定取0的），
                                              # (x0取常数1，x1就是x轴，x2就是y轴)所有的分界点都在这条直线上，解出y=X2=(-w0-w1*x1)/w2
    ax.plot(x, y, 'ro-')  #r红的 o圆的 #### 绘制 最佳拟合直线(实际是绘制相邻点和点连接在一起的直线图，因为都在一条之间上，所以就成了直线图了)
    plt.xlabel('X1  (X0=1)');plt.ylabel('X2')   #绘制 轴的名称
    plt.show()  #绘图的标准结尾 格式要求


####333333333333333333333333333333333333333333333333            #例子3：从疝气病症预测病马的死亡率
#1 处理缺失值（数据是相当贵的）的方法：舍弃数据，取特殊值，取平均值，取相识样本的值，利用机器学习算法预测缺失值
#2 程序提供的数据是完整的，是处理过缺失值之后的数据
#预测分类
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

#1读取训练数据信息 2根据训练数据信息计算权重 3读取测试数据，预测分类，计算错误率 4 返回错误率
def colicTest():    #colic 绞痛（马得病）　测试函数
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t') #按\t字符 提取一行的数据 存放在列表中
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))  #将每一行的所有21个属性数据存放在 lineArr列表中
        trainingSet.append(lineArr) #列表的列表 存放所有的属性数据信息
        trainingLabels.append(float(currLine[21]))  #存放最后一列的 label信息
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)    #计算训练数据集的 权重系数
    errorCount = 0  #错误数
    numTestVec = 0.0    #用于统计总的测试数据数量
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): #  训练样本 0到20 有21个属性值
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)    #错误率
    print "the error rate of this test is: %f" % errorRate
    return errorRate

#多次运行上面的程序   计算平均错误率
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest() #错误率之和
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))
