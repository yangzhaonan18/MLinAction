#coding=UTF-8
#定义的函数有




from numpy import *
################### 第一个例子（线性函数）的函数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 # -1是为了添加数据时最后一列的ylabel不被添加进去。get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))   #只添加前两列的数据
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1])) #添加最后一列的数据作为 ylabel
    return dataMat,labelMat

#@ xArr: (m, 2)的矩阵 第一列是常数项1 第二列是原始数据的x
#@ yArr: (m, 1)的矩阵 数据的y值（类标签）
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


###################### 第二个例子（局部加权回归）的函数
#@ testPoint    :当前的点xi,
#@ xMat :整个数据集 x
#@ yMat :y
#@ k :高斯核函数的中的参数k（k值决定对附近的点赋予多大的权重，k值越大就会对周围点赋值相同的权重，差别不大，反之则权重差别很大）P142
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T   #转换成列向量
    m = shape(xMat)[0]
    weights = mat(eye((m))) #单位矩阵
    #对每一个输入的testPoint 都要计算出一个主对角矩阵weights(只有主对角非零)
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #行向量减去行向量 表示求未知点testPoint与已知数据集xArr中每一个点之间的差别/距离
        # （差别diffMat越小表示两者之间越相似，diffMat*diffMat.T的值越小， weights[j,j]的值越接近1（越大））
        #print j,"  diffMat的值是：\n", diffMat
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) # 行向量乘以行向量的转置得一个数值
    #print "###########weights最后是:\n",weights
    xTx = xMat.T * (weights * xMat) # 计算公式在Page142
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))    #利用计算出的主对角矩阵weights 大小是(n+1,1)。
    # 这个行向量权重是一个数据点testPoint的权重，每一个数据行都会计算出一个行权重
    return testPoint * ws   #（1,n+1)*(n+1,1) 返回当前输入数据行testPoint 的局部线性回归预测值



########      第三个例子：Page145 预测鱼的年龄      #################
#计算testArr数据集中每一行的预测值 存放在yHat[i]中
def lwlrTest(testArr,xArr,yArr,k = 1.0):  #循环遍历所有数据点并将lwlr应用于每个数据点loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTetPlot(xArr,yArr,k = 1.0):  #对xArr进行排序后计算yHat(预测值) same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #计算（预测值和实际值之间的）平方差和 yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


########      第四个例子：Page147 岭回归      #################
def ridgeRegres(xMat, yMat, lam = 0.2):   #岭回归 用于计算回归系数
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat) #如果矩阵是非奇异矩阵就计算回归系数并返回
    return ws


def ridgeTest(xArr, yArr):  #用于在一组lam上测试结果
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar   #标准化:所有特征都减去各自的均值并除以方差
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat #计算出30个不同的lam值对应的回归系数ws





########      第五个例子：Page149 前向逐步线性回归      #################

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):    #前向逐步线性回归的方法计算系数
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T #找到第i行的每个列j的最小情况（+1 -1）
    return returnMat

