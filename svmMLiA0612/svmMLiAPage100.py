# coding=UTF-8
#定义的函数有：
# def loadDataSet(fileName):# SMO算法中的三个辅助函数之一 加载数据
# def selectJrand(i, m):# SMO算法中的三个辅助函数之二  随机选择另一个数j
# def clipAlpha(aj, H, L):# SMO算法中的三个辅助函数之三  调整a的值，让其等于边界L或H
# def kernelTrans(X, A, kTup):  # 使用核函数
# class optStruct:  # 构建一个数据结构 保存重要的数据信息
# def calcEk(oS, k):  # 计算数据集中数据k的预测值与真实值之间的误差
# def selectJ(i, oS, Ei):  # 选择第二个alphas的值或者说是内循环的值，this is the second choice -heurstic, and calcs Ej
# def updateEk(oS, k):  ## 辅助函数 after any alpha has changed update the new value in the cache
# def innerL(i, oS):  # 优化alpha i 的值 优化成功return 1,失败则return 0
# def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # full Platt SMO
# def calcWs(alphas, dataArr, classLabels):  # 计算ws的值,使用alphas
# def testRbf(k1=1.3):  ## 例子二的测试（使用核函数）
# def img2vector(filename):# 将例如1_0.txt文档中的数字图像数据转化成一个行向量 　
# def loadImages(dirName): # 读取训练或者测试文件夹中的文档数据，将数据和标签存储到trainingMat, hwLabels中去
# def testDigits(kTup=('rbf', 10)):  # 第四个例子 111 手写数字识别


from numpy import *
from time import sleep

#svmMLiAPage100是使用完整的SMO算法和核函数实现的，svmMLiAPage96是使用简化的SMO算法实现的
################################################################################## 第二个例子　
# SMO算法中的三个辅助函数之一 加载数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat



# SMO算法中的三个辅助函数之二  随机选择另一个数j
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


# SMO算法中的三个辅助函数之三  调整a的值，让其等于边界L或H
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


#calc the kernel or transform data to a higher dimensional space
#@kTup[0] 元组的第一个数表示线性还是高斯核
#@kTup[1] 元组的第二个数表示高斯核函数计算公式中的代尔塔（周志华P128表6.1），高斯函数值跌落到零的速率
def kernelTrans(X, A, kTup):    #使用核函数
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':K = X * A.T   #linear kernel 一种是线性核
    elif kTup[0]=='rbf':    #一种是高斯核
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct: #构建一个数据结构 保存重要的数据信息
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters(参数)
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C  #权衡因子
        self.tol = toler    #松弛变量/容忍度
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2))) #第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值 first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m): #通过高斯核将低维数据转成高维数据
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup) #kTup[0]决定是线性还是高斯核；kTup[1]是高斯核函数值跌落到0的速率参数


def calcEk(oS, k):  #计算数据集中数据k的预测值与真实值之间的误差
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # 选择第二个alphas的值或者说是内循环的值，this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):    #选择合适的alphas值，以保证每次优化中采用最大步长
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 上面的方法找不到的话就随机另外再找一个alphas
        # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#辅助函数
def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek] #计算误差值并存入缓存当中，在对alpha值进行优化之后会用到这个值。


def innerL(i, oS):  #优化alpha i 的值 优化成功return 1,失败则return 0
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print "L==H"; return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0



def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

#计算公式是周志华P123式子6.9
#@alpha:       α
#@dataArr:     x
#@classLabels: y
def calcWs(alphas,dataArr,classLabels): #计算ws的值,使用alphas
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)   #计算公式是周志华P123式子6.9
    return w


############ 例子二的测试（使用核函数）
#程序的思路：
#通过训练数据集找到支持向量sVs以及他们的标签labelSV，
# 1.使用支持向量计算训练数据集的标签和正确分类率，
# 2.使用支持向量计算测试数据集的标签和正确分类率。
def testRbf(k1=1.3):    ## 例子二的测试（使用核函数）
    dataArr,labelArr = loadDataSet('testSetRBF.txt')    #这里加载的是新的测试数据集
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()    #将数据和标签都转化为列向量
    svInd=nonzero(alphas.A>0)[0]    #阿尔法值大于0的量就是支持向量，找到这些非零量的下标存放在svInd中
    sVs=datMat[svInd] #get matrix of only support vectors 通过下标找到支持向量的数据存放在sVs中
    labelSV = labelMat[svInd]   #通过下标找到支持向量的标签存放在labelSV中

    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))   #使用找到的支持向量来预测 测试数据的标签，周志华P127式子6.22
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b  #周志华P127 式子6.24
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    #预测值与实际值对比
    print "the training error rate is: %f" % (float(errorCount)/m)

    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)





########### 第四个例子:示例　手写数字识别问题回顾
#将例如1_0.txt文档中的数字图像数据转化成一个行向量 　
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

#读取训练或者测试文件夹中的文档数据，将数据和标签存储到trainingMat, hwLabels中去
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:    #这里只确定是否是二分类：数字9是-1类 数字1是1类（数据集中只有1和9两个数字）
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):   #      第四个例子 111 手写数字识别
    #读取训练数据集的数据找到其中的支持向量
    dataArr, labelArr = loadImages('/home/yzn/PycharmProjects/svmMLiA0612/digits/trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    #计算训练数据集的预测错误率
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)
    #计算测试数据集的预测错误率
    dataArr, labelArr = loadImages('/home/yzn/PycharmProjects/svmMLiA0612/digits/testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)

