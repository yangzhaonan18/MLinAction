# coding=UTF-8


from numpy import *
from time import sleep

#svmMLiAPage100是使用完整的SMO算法和核函数实现的，svmMLiAPage96是使用简化的SMO算法实现的
################################################################################## 第一个例子　　二维数据　使用支持向量机实现
# SMO算法中的三个辅助函数之一 加载数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])]) #两个元素看成一个整体添加
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat #　数据点　　数据标签

# SMO算法中的三个辅助函数之二  随机选择一个不等于i的j
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

#11111111111111111111111111111111     简化版的SMO算法
#@dataMatIn  ：数据 asdf 阿斯顿发asdf
#@classLabels：标签asdf阿斯顿发asdf 
#@C          ：权衡因子 规定C>0（增加松弛因子而在目标优化函数中引入了惩罚项）当距离大于等于１时，惩罚项为０；当距离小于１时惩罚项>0
#@toler      ：容错率/松弛变量 （就是书上的反3符号）
#@maxIter    ：最大迭代次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn);labelMat=mat(classLabels).transpose() #将列表形式转为矩阵或列向量形式
    b=0
    m,n=shape(dataMatrix)    #初始化b=0，获取矩阵行列
    alphas=mat(zeros((m,1)))    #新建一个m行1列的向量
    iter=0   #迭代次数为0
    while(iter<maxIter):
        alphaPairsChanged=0   #改变的alpha对数
        for i in range(m): #遍历样本集中样本
            fXi=float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T))+b #计算支持向量机算法的预测值
            # #multiply 列向量乘以列向量　对应相乘相加的同样大小的列向量
            Ei=fXi-float(labelMat[i])   #计算预测值与实际值的误差
            #如果不满足KKT条件，即labelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            #and alpha<C 或者labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if((labelMat[i]*Ei<-toler)and(alphas[i]<C)) or ((labelMat[i]*Ei>toler)and(alphas[i]>0)):
                j=selectJrand(i,m)      #随机选择第二个变量alphaj
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T))+b  #计算第二个变量对应数据的预测值
                Ej=fXj-float(labelMat[j])  #计算与测试与实际值的差值
                alphaIold=alphas[i].copy()  #记录alphai和alphaj的原始值，便于后续的比较
                alphaJold=alphas[j].copy()
                if(labelMat[i]!=labelMat[j]): #如何两个alpha对应样本的标签不相同
                    L=max(0,alphas[j]-alphas[i])  #求出相应的上下边界
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H: print("L==H");continue
                #根据公式计算未经剪辑的alphaj
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T- dataMatrix[i,:]*dataMatrix[i,:].T- dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:print("eta>=0");continue  #如果eta>=0,跳出本次循环
                alphas[j] -= labelMat[j] * (Ei-Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #如果改变后的alphaj值变化不大，跳出本次循环
                if(abs(alphas[j]-alphaJold)<0.00001):print("j not moving enough");continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])    #否则，计算相应的alphai值
                #再分别计算两个alpha情况下对于的b值
                b1 = b-Ei - labelMat[i] * (alphas[i]-alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]* \
                     (alphas[j]-alphaJold) * dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej - labelMat[i] * (alphas[i]-alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]* \
                     (alphas[j]-alphaJold) * dataMatrix[j,:]*dataMatrix[j,:].T
                if(0 < alphas[i]) and (C > alphas[i]):b=b1  #如果0<alphai<C,那么b=b1
                elif (0 < alphas[j]) and (C > alphas[j]):b=b2   #否则如果0<alphai<C,那么b=b1
                else: b = (b1 + b2) / 2.0   #否则，alphai，alphaj=0或C
                alphaPairsChanged += 1    #如果走到此步，表面改变了一对alpha值
                print "iter: %d i:%d,paird changed %d" % (iter,i,alphaPairsChanged)
        if(alphaPairsChanged==0):iter+=1    #最后判断是否有改变的alpha对，没有就进行下一次迭代
        else:iter=0     #否则，迭代次数置0，继续循环
        print("iteration number: %d" %iter)
    return b,alphas     #返回最后的b值和alpha向量


