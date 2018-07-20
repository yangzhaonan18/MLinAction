
# coding=utf-8

from numpy import *
import  operator
from os import listdir

#将32*32像素的图片（图片和每一行只有0/1）转化成1024的行向量
def img2vector(filename):
    returnVect = zeros((1 ,1024))
    fr = open(filename)
    for i in range(32):   # i是行，j是列
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0 ,32*i+j] = int(lineStr[j])   #按行从左到右的读取并存放在行向量returnVect中
    return returnVect

#kNN的算法 注意：输入inX是一个行向量/一个待分类的行向量，长度任意 ，函数返回 类label
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#手写数字识别系统的测试代码，
#trainingDigits中共约2000个例子，testDigits中约有900个测试例子，每一个例子是一个字（32*32的图）文档名的格式：9_200.txt
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')    # load the training set 将文件夹中的文档的名称作为一个个元素，存放在列表中
    m = len(trainingFileList)   #文件名列表的长度m，即训练集的大小 m=2000
    trainingMat = zeros((m ,1024))
    for i in range(m):      ####读取trainingDigits文件夹中文档的信息并存放在trainingMat矩阵中
        fileNameStr = trainingFileList[i]   #读取文件名
        fileStr = fileNameStr.split('.')[0]  # take off .txt   取文档名称中.前面(0)的字符串作为fileStr. 【0】表示前面，【1】表示后面（0表示行，1表示列）
        classNumStr = int(fileStr.split('_')[0])    #从文档名称中提取文档内容表示的数字
        hwLabels.append(classNumStr)    #将文档名称中的数字 作为 真实的label标签
        trainingMat[i ,:] = img2vector('trainingDigits/%s' % fileNameStr)   #将每一个文档转成一个行向量后 存放在训练矩阵中

    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):      ##读取testDigits文件夹中文档的信息，对每一个输入求kNN算法，将预测值与真实值对比，统计预测的错误率
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 1)     #预测 测试数据的label
        if (classifierResult != classNumStr):
            errorCount += 1.0     #统计预测错的数量
            print "in",testFileList[i],"the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)  # 预测的label和真实的label(从文档名称中提取)
    print("\nthe total number of errors is: %d" % errorCount)
    print("the total test number is: %d" % mTest)
    print("the total error rate is: %d/%d = %f" % (errorCount,mTest, errorCount/float(mTest)))
    print("the total train number is: %d" % m)









