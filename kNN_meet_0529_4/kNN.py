
# coding=UTF-8

from numpy import *
import operator


#读取文件转换成矩阵：输出矩阵的列为3，即数据有三个特征。label以行向量的形式输出
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines ,3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index ,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat ,classLabelVector             #返回数据矩阵 和 类标签向量

#归一化处理;输出归一化后的矩阵，范围，最小值
def autoNorm(dataSet):
    minVals = dataSet.min(0)                #(0)表示按列取 每一列的最小值， minVals 的大笑是 3*1的
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]                    #数据矩阵行的值是 m
    normDataSet = dataSet - tile(minVals, (m, 1))     #tile()的用法，将minVal列表/行向量看成一个整体 复制成m行1列的矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals

#k-Nearest Neighbors 的算法：
def classify0(inX, dataSet, labels, k):                 #inX 是一个行向量/数据/样本/待估计的输入信息
    dataSetSize = dataSet.shape[0]                      #行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      #将inX 复制成和测试数据集dataSet一样的大小,即复制成m行，1列
    sqDiffMat = diffMat**2                              #对矩阵中的每一个元素求 2次方
    sqDistances = sqDiffMat.sum(axis=1)                 #axis=1:按行求 和
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()            #返回的是排序（默认是升序）后元素的索引值/下标，例如sortedDistIndicies[0] 就是distances数组中的最小值
                                                        #距离最小的值  就是最像是的kNN中需要的k中的第一个
    classCount={}                                       #字典
    for i in range(k):                                  #将距离最小的前K个值的label 用字典来统计出现的次数，键值为label,值为k次中出现的次数
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1     #在字典classCount 中找为voteIlabel 的键值，如果不能找到就将键值：值添加进取，值取0，如果能找到这个键值，就将其值+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
                                                        # 将classCount按每个元素（每一个 键值：值 看成一个整体排序）按key的取值排降序reverse=true，key取.itemgetter(1) 元素中的第二个，即按值排序

    return sortedClassCount[0][0]                       #返回字典的[o][o]的值，即第一个元素的第一个键值（K次中出现的次数最多的label 就是预测结果）

##########################################################
def datingClassTest():                                  #分类器的测试代码，前hoRatio的数据作为测试集，后面的其他数据作为训练集
    hoRatio = 0.10  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)                     #测试集的大小
    errorCount = 0.0                                   #错误率
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)  #第i行的数据作为测试，后面行的数据作为训练集，训练集的label,k=3
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0    #若不相同，则统计错误的次数
            print "line %d,the classifier came back with: %d, the real answer is: %d" % (i, classifierResult, datingLabels[i])  # 输出预测值和真实值
    print "预测出错的次数是", errorCount
    print "预测集的大小是",numTestVecs
    print "the total error rate is: %d/%d = %f" % (errorCount,numTestVecs,errorCount / float(numTestVecs)),"\n" #预测的错误率



