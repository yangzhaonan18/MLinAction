# coding=utf-8

from math import log

#手动定义数据集的内容：最后一列是 类标签classLabel值，前面的两列是特征值，特征存放在label列表中
#这里label用来表示的内容和kNN中表示的有所不同：kNN中label用于表示每一行数据（样本/实例）的标签，DecisionTree中却用于表示特征feature。

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算香农熵（数据集的）：利用数据集的最后一列（classLabel）的值来计算数据集未分类前的香农熵，即原始熵（与前几列的特征值无关）
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}    #用字典来存放最后一列的类标签classLabel：出现的次数
    for featVec in dataSet:     #按行读取
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0    #如果查找到的键值在字典中不存在，就将其添加并将值赋值为0；否则值+1
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:    #根据labelCounts字典计算香农熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2 ,表达式的解释： s = s + (-p) 可以写成  s = s - p 即 s - = p
    return shannonEnt

#从数据集dataSet中划分（抽取）出axis列取值为value的数据：dataSet是数据集，axis是数据集（特征）的列，value是特征列的取值。
def splitDataSet(dataSet, axis, value):
    retDataSet = []     #retire：退休 退化
    for featVec in dataSet:     #按行读取
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]      # 取[axis]前面的部分，不含[axis]
            reducedFeatVec.extend(featVec[axis + 1:])       #取后面的部分，即除了[axis]列，该行数据的其他列都要存放在列表中。
            retDataSet.append(reducedFeatVec)       #append()是将列表整个添加，extend()是将类表中的元素单独添加
    return retDataSet

#选择最好的数据集划分方式，返回划分特征的下标[i]( int值 )
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  #特征的数量（最后一列(是classLabel)不是特征故需要-1处理） the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)   #数据集的原始熵
    bestInfoGain = 0.0      #最大信息增益值（熵的减小值=原始熵-划分后的熵）
    bestFeature = -1        #最佳特征
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # 将每一行example的[i]列全部存放在featList列表中  create a list of all the examples of this feature
        uniqueVals = set(featList)  # 用集和Set()去重   get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:    #计算[i]列划分数据的信息熵（每一列可以划分成多种，香农熵=每一种出来的概率乘以其熵 再求和）
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))        #出现的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     #一种的概率乘以其香农熵
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer（整数），即特征的下标[i]


#多数表决的方法：当数据集已经处理完了所有的属性，但是类标签依然不是唯一的，此时采用多数表决的方法决定叶子节点的类标签
# （理解为划分时特征不够用，划分不到底了，就这样处理）
#返回 类 字符串
def majorityCnt(classList):
    classCount={}   #字典：键值：值 = 类名称：出现的次数
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #降序
    return sortedClassCount[0][0]

#递归方法来创建树（搞懂具体的递归过程）
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #下面两个if是结束递归调用两种可能性 返回的类class
    if classList.count(classList[0]) == len(classList): #划分之后出现该分支下的所有实例都具有相同的分类，即划分好了，不用再划分，可以停止划分了。
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # 遍历完了所有划分数据集的属性，一行中只剩下一个数据（类标签）时结束划分。 stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    #输出是下标 整型的
    bestFeatLabel = labels[bestFeat]        #输出是特征名称 字符串
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  #将[bestFeat]那一列的特征值 都存放在featValues列表中
    uniqueVals = set(featValues)    #利用集合Set() 去重复的特征值
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up（弄乱，搞乱） existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)    #每一个value值创建一个树
    return myTree


# for  test  whether  it is  correct
# featLabels is list ,testVec is the choose list  to decided where to go
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]  #字典的第一个键值存放在firstStr中
    secondDict = inputTree[firstStr]    #第一个键值的值作放在字典secondDict中,方便后续的继续访问（递归调用）
    featIndex = featLabels.index(firstStr)  #根据键值firstStr在特征标签中找对应的下标featIndex
    key = testVec[featIndex]    #根据下标featIndex对应的键值key
    valueOfFeat = secondDict[key]   #在树的键值对应的值
    if isinstance(valueOfFeat, dict):   #如果键值对应的值是字典，则递归访问字典，否则输出 类标签
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

#使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
#使用pickle模块grab决策树 grab:提取/抓取
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)








