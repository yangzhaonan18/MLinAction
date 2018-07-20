#coding=UTF-8


from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


#划分数据集 大于特征特征值的作为右子树 小于等于以后作为左子树
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

#返回数据集的数据标签的 均值(叶子节点是常数模型时使用)
def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

#计算数据的 总体方差(叶子节点是常数模型时使用)
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]



##############################      预剪枝方法的树回归

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

#@ dataSet:数据集
#@ leafType：叶子节点模型，leafType叶子节点生成方式（regLeaf是常数的模型，regTrees是线性的模型）
#@ errType：误差计算
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val # 当不能再继续划分的时候， chooseBestSplit函数返回的特征就是none 返回的特征值就作为树的叶子节点
    # （常数模型时就是这些不可划分的的数据y标签的均值，线性模型就是这些数据的权重值ws）
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


##############################  后剪枝方法的回归树
#判断是否是树（字典）
def isTree(obj):
    return (type(obj).__name__ == 'dict')

#递归实现 返回叶子节点的均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

#@tree: 需要后剪枝的树
#@testData: 测试数据集
#剪枝处理
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree

#############################    模型树（叶子节点是线性模型非常数量）

#对输入的数据建立线性模型 返回参数:权重ws 添加常数偏量的数据X  标签向量Y
#数据说明：如果输入数据dataSet大小是（m,n）的，则前m-1列是属性值，最后一列是y标签值。处理数据是会给数据加一列常数1的偏量
#@ ws:线性模型的权重，是一个行向量（1,n） 注意：给数据添加了一列常数项1，就相当于加了一个常数b值，所以最后y=wsT*x的值就是估计值。
#@ X:加常数偏量之后的数据值，大小是（m,n）
#@ Y:y标签的值，大小是（m,1）
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


#返回线性模型的权重参数ws  (叶子节点是模型树（线性模型）时使用)
def modelLeaf(dataSet): #create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

#返回根据数据建立的线性模型计算出的预测误差和  (叶子节点是模型树（线性模型）时使用)
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))




#####################################   用树回归进行预测的代码   P174
#回归树（常数型） 叶子节点的值是浮点型常数（model = 常数）
def regTreeEval(model, inDat):  #返回回归树叶子节点的预测值（常数）
    return float(model) #属于这个叶子节点分类的输入的预测值 = model

#模型树（线性型） 叶子节点的值是权重ws向量（model = ws）
def modelTreeEval(model, inDat):    #返回模型树叶子节点的预测值（ws*X）
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model) #计算属于这个叶子节点的输入的预测值 =（X * model）

#预测树
#@ tree:树回归模型（是一个树/字典）
#@ inData:一个待预测的数据，这里是一个行向量（数据集的其中一行）
#@ modelEval:叶节点生成类型，默认是回归型（常数型）
def treeForeCast(tree, inData, modelEval=regTreeEval):  #这个函数预测的是一个数据行向量的值（inData是一个行向量,大小是(m,1)）
    #当前节点tree不是树的话,说明已经递归到了叶子节点，测试就可以返回叶子节点的预测值（回归树就放回常数项的值(预测值)，模型树就放回ws*X（预测值））
    if not isTree(tree): return modelEval(tree, inData) #当前树（当前节点节点）是叶子节点（非树），生成叶子节点
    #如果当前节点tree是树，需要分左右子树才能处理，
    # 满足 > 的条件的话，就在右子树这边处理，
    if inData[tree['spInd']] > tree['spVal']:   #非叶子节点，则对树进行切分
        #如果右子树也是一颗树的话，递归右子树
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval) #递归直至叶子节点
        #否则右子树就是叶子节点（回归树或者模型树），这个时候就可以放回预测的值了（返回常数项或者ws*X的值）
        else:
            return modelEval(tree['left'], inData)
    #小于等于的情况就是左子树了，遍历方式和右子树相同。
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)#递归直至叶子节点
        else:
            return modelEval(tree['right'], inData)


#创建预测树
#@ tree: 树回归模型，一个字典形式的树
#@ testData:测试数据集。利用树tree对测试数据集的每一行进行预测，返回测试数据集的预测值
def createForeCast(tree, testData, modelEval=regTreeEval):  #这个函数预测的是一个数据集的值（inData是一个数据集，大小是（m,n））
    m = len(testData)
    yHat = mat(zeros((m, 1)))    #初始化yHat的行向量各维度值为1
    for i in range(m):  #遍历所有的行/样本
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval) #利用树预测函数对测试集进行树构建过程，并计算模型预测值
    return yHat #返回预测值（列向量）


