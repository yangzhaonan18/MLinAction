#coding=UTF-8
#程度名称：第9章:树回归
#程序说明：树回归根据叶子节点的模型分为两种：1常数型即回归树；2模型树即线性模型
#程序功能：1根据训练数据生成树（分类的模型），2使用树（分类模型）预测测试数据集，
# 3计算预测值与真实值之间的相关系数，比较两个算法的效果（模型树优于回归树）
# 4使用图形用户界面GUI来显示真实数据和预测值，方便调参
#程序时间：2018年7月2日开始，7月4日下午15:20完成出GUI之外程序备注分析。

# treeExplore.py 图形用户界面GUI

import numpy
import regTrees

testMat = numpy.mat(numpy.eye(4))
testMat[2, 1] = 5
print testMat
rMat0, lMat0 = regTrees.binSplitDataSet(testMat, 1, 0.5)
print "\n特征1 小于等于0.5的数据是：\n", rMat0
print "\n特征1 大于0.5 部分的数据是：\n", lMat0

###################  预剪枝方法的 树回归 P164
myDat1 = regTrees.loadDataSet('ex00.txt')
myMat1 = numpy.mat(myDat1)
trees1 = regTrees.createTree(myMat1,ops=(1,4))
print "预剪枝方法生成树回归是：", trees1
trees1 = regTrees.createTree(myMat1,ops=(1,4))

##################  后剪枝方法的 树回归 P169
myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = numpy.mat(myDat2)
trees2 = regTrees.createTree(myMat2,ops=(10,10))
print "\n后剪枝之前的树trees2是：\n", trees2
myDat3 = regTrees.loadDataSet('ex2test.txt')
myMat3 = numpy.mat(myDat3)
trees3 = regTrees.prune(trees2, myMat3)
print "\n后剪枝之后的树trees3 是:\n", trees3

##################  叶子节点是模型树（线性模型）P172
myMat4 = numpy.mat(regTrees.loadDataSet('exp2.txt'))
#Page170 是书上的模型树即叶子节点是线性模型， modelLeaf函数返回的是线性的权重ws，modelErr函数返回的误差的值
trees4 = regTrees.createTree(myMat4, leafType = regTrees.modelLeaf, errType = regTrees.modelErr, ops=(1,4))
print "\n叶子节点是模型树的树回归：\n", trees4



################## 树回归与标准回归的比较 P174
###回归树的预测情况和相关系数的计算
trainMat = numpy.mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt')) #加载训练数据
testMat = numpy.mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))   #加载测试数据
myTree = regTrees.createTree(trainMat, ops = (1, 20))   #获得训练数据的树：树回归（叶子节点是常数项）
#@ testMat[:,0] ：待预测的数据集（读取的数据集中除了最后一列标签之外的其他数据）
yHat  = regTrees.createForeCast(myTree, testMat[:,0])   #利用训练数据集生成的树myTree来预测测试数据的testMat的值 存放在列向量yHat中。
#输入的数据是列向量的话，rowvar应该赋值为False
#相关系数是一个方阵，取其中的[0,1],0行1列的值，即是两者的相关系数
corrValue = numpy.corrcoef(yHat, testMat[:,-1], rowvar = False)[0,1] # 计算预测值与真实值之间的corrcoef:相关系数
print "\n回归树（叶子节点是常数项）时，数据的相关系数是:", corrValue

###模型树的预测情况和相关系数的计算
myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, (1, 20))
yHat = regTrees.createForeCast(myTree , testMat[:,0], regTrees.modelTreeEval) #testMat[:,0]:待分类的测试数据集
corrValue = numpy.corrcoef(yHat, testMat[:,-1], rowvar = False)[0,1] #testMat[:,-1]:实际的数据集
print "模型树（叶子节点是线性模型）时，数据的相关系数是:", corrValue
print "\n结果对比如下："
print "从相关系数来看，模型树优于回归树"
print "说明：相关系数最大值为1，所以预测数据与真实数据的相关系数越接近1，表示预测结果越好。"

###标准线性回归的预测情况和相关系数的计算
print "\n标准线性回归（上一章）的预测情况"
ws, X, Y = regTrees.linearSolve(trainMat)   #生成线性回顾模型的权重ws
print "回归系数ws.T的值是：", ws.T,"(第一个量是常数偏量)"
for i in range(numpy.shape(testMat)[0]):  #计算预测值（列向量）
    yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0]  #ws（列向量）的第一个数据ws[0,0]是常数偏量值。
    # 这里的测试数据集值testMat是未添加常数列1的，所以这个式子不是一个通用的计算方法，只适用于只有一个特征的数据集
corrValue = numpy.corrcoef(yHat, testMat[:,1], rowvar = False)[0,1] #计算相关系数
print "标准线性回归的相关系数是:", corrValue
print "\n结果对比如下："
print "模型树 优于 回归树 优于 标准线性回归（上一章）"

# 程序运行情况如下：
# 回归树（叶子节点是常数项）时，数据的相关系数是: 0.9640852318222141
# 模型树（叶子节点是线性模型）时，数据的相关系数是: 0.9760412191380623
#
# 结果对比如下：
# 从相关系数来看，模型树优于回归树
# 说明：相关系数最大值为1，所以预测数据与真实数据的相关系数越接近1，表示预测结果越好。
#
# 标准线性回归（上一章）的预测情况
# 回归系数ws.T的值是： [[37.58916794  6.18978355]] (第一个量是常数偏量)
# 标准线性回归的相关系数是: 0.9434684235674758
#
# 结果对比如下：
# 模型树 优于 回归树 优于 标准线性回归（上一章）


#####################   使用Python的Tkinter库创建GUI
