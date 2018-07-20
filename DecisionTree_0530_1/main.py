# coding=utf-8

#程序说明
#标题：决策树的生成和绘制
#内容：1.读取文档数据生成数据集，2.利用数据集生成决策树，3.绘制决策树的图形
#时间：2018年5月30日 6月6日添加备注

import trees
import treePlotter
#11111111111111111111111111111111111111111111111111111
#利用手动创建的数据集生成树，绘制树的图形，测试程序过程步骤
#输出手动创建的数据集，计算香农熵
myDat,labels=trees.createDataSet()
print "myDat 数据集是：",myDat
print "\nlabels 标签是:",labels
rawCalc =trees.calcShannonEnt(myDat)
print "\ncalcShannonEnt(myDat) 数据集的原始熵是：",rawCalc
print "\ntrees.splitDataSet( myDat,1,1)将数据集的按 特征[1]=1(即 flippers==1) 提取出来的矩阵是:",trees.splitDataSet(myDat,1,1)
#
bestLabel = trees.chooseBestFeatureToSplit(myDat)
print "\nchooseBestFeatureToSplit(myDat) 数据集的bestLabel最好特征的[下标]是：",bestLabel,"\tlabels[bestLabel]最好特征是：",labels[bestLabel]
#
myTree = trees.createTree(myDat,labels)
print "\ntrees.createTree(myDat,labels) 根据数据集创建的树是:", myTree
#读取预先存储的树[0] 并绘制图形
print "\n读取预先存储的树[0] 并绘制出第一个图形:"
myTree0 = treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree0)
#读取预先存储的树[1] 并绘制图形
print "\n读取预先存储的树[1] 并绘制出第二个图形:"
myTree1 = treePlotter.retrieveTree(1)
treePlotter.createPlot(myTree1)

#change one date in "no surfacing"
#and print
'''
myTree['no surfacing'][3] = 'maybe'
print('after change is:')
print myTree
treePlotter.createPlot(myTree)rag
'''

#22222222222222222222222222222222222222222222222222222
#根据文档中的数据创建树，绘制树的图形
fr = open('lenses.txt')
lenses = [ inset.strip().split('\t') for inset in fr.readlines()]   #按行读取，每一行生成一个列表，将每一个行/列表 添加到 lenses 中（.split('\t')每一行按照\t划分成列表）
print "\n按列表格式输出lenses.txt文档中的lenses镜头数据集:"
for lense in range(len(lenses)):
    print lenses[lense]

lensesLabels = [ 'age','prescript','astigmatic','teatRate']
print "\nlenses.txt数据集的标签lensesLabels是：",lensesLabels

lensesTree = trees.createTree(lenses,lensesLabels)
print "\n根据lenses.txt文档的数据集生成的树ensesTree是： ", lensesTree

print "\n根据lenses.txt文档的数据集生成的树 绘制出第三个图形:"
treePlotter.createPlot(lensesTree)
