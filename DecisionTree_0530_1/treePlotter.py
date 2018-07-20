
# coding=UTF-8

import matplotlib.pyplot as plt
#使用文本注解绘制树节点  决策树节点 叶节点
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  #sawtooth:锯齿状
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")      #箭头格式


#输入字典形式的树，输出叶子节点的数目
#每一个花括号都是字典
# 字典例如：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
#计算叶子节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]     #取字典的第一个[0]键值
    secondDict = myTree[firstStr]   #读取第一个键值对应的值作为第二个字典
    for key in secondDict.keys():   #遍历访问第二个字典的元素的键值，如果键值是字典类型，则递归访问该字典，否则就是叶子节点，统计数目+1
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

#计算字典的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():       #递归一次，深度+1，不是字典深度从1开始计数
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


#绘制节点 带箭头的注解（nodeTxt文本框的内容，centerPt子节点，parentPt父节点，nodeType点的类型（两种：决策节点和叶子节点））
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontsize=25 )

#在父子节点的中间位置添加简单的文本标签信息txtString
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]  #计算父子节点中间的x坐标
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]  #计算父子节点中间的y坐标
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=0, fontsize=35)

#
def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #计算叶子节点的数目 决定树的宽
    depth = getTreeDepth(myTree)    #计算树的深度 决定树的高
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)      #标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD     #减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

#
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #无需边框frameon
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))    #树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))   #树的深度
    plotTree.xOff = -0.5/plotTree.totalW    #追踪已绘制的节点位置，以及放置下一个节点的恰当位置
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


#def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()


#输出预先存储的树信息，避免每次测试代码都要从数据中创建树的麻烦
#列表里存放了两个字典/两棵树 i可以取值 0或1
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]




