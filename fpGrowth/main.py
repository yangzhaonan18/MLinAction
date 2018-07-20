#coding=UTF-8
#程序名称：第十二章 使用FP-growth算法来搞笑发现频繁项集
#程序说明：参考理解：https://www.cnblogs.com/pinard/p/6307064.html
#程序功能：
#创建时间：2018年7月5日 星期五
#程序进度：
# 2018年7月5日 星期五 实现程序运行，不含例子
# 2018年7月9日 星期一 完成程序分析，不含例子
import time
start = time.clock()
import fpGrowth
#
# #############################################  Page 226
# #(类实例化成rootNode1)根节点添加元素： 名称 计数
# rootNode1 = fpGrowth.treeNode('pyramid', 9, None)
# #添加根节点的子节点：名称 计数
# #子节点存放在字典中，字典这个属性和其他的名称，计数是并列关系（字典是类/实例.字典）
# #将新的树/字典作为根节点子节点的值存放
# #子节点存放在字典中，这里的第一个eye是子节点的键；键值是这个子节点（第二个eye是这个子节点的内容的名称）
# rootNode1.children['eye'] = fpGrowth.treeNode('eye', 13, None)
# rootNode1.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)#继续添加根节点的子节点：名称 计数
# rootNode1.children['eye'].children['books'] = fpGrowth.treeNode('book', 23, None) #添加第三层节点
# print "\n\n手动添加方法生成的FP树是："
# print "\n这棵树是（含2个节点）：\n", rootNode1.disp() #调用类的方法来打印树的内容 在屏幕上显示出树时，子节点要在父节点的基础上缩进
#
#
# #############################################  Page 230
# simpDat = fpGrowth.loadSimpDat() #加载数据集 列表
# initSet = fpGrowth.createInitSet(simpDat) #将数据集列表 转化为 字典形式存储（事务：频数=1）
# myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3) #根据初始化的数据字典和最小支持度，生成FP树和头指针列表
# print "\n加载数据集中的内容，原始数据列表内容是：\n", simpDat
# print "\n原始数据转换成数据字典形式后是：\n", initSet
# print "\n显示生成的FP树是：\n", myFPtree.disp()
#
# #############################################  Page 232
# print "\n从FP树中找到相应元素的前缀路径的集合（条件模式基）是："
# print "\nx的前缀路径:\n", fpGrowth.findPrefixPath('x', myHeaderTab['x'][1]) #输入待查找的元素和(头指针列表指向的FP树中相同元素的)第一个元素
# print "\nz的前缀路径:\n", fpGrowth.findPrefixPath('z', myHeaderTab['z'][1])
# print "\nr的前缀路径:\n", fpGrowth.findPrefixPath('r', myHeaderTab['r'][1])
# print "\n","#" * 100
#
# #############################################  Page 234
# freqItems = []
# #@ myFPtree：FP树的根节点
# #@ myHeaderTab：头指针表（字典）
# #@
# #递归查找频繁项集（每一个频繁项集都要创建一颗FP树）
# fpGrowth.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems) #发现频繁项集并显示前缀路径
# print "\nfreqItems:\n", freqItems #显示频繁项集

####   例子:在Twitter源中发现一些共现词
# lotsOtweets = fpGrowth.getLotsOfTweets('RIMM')
# print lotsOtweets



parsedDat = [line.split() for line in open('kosarak.dat').readlines()] #加载数据存放在列表中
initSet = fpGrowth.createInitSet(parsedDat) #数据存放形式从列表转化成字典
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 100000) #创建FP树和头指针表
myFreqList = []
fpGrowth.mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList) #生成频繁项集
print "\nlength of 'myFreqList' is:", len(myFreqList)
print "myFreqList is：", myFreqList






end = time.clock()
print "The run time of the program is：",end-start, "seconds"









