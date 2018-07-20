#coding=UTF-8
#程序名称：第十一章，使用Apriori算法进行关联分析（聚类）
#程序说明：
#程序功能：聚类根据两个方面来处理，1频繁项集，2关联规则；两者的度量方式分别是支持度和可信度（可信度等于两个支持度的商）
#创建时间：2018年7月4日 22:08
#程序进度：7月5日晚初步完成程序分析，对程序中关系规则的建立不是特别流畅（不能将书上的那幅图和程序很好的对接，关联起来，不清楚具体的遍历流程，程序运行的起始点和截止点）
# 频繁项集：
# 关联规则：

 # import numpy
import apriori
#
# dataSet = apriori.loadDataSet() #加载数据
# C1 = apriori.createC1(dataSet) #创建集合C1
# D = map(set, dataSet) #数据集dataSet用集合来表示
# L1, suppData0 = apriori.scanD(D, C1, 0.5) #根据C1生成K1
# # L, suppData = apriori.apriori(dataSet) #L是所有的频繁项集
# # L的值是:[[frozenset([1]), frozenset([0]), frozenset([3]), frozenset([5])], [frozenset([0, 1]), frozenset([0, 3]), frozenset([1, 3]), frozenset([3, 5]), frozenset([0, 5]), frozenset([1, 5])], [frozenset([0, 1, 3]), frozenset([0, 1, 5]), frozenset([1, 3, 5]), frozenset([0, 3, 5])], [frozenset([0, 1, 3, 5])], []]
#
# # print "\n数据集dataSet是：\n", dataSet
# # print "\n只含一个元素的项集C1是：\n", C1
# # print "\n数据集dataSet用集合表示成D：\n", D
# # print "\n满足支持度的集合L1（C1-->L1）是：\n", L1
# # print "\nL:\n", L
# # print "\nL[0]:\n",L[0]
# # print "\n apriori.aprioriGen(L[0], 2):L1生成的C2是\n", apriori.aprioriGen(L[0], 2)
# # print "\n apriori.aprioriGen(L[1], 3):L2生成的C3是\n", apriori.aprioriGen(L[1], 3)
#
# # ###################   从频繁项集中挖掘关联规制
# L, suppData = apriori.apriori(dataSet, minSupport=0.5)
# print "L:", L
# rules = apriori.generateRules(L, suppData, minConf=0.7)
# print "\nminConf = 0.7时：\n", rules
#
# # rules = apriori.generateRules(L, suppData, minConf=0.5)
# # print "\nminConf = 0.5时：\n", rules


############################ 第二个例子 投票记录的事务数据集
from votesmart import votesmart
votesmart.apikey  = ''
bills = votesmart.votes.getBillsByStateRecent()
for bill in bills:
    print bill.till, bill.billId
