# coding=UTF-8

#程序名称:k-Means 非监督聚类算法（非监督的特点 没有打标类，没有训练数据这一步，直接测试算法）
#程序内容：
#程序功能：
#程序说明：
#时间：
#进度：2018年6月13日（星期三）下午看课本，计划完成分析程序代码，能跑起来。
## 2018年6月17日（星期一）完成三个例子的调试。

import  kMeans
import  plotMap
import  kMeansEx
from numpy import *

##############################      #k均值聚类算法         ############################
#例子１：k均值聚类算法
datMat = mat(kMeans.loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans.kMeans(datMat, 4)  #k均值聚类算法
print "\n\n\n###############################         第一个例子： K均值算法     ###############################"
print "myCentroids 随机产生的质心点坐标:\n", myCentroids
print "clustAssing 分配结果簇评估矩阵（0列是簇类，1列是与所属簇心的距离平方）:\n", clustAssing
plotMap.plotCenter(array(datMat), myCentroids, clustAssing)#sudo cp ~/arial\ unicode\ ms.ttf /usr/share/fonts/arial\ unicode\ ms.ttfclustAssing)
print '\n第一个图是Ｋ均值算法：'


###############################       #二分k-均值聚类算法          #############################
#例子２：二分k-均值聚类算法
datMat3 = mat(kMeans.loadDataSet('testSet2.txt'))
centList, myNewAssments = kMeans.biKmeans(datMat3, 3)    #二分k-均值聚类算法
print "###########################   　第二个例子：二分K均值算法       ################################"
print  "centList:\n", centList
print  "myNewAssments:\n", myNewAssments
plotMap.plotCenter(array(datMat3), centList, myNewAssments)
print '\n第二个图是二分k均值算法：'


######################    #二分k-均值聚类算法 API接口读取地址坐标      ##########################
#第三个例
print "\n#####################            例子3：对地图上的点进行聚类：       #####################"
xAddressList, yAddressList, cityList  = kMeansEx.massPlaceFind('placeAddress.txt', 'placeName.txt')

kMeansEx.clusterClubs(xAddressList, yAddressList, cityList, 3, 'placeAddress.txt')
