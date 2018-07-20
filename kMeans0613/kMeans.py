# coding=UTF-8

from numpy import *

#k均值聚类算法的三个支持函数 其一 加载数据函数
def loadDataSet(fileName):  # general function to parse tab -delimited floats 通用函数 解析 制表符 分隔的浮点
    dataMat = []  # assume last column is target value假定最后一列为目标值
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float() 将每一行的数据映射成float型 ？？？？？？？
        dataMat.append(fltLine)
    return dataMat

#k均值聚类算法的三个支持函数 其二 计算欧氏距离
def distEclud(vecA, vecB):  #计算两个点（行向量）之间的距离
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)

#k均值聚类算法的三个支持函数 其三 根据数据集随机生成k个质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]   #列的数目：表示点的属性个数
    centroids = mat(zeros((k, n)))  # create centroid mat 创建质心矩阵
    for j in range(n):  # create random cluster centers, within bounds of each dimension 在每个维度的界限内，创建随机簇中心 （每一次循环创建随机质心的一个属性量）
        minJ = min(dataSet[:, j])   #求J列（J属性）的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))    #random.rand(k, 1)产生一个k行1列的数组，即产生一个属性J的随机值，K行表示，有K个点
    return centroids    #返回含k个点坐标的点矩阵（n维）    K个随机质心


##########################   第一个例子 k均值聚类算法    #######################################
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):    #数据集  K个质心 默认距离函数 默认k个随机点生成函数
    m = shape(dataSet)[0]   #行数即数据集点的数目
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points 创建一个和数据集一样长的矩阵用来存储该数据的两个信息（存储数据的分类 和 该数据与所属簇心的距离平方）
    # to a centroid, also holds SSE of each point 对质心来说，同事也存放每个点的SSE值（距离）
    centroids = createCent(dataSet, k)  #  centroids 质心  调用函数创建随机质心
    clusterChanged = True   # 表示 flag
    num = 0
    while clusterChanged:
        num += 1
        #print num, num, num, num, num, num, num, num, num, num   #用于测试程序便利的次数
        #print 'asddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'
        clusterChanged = False
        for i in range(m):  #
            #print '对每一个数据点i对每一个数据点i对每一个数据点i对每一个数据点i %d' % i
        # for each data point assign it to the closest centroid 将每一个数据点分配给最近的质心
            minDist = inf   #inf 表示正无穷大 -inf 表示负无穷大
            minIndex = -1
            for j in range(k):  #
                #print '对每个随机簇心j对每个随机簇心j对每个随机簇心j对每个随机簇心j对每个随机簇心j ：%d ' % j
                distJI = distMeas(centroids[j, :], dataSet[i, :])   #计算数据点i到簇心j之间的距离
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j    #找到离数据点最近簇心J 和 距离平方minDist
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            #在两次for循环中，只要出现了一次点i的簇类发生变化就要再执行一次while##########要重点理解这句话！！！！！！！！
            #直到所有的点不需要分配时
            #如果这个点的需要重新分配簇心J的话,继续下一个点的分配？？？？？这样理解？吗 直到所有的点都分配好了簇心
            clusterAssment[i, :] = minIndex, minDist ** 2   #存放该点的簇心类别和误差（距离平方）
        #for循环之后已经找到了所有点所属的类别和与中心的误差
        #print "\ncentroids 输出当前使用的 随机生成的K个质心:\n", centroids     #打印质心 书上为什么要把这句话放在这里呢？？？？？？？没道理啊
        #执行一次上面的while程序之后，会给每个点分配一个簇类，执行下面的while程序用全部的点计算此时的各个簇心。
        # 第二次while循环时用更新后的簇心来给每一个点分簇类，直至某一次while循环中所有的点的簇类都没有被更新/纠正。
        # （因为第一次给所有点分的簇类通常存在不合理的，所以程序还会有第二次while循环）
        for cent in range(k):  # recalculate centroids 重新计算质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  #！！！！！！ .A是将前面的转换成数组（这样就可以读取数组的行列的大小了，矩阵是不能直接读取大小的）
            #认真分析上面这句，==　是一个判断语句，成立则为１，否则为０，在利用nonezero()[0]找到clusterAssment的第一列为cent的行的下标。
            #dataSet[行下标列表]就可以提取出其中属于cent类的行样本（点）。
            #在类簇评估矩阵的第0列（即簇类）的所有行找到是cent类的数据                                                                 # get all the point in this cluster 获取这个簇的所有点
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean 　按列求和，直接想象成将数据按列亚索成一行，最后的到一个行向量，再存放在centroids质心中
    return centroids, clusterAssment  #返回k个 簇中心点的坐标（k行n列的矩阵）  簇评估分配结果（m行2列 列0存放类，列1 存放该点与该簇心的距离平方）


######################### 　　　第二个例子  二分k-均值聚类算法       ###############################################
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]#这里的0如何理解？？？？？#dataSet是一个ｍ维数组（元组），menan之后是一个１维数组，.tolist()之后是列表
    centList =[centroid0]  #这个[列表]是什么意思？？？？？？  #create a list with one centroid　列表存放质心的坐标
    for j in range(m):     #calc initial Error　计算每一个点到质心的距离平方　并存放在１列中
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):      #将质心存放在列表cenList中　质心的数量小于k说明还需要继续划分（刚开始的时候只有一个簇心）
        lowestSSE = inf #假设无穷大
        for i in range(len(centList)):  #遍历每一个簇心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]  #提取出簇心为i的数据　get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  #将簇心i二分二分二分二分二分二分二分二分
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum　　计算簇心i二分之后的两个簇的SSE之后
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])  #计算不是簇心i的SSE (因为这些簇心没有改变过所以直接使用之前计算出的值就能使用)
            #print "sseSplit, and notSplit　SSE : ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:    #寻找某个簇心i。使的按这个簇心二分之后能够使　SSE最小
                bestCentToSplit = i #存储簇心的ｉ
                bestNewCents = centroidMat  #存储新生成的两个簇心
                bestClustAss = splitClustAss.copy() #存储新生成的簇心ｉ的所有数据的评估矩阵
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)  #change 1 to 3,4, or whatever　给新生成的簇心点i的类别存放为　len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #将不变动的那部分簇心的点类　存放为ｉ
        #print 'the bestCentToSplit is: ',bestCentToSplit
        #print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]   #replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])  #原来的簇心已经划分成两个簇心了，将新生成的一个簇心替代原来的那个簇心，再将新生成的簇心添加到　簇心列表中去
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss    #reassign new clusters, and SSE　更新变种的那部分点所属的簇心和距离平方
    return mat(centList), clusterAssment



