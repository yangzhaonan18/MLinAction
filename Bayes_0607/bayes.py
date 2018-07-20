#coding=UTF-8
from numpy import *

#一条评论就是一个文档，文档标签用1表示是侮辱性质的，0表示正常性质的,文档数目应该是标签数目相同且一一对应

#从文本中构建 词向量 类别标签
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性文字 abusive, 0 notd
    return postingList, classVec

#将所有的文档中出现过的词语放在一个行向量（集合）中。  输入：文档列表的列表，输出：含所有词的长列表
def createVocabList(dataSet):   #全聚德
    vocabSet = set([])  # create empty set 创建一个空的集合（元素不重复）
    for document in dataSet:
        vocabSet = vocabSet | set(document)     #求并集
    return list(vocabSet)

#词集模型 set-of-words model   将出现与否作为一个特征 1出现 0未出现
#将一条文档信息 转化成 行向量。将一条文档inputSet与vocabList对比生成（0,1,0,1,0,1……）形式的向量
def setOfWords2Vec(vocabList, inputSet):# 输入都是行列表 短文档变成长向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:       #inputSet文档中的单词如果出现在vocabList中，则将该词的出现标记为1，未出现标记为0，出现多次也标记为1
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec


#词袋模型 bag-of-words model  将出现的的次数作为特征值 可以表示出现多次的信息
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec  



#训练函数 输入：训练文档矩阵 文档类别标签 输出:p(w|c0) p(w|c1) p(c1)  类别为0/1的文本中各个词语出现的频率 类别为1的文本出现的频率
def trainNB0(trainMatrix,trainCategory):    #使用训练数据集本身计算
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #计算标记为为1的文档的数量
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()   初始化概率 设置为1不设置成0的原因是为了避免0与其他数相乘的0,掩盖了其他数存在的作用
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]     #把标记为1的行向量全部相加  得表示词条出现频数的 向量
            p1Denom += sum(trainMatrix[i])      #求标记为1的文档的全部词的数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()  每个词出现的次数 除以 类别为1的所有文档的总词数 得p(wi|c1)即每个单词在类别为1的文档中出现的频率
    p0Vect = log(p0Num/p0Denom)          #change to log()  取对数的目的是为了避免多个概率（<1）的数相乘后，得到的数很接近0，最后四舍五入就变成0的情况。即避免下溢出
    return p0Vect,p1Vect,pAbusive       #向量 向量 数值

#输入 待分类的长向量 ，计算该向量在类1和类0中出现的概率（贝叶斯公式计算 只比较分子部分，分母相同）
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


#####11111111111111111111111111111111111111111111111111111111111111111111111111111   第一个例子过滤（预测）网上的恶意留言
#这是一个便利函数（convenience function），封装所有操作
def testingNB():
    listOPosts,listClasses = loadDataSet()      #所有文档列表  文档标签
    myVocabList = createVocabList(listOPosts)   #将文档中出现过的所有词构成的一个行列表
    trainMat=[]
    for postinDoc in listOPosts:    #遍历每一行的文档 将字符串内容的postinDoc文档 转换成 行向量 添加到trainMat 训练集中
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))      #根据生成给定的训练集输出 类别0/1中每个单词出现的频率 类别1文档出现的频率
    print "\np0V的值是：\n",p0V
    print "\np1V的值是：\n",p1V
    print "\npAb的值是：",pAb,"\n"
    #测试例子1

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry)) #将输入的文档转化成 长向量
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)   #输出测试向量 预测类别情况
    #测试例子2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)


#运行上面的程序，用于测试朴素贝叶斯算法的运行情况
#testingNB()

###222222222222222222222222222222222222222222222222222222222222222222    第二个例子过滤垃圾邮件   数据从文件中读取
#接受一个大的字符串（句子）并将其解析为一个字符串列表
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)  #z正则表达式 按空格划分成列表
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]        #去掉少于两个字符的字符串，并将字符串转换成小写

#对邮件进行自动化分类处理
def spamTest():     #spam:垃圾邮箱
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26): #两个文件夹中都有25个文档  从50个文档中读取句子并存放在列表中 同时记录类别
        wordList = textParse(open('email/spam/%d.txt' % i).read())  #将文本文件解析成 词的列表
        docList.append(wordList)    #将每个文档的词列表 都存放在docList中 列表的列表
        fullText.extend(wordList)   #将每一个词 都存放在fullText中 所有词的列表
        classList.append(1)     #将从spam文件中读取出的列表 标记为1
        wordList = textParse(open('email/ham/%d.txt' % i).read())#和上面相同
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 将文档列表（列表的列表）内的元素 求并集得到 包含所有词的长列表
    #共50个样本，选10个作为测试，其余40个作为训练样本
    trainingSet = range(50) #训练集的下标
    testSet = []  # create test set
    for i in range(10):     #随机选择10个样本作为测试数据    这种方法叫“留存交叉验证”hold-out cross validation
        randIndex = int(random.uniform(0, len(trainingSet)))    #uniform保证不出现重复的随机数
        testSet.append(trainingSet[randIndex])      #添加到测试数据集中
        del (trainingSet[randIndex])    #并从训练集中删除
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # 将训练集中的每个列表 转换成行向量
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #存放在训练矩阵
        trainClasses.append(classList[docIndex])    #添加训练样本的 类
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    errorCount = 0  #对10个测试样本进行估计，计算错误率
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", "预测分类是：",classifyNB(array(wordVector), p0V, p1V, pSpam),"实际类别是",classList[docIndex]
            print docList[docIndex]
    print "\n the error rate is: %d/%d = %f" % (errorCount , len(testSet),float(errorCount) / len(testSet))
    # return vocabList,fullText

###33333333333333333333333333333333333333333333333333333333333333  第三个例子   朴素贝叶斯从个人广告中获取区域倾向
#ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')  #len=60  书上的源长度为0 不能使用。
#print len(ny['entries']
#import feedparser

#遍历词汇表中的每个词并统计它在文本中出现的次数　输出频数最高的前 30 个单词  测试得出的经验值
def calcMostFreq(vocabList,fullText,topnum2 = 30):   #vocabList单词列表（无重复的）  fullText所有单词的列表（有重复的）
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[: topnum2 ]   ## 找到频数最高的前2个词     经验值******************************************经验值

#
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):     #两种类型的取相同数量的（以最小的为准，多余的不使用），从RSS中读取数据和类别并存放在 docList（文档列表的列表）fullText（依次存放含所有词的列表） classList（源1 标记类1.源2标记类0）
        wordList = textParse(feed1['entries'][i]['summary'])        #每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)     #  将这部分内容的class设置为1
        wordList = textParse(feed0['entries'][i]['summary'])        #每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)     #  将这部分内容的class设置为0
    #print "docList 文档列表:", docList
    #print "fullText 所有词列表:", fullText
    #print "classList 类别列表:",classList
    vocabList = createVocabList(docList)    #create vocabulary 长单词向量
    #print "vocabList 长 词列表:", vocabList
    topnum = 3                     ## 找到频数最高词     经验值***************************×××××××××××××××××***************经验值
    top30Words = calcMostFreq(vocabList,fullText,topnum)   #remove top 30 words    #将频数前30个词移除   实际移除多少词由calcMostFreq()函数内的参数决定
    #print "top30Words 频数前%d的单词" % topnum, top30Words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set    当个类两倍的数量就是 总的样本数量
    testnum = 8 #这里为什么会出现波浪线呢？？？？？？ 全局变量？
    for i in range(testnum):      #部分数据作为作为测试集  测试数据量与训练数据量关系 经验值******************************************经验值
        randIndex = int(random.uniform(0,len(trainingSet)))#随机生成不重复的数 作为下标，将数据添加到测试集中去，并从训练集中删除
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:    #train the classifier (get probs) trainNB0　　　　　　　　　　　　　训练
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #生成训练矩阵
        trainClasses.append(classList[docIndex])    #生成训练矩阵的class类
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))   #计算训练矩阵中两个样本的词的出现频率向量，和类1文档的频率
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items   #对测样本进行预测分类　　　　　　　　预测
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:  #统计预测错误的数量
            errorCount += 1
    print '\nRSS总数据集是%d个，将其中%d个作为测试数据集，删除频数前%d的词后，预测预测错误率为: 错误数/测试集= %d/%d = %f\n' % (2*minLen, testnum, topnum, errorCount, len(testSet), float(errorCount)/len(testSet) )#预测出错的数量/预测集大小
    return vocabList,p0V,p1V  #这里返回的数据是已经去除‘无关高频词’之后的高频词

###第三个例子中 分析数据显示高频的用词
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)  #当localWord（）函数被调用使用时，函数里面的print函数也会打印出东西，这是我不想让他出现的，但直接运行他时却又希望它出现呢？
    #如何实现呢？ if__name__==__init__ 这个种类似的用法，好像可以实现。
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -4.0 : topSF.append((vocabList[i],p0V[i]))  #这是将(vocabList[i],p0V[i])两个数据（词 和 词出现的log概率）看成一个元组 作为列表元素添加到p0V[]列表中
        if p1V[i] > -4.0 : topNY.append((vocabList[i],p1V[i]))
    sortedNasa = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "显示两个RSS源中最具表征性的词汇：\n"
    print "NASA***NASA***NASA***NASA***NASA***NASA***NASA***NASA***NASA***NASA***NASA***NASA***"
    print "sortedNASA 是:\n",sortedNasa
    for item in sortedNasa:
        print item[0]       #只输出列表中每一个元组的一个参数，即 词
    sortedFt = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "\nFT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***FT***"
    print "sortedFT 是:\n",sortedFt
    for item in sortedFt:
        print item[0]
