#coding=UTF

def loadDataSet():  #加载数据，类型是一个列表list
    return [[1,0, 3, 5 ,7], [1, 2,0, 3, 5], [0,1,3,5]]

#函数功能：C1列表的创建需要遍历数据集中的每一个元素
def createC1(dataSet): #创建集合C1（只包含一个项集的集合C），C1中满足支持度要求的项集将构成集合L1（满足支持度的C1）
    C1 = [] #创建一个空列表
    for transaction in dataSet: #transaction：事务，事件
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    return map(frozenset, C1)  # use frozen set so we can use it as a key in a dict(将项集作为键)

#@ D：数据集
#@ Ck:候选项集列表Ck
#@ minSupport:要求的最小支持度
#函数功能：从Ck生成Lk(例如：从C1到L1)
def scanD(D, Ck, minSupport):
    ssCnt = {} #字典：遍历Ck中的所有项集，统计每一个项集在数据集D中出现的频数（项集列表作为键：频数作为键值）
    for tid in D:
        for can in Ck:
            if can.issubset(tid): #如果CK中的项集can在数据集D中
                if not ssCnt.has_key(can): #字典中没有项集can，则添加can且计数为1，否则就是之前已经添加过can了，计数加1
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D)) #数据集的大小，作为计算支持度时的分母使用
    retList = [] #创建一个新的列表Lk（存放满足支持度的的项集），用来更新之前的Ck（存放含k个元素的项集）
    supportData = {} #创建一个新的字典supportData（存放项集的支持度），用来更新之前的字典ssCnt（存放项集的频数）
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key) #将满足支持度的项集添加到另一个列表Lk中去，0表示从前面插入
        supportData[key] = support #将计算的支持度存放在另一个字典中。无论是否满足要求的支持度都存放进字典中去。
                                    # ???????????这里为什么也要存放不满足支持度的数据呢？
    return retList, supportData #返回 Ck 和 k个项集的支持度

############## 组织完整的Apriori算法  Page 207
#从Lk生成Ck+1
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):#遍历的最后一种情况是i访问倒数第二个元素，j访问最后一个元素；此时的i+1=lenLk,后面就不能继续遍历了。
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] #L1 L2列表的长度是LK中项集长度短1，
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union #求并集
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet) #[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]] 转成　[set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]
    #L1中只有一个元素的项集（每一个元素都是一个集合形式存放）
    L1, supportData = scanD(D, C1, minSupport)  # 从C1生成L1列表  和  存放C1中所有项集的支持度字典supportData
    L = [L1] #列表外面再套一个列表
    k = 2
    #k=2时，判断的是L[0]的长度，满足循环条件>0，
        #此时L[k-2]=L[0]=L1;函数aprioriGen的作用是根据L1生成C2  (L[k-2]生成Ck)
        #scanD函数的作用:根据C2生成L2 (Ck生成Lk)
        #将L2添加到L中去
        #k加1
    while (len(L[k-2]) > 0): # L中的第一个元素是L[0]：存放元素个数是1的项集，第二个元素是L[1]:存放元素个数是2的项集,后一个元素都是根据前一个的元素生成的
                             #（L[1]是根据L[0]生成的；注意：L[1]项集为2，L[0]项集为1）
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport) #scan DB to get Lk
        supportData.update(supK) #更新变动的数据supK
        L.append(Lk) #将新的Lk添加到L中去，并根据这个Lk生成下一个Ck+1
        k += 1
    return L, supportData

###################   从频繁项集中挖掘关联规制

#@ L:频繁项集列表
#@ supportData:包含频繁项集支持数据的字典
#@ minConf:最小可信度阈值
#创建 关联规则
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):  #含两个或两个以上元素的项集的集合 形如：[frozenset([0, 1]), frozenset([0, 3]), frozenset([1, 3]), frozenset([3, 5]), frozenset([0, 5]), frozenset([1, 5])]
        for freqSet in L[i]: #具体的项集 形如：frozenset([0, 1])
            H1 = [frozenset([item]) for item in freqSet] #项集中具体的元素 形如：[frozenset([0]), frozenset([1])]
            if (i > 1): #遍历两个以上元素的项集集合的时候，形如：[[0,1,3][0,2,4]]
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf) #作进一步的合并
            else: #i=1 遍历两个元素的项集集合的时候，形如：[[0,1][0,2]]
                #计算可信度  这条语句只能遍历元素为2的项集，讲满足要求的关联规则存放在bigRuleList列表中去。
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    return bigRuleList

# 对规则进行评估
# #计算可信度，找到满足最小可信度的要求规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:    #当有两个元素，freqSet=[0,1]时，H为[frozenset([0]), frozenset([1])]，这里的for会先后遍历两种情况 0->1 和 1->0
        conf = supportData[freqSet]/supportData[freqSet-conseq] # 计算 freqSet-conseq --->conseq的可信度，第一次是计算1到0的可信度，第二次是计算0到1的可信度。
        if conf >= minConf:
            print freqSet-conseq,'-->',conseq,'conf:',conf
            #brl列表中的元素是： (frozenset([1]), frozenset([0]), 1.0) 即 P(1——>0)=1 表示出现1的地方同时出现0的概率是1
            brl.append(('ssssssssssssssssssssssssssssssss',freqSet-conseq, conseq, conf)) #将满足要求的关联规则添加进brl中去，不会直接返回它的值，但会间接存放在bigRuleList中
            prunedH.append(conseq) #将后件conseq添加到prunedH列表中
    return prunedH

#@ freqSet:频繁项集
#@ H：可以出现在规则右侧的元素列表H
# 下面的函数是用来合并子集的，比如我现在的频繁项集是{2,3,5},它的构造元素是{2},{3},{5}，所以需要将{2},{3},{5}两两合并然后再根据上面的calcConf函数计算置信度。
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7): #freqSet是三个元素的项集集合时[frozenset([0]), frozenset([1]), frozenset([5])]
    print "freqSet:", freqSet
    print "H:",H
    m = len(H[0]) #第一运行这行程序的时候，freqSet形如[0,1,5],H形如[[0][1][5]]。m为1（H中的每一个项集都只含一个元素）。之后递归调用rulesFromConseq时 m会变大
    if (len(freqSet) > (m + 1)):  #第一次是3>1+1  大到可以移除大小为m的子集   freqSet为3，m=1时
        #L1（H） 生成 C2
        Hmp1 = aprioriGen(H, m + 1)  #将只有一个元素的项集集合，组合成有两个项集的集合，如：将 0 1 5（H）组合成01 05 15（ Hmp1）
        #计算可信度，将Hmp1中的元素依次作为后件计算可信度，满足条件的添加到brl中，返回后件集合并存放在Hmp1中去，
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf) #这个函数返回的是后件的集合#计算5推01，1推05，0推15三种情况的可信度，并将可信度满足要求的关系规则添加进brl（bigRuleList）中去。
        #当freqSet的数据是[[0], [1], [5]]时，如何才能遍历完所有可能的 前件推后件（A->B）情况呢?
        # 这里的方法是先遍历 后件长度为2的，如5推01，1推05，0推15，后遍历后件长度为1的，如：50推1，10推5，01推5，由后件有长 到 后件短
        if (len(Hmp1) > 1):  # 3>1 递归判断直至 后件的长度为1（）
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
