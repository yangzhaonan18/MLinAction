#coding=UTF-8




###  FP树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue #当前节点名称
        self.count = numOccur #计数
        self.nodeLink = None #用于连接相似的元素项（指针，指向树中的相同元素，抽取条件模式基的时候要用）
        self.parent = parentNode  #当前节点的父节点
        self.children = {} #当前节点的子节点

    def inc(self, numOccur): #当前节点的计数增加numOccur
        self.count += numOccur

    def disp(self, ind=1): #在屏幕上显示当前节点下的整棵树 （ind的大小表示子节点缩进的空格长度）
        print '  ' * ind, self.name, ' ', self.count #显示当前节点的名称+计数（根节点只有一个，第一次运行这行时，直接显示根节点）
        for child in self.children.values(): #遍历当前节点.children字典的所有键，即当前节点的子节点的名称
            child.disp(ind + 1) #遍历当前节点的所有子节点并显示（子节点在父节点的基础上会多一个缩进）

##################       根据数据字典dataSet创建FP树

###  简单数据集及数据包装器
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet): #初始化数据集，将集合转化成字典（事务：频数=1）
    retDict = {} #这个字典名字是像是局部变量一样，只在这个程序中有意义，取名叫什么对输入输出完全没有影响。函数被调用的时候也不会关心函数内部变量的名称。
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


#让头指针表中的元素nodeToTest最终指向最新的节点targetNode。
def updateHeader(nodeToTest, targetNode):  # this version does not use recursion
    while (nodeToTest.nodeLink != None):  # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

#@ items:一个事务过滤和重排序后的频繁元素的集合
#@ inTree:FP树的当前节点
#@ headerTable:频繁项集的头指针列表
#@ count:（当前元素所属）事务的频数
def updateTree(items, inTree, headerTable, count):#更新FP树
    if items[0] in inTree.children:  # check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count)  # incrament count
    else:  # add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:  # 头指针表中的元素指向为空的话，说明这个item[0]元素是第一次被添加到FP树中去，
            # 这个时候则需要将头指针表字典的元素指向FP中这个相同的元素；更新头指针表的指针，指向FP树中相同的元素
            headerTable[items[0]][1] = inTree.children[items[0]]
        else: # 头指针表中的元素指向不为空的话，说明这个元素之前已经在FP树中出现过了，
              # 这个时候则需要调用函数来实现指向最后一个（当前提供）的节点inTree.children[items[0]]
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        # 这里的item[1::]的用法是一种省略，完整的应该是形如：[1:9:1]，前两个是小标，最后一个是步长，最终提取出新的集合，
        # 这里的[1::]是提取除第0个元素之外的所有作为新的集合。
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        # 递归创建新的树（一个节点），一个事务中的所有元素在FP树中是排列成一串的，是父子关系，
        # 所以这里的节点使用的是当前节点的子节点（用来存放同一个事务中的第二个元素）


### 构建FP树 这个函数会调用上面的两个update函数
def createTree(dataSet, minSup=1): # 根据数据集（字典形式存放整个事务，所有事务的频数即键值通常为1），和最小支持度创建FP树
    headerTable = {} #字典 用于存放头指针表
    #go over dataSet twice 第一次遍历dataSet字典：创建头指针列表字典，存放满足最小支持度的元素和频数
    for trans in dataSet: #遍历的是所有的键（事务）   first pass counts frequency of occurance
        for item in trans: #遍历每一个键（事务）中的每一个元素
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans] #在headerTable字典中找item键（元素）的键值，找不到就返回1，
            # dataSet[trans]表示item键（元素）所属的事务出现的频数，通常为1。
    for k in headerTable.keys():  #去除头指针列表中不满足最小支持度的元素 remove items not meeting minSup（不满足）
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys()) #提取头指针列表字典中的键存放在集合freqItemSet中
    #print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
    for k in headerTable: #遍历字典的所有键（元素）（满足最小支持度的）
        headerTable[k] = [headerTable[k], None] #扩展头指针报键值的内容，除了需要频数外，有添加了一个指针，用于指向FP树中相同元素的节点。
        # reformat headerTable to use Node link
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #创建根节点 create tree
    for tranSet, count in dataSet.items():  #go through dataset 2nd time 第二次遍历dataSet字典：
        # 1创建一个新的字典localD每一次存放一个过滤和重排序后的事务orderedItems；2在根节点的基础上更新orderedItems列表中的元素，刷新FP树
        localD = {} #对每一个事务都创建一个新的字典，用来存放过滤和重排序后的事务
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0] # 给字典localD添加 元素：频数
        if len(localD) > 0:
            # 按键值p[1]的降序排列，有序提取出键（元素）存放在orderedItem列表中。 这里的v和p指的都是字典中的项（键：键值）
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # orderedItems：一个原始的事务过滤和重排序后的[元素的集合]；retTree：FP树的根节点；
            # headerTable：频繁项的头指针表；count：当前元素所属原始事务的出现频数（通常为1）
            updateTree(orderedItems, retTree, headerTable, count) #populate tree with ordered freq itemset（用频繁项集填充树，即更新FP树）
    return retTree, headerTable #return tree and header table


##########################################         从一颗FP树中挖掘频繁项集

#根据节点leafNode上溯这颗树，将上溯路径返回到列表prefixPath中
def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name) #添加起始节点的上溯路径（包含起始节点leafNode）
        ascendTree(leafNode.parent, prefixPath) #向上访问父节点

#@ basePat:起始元素 （这个函数中不适用这个参数，只使用FP树中的节点treeNode就可以找到所有的上溯路径（不包含起始元素））
#@ treeNode:FP树中相同元素的第一个节点
#找到前缀路径（prefix path）#发现以给定元素项结尾的所有路径的函数
def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: #长度大于1，除了起始元素之外还有其他的元素
            condPats[frozenset(prefixPath[1:])] = treeNode.count #将起始元素的路径（除去起始元素）和起始元素的频数作为字典元素存放
        treeNode = treeNode.nodeLink #指向FP树中下一个相同的元素（找到这个元素的所有上溯路径）
    return condPats


#@ inTree:FP树的根节点
#@ headerTable：头指针列表（字典）
#@ minSup：最小支持度
#@ preFix：前缀列表
#@ freqItemList：频繁项列表
###   递归查找频繁项集的mineTree函数
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])] # 升序排列头指针表（字典）中的元素
    for basePat in bigL:  #start from bottom of header table 从低频的元素开始遍历
        newFreqSet = preFix.copy() # 前缀列表
        newFreqSet.add(basePat) # 创建新的高频元素集合
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet) #频繁项集 添加进列表（这就是我最终要找的频繁项集）
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1]) # 在FP树中找当前节点basePat的前缀路径（条件模式基）
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup) #根据条件模式基字典（和之前的使用过的数据集的字典形式是相同的结构）
        # 创建条件FP树（myCondTree是树的根节点，myHead是头指针列表字典）
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


# ####   例子:在Twitter源中发现一些共现词
# import twitter
# from time import sleep
# import re
#
# def textParse(bigString):
#     urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
#     listOfTokens = re.split(r'\W*', urlsRemoved)
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
# def getLotsOfTweets(searchStr):
#     CONSUMER_KEY = ''
#     CONSUMER_SECRET = ''
#     ACCESS_TOKEN_KEY = ''
#     ACCESS_TOKEN_SECRET = ''
#     api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
#                       access_token_key=ACCESS_TOKEN_KEY,
#                       access_token_secret=ACCESS_TOKEN_SECRET)
#     #you can get 1500 results 15 pages * 100 per page
#     resultsPages = []
#     for i in range(1,15):
#         print "fetching page %d" % i
#         searchResults = api.GetSearch(searchStr, per_page=100, page=i)
#         resultsPages.append(searchResults)
#         sleep(6)
#     return resultsPages
#
# def mineTweets(tweetArr, minSup=5):
#     parsedList = []
#     for i in range(14):
#         for j in range(100):
#             parsedList.append(textParse(tweetArr[i][j].text))
#     initSet = createInitSet(parsedList)
#     myFPtree, myHeaderTab = createTree(initSet, minSup)
#     myFreqList = []
#     mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
#     return myFreqList
#
#





