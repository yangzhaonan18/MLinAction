
# coding=UTF-8

from numpy import *
import matplotlib.pyplot as plt

###自己写的画图程序　　
# 根据clustAssing的第一列提供的类　画出　datMat数据集80个点，和ｍyCentiods提供的４个质心（程序中的质心的数目是４个，能自动适应质心的个数）
def plotCenter(datMat, myCentroids,  clustAssing):

    lenOfDatMat = len(datMat)
    myCentroidsNew = array(myCentroids)
    clustAssingNew = array(clustAssing)
    n = shape( myCentroidsNew)[0]
    xcord0 = []; ycord0 = []
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3 = []; ycord3 = []
    for i in range(n):
        if int(clustAssingNew[i][0] == 0):
            xcord0.append(myCentroidsNew[i,0]); ycord0.append(myCentroidsNew[i,1])
        elif int(clustAssingNew[i][0] == 1):
            xcord1.append(myCentroidsNew[i,0]); ycord1.append(myCentroidsNew[i,1])
        elif int(clustAssingNew[i][0] == 1):
            xcord2.append(myCentroidsNew[i,0]); ycord2.append(myCentroidsNew[i,1])
        else:
            xcord3.append(myCentroidsNew[i,0]); ycord3.append(myCentroidsNew[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(lenOfDatMat):
        if clustAssingNew[i][0] == 0:
            ax.scatter(datMat[i][0], datMat[i][1], s = 50 , marker = 's', c ='#BB00FF')
        elif clustAssingNew[i][0] == 1:
            ax.scatter(datMat[i][0], datMat[i][1], s = 100, marker = 'o', c ='#00FF80')
        elif clustAssingNew[i][0] == 2:
            ax.scatter(datMat[i][0], datMat[i][1], s = 150, marker = '<', c ='#FF0000')
        elif clustAssingNew[i][0] == 3:
            ax.scatter(datMat[i][0], datMat[i][1], s = 200, marker = 'p', c ='#9000FF')
    ax.scatter(xcord0,ycord0, s = 550, marker = 'v', c = 'Chocolate')
    ax.scatter(xcord1,ycord1, s = 550, marker = 'v', c = 'Chocolate')
    ax.scatter(xcord2,ycord2, s = 550, marker = 'v', c = 'Chocolate')
    ax.scatter(xcord3,ycord3, s = 550, marker = 'v', c = 'Chocolate')
    plt.show()



