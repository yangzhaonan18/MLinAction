#coding=UTF-8

import matplotlib.pyplot as plt
from numpy import mat

def plot1(xMat, yMat, ws, titlename = "Please fill in the title"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0], s=5, c = 'r')   #绘制数据集的散点图
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)    #绘制拟合直线的折线图
    plt.title(titlename)
    plt.show()

def plot2(xMat, yMat, yHat, titlename = "Please fill in the title"):
    srtInd = xMat[:, 1].argsort(0)  # 按升序排序，返回下标
    xSort = xMat[srtInd][:, 0, :]  # 将xMat按照升序排列
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0], s=5, c='red')  # 绘制数据集的散点图
    plt.title(titlename)
    plt.show()



def plot3(ridgeWeights,titlename = "Please fill in the title"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights[0:len(ridgeWeights)]) #数据的每一行绘制一条折线
    plt.title(titlename)
    plt.show()
