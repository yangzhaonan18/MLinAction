# coding=UTF-8
#Fri June 29 2018 11:45

import matplotlib.pyplot as plt


def plotmap(dataArr, labelArr):
    fig = plt.figure()  # 画图
    ax = fig.add_subplot(111)
    ax.scatter(dataArr[:,0],dataArr[:,1],s =labelArr[:]*50.0+70 ,c =labelArr)
    plt.show()