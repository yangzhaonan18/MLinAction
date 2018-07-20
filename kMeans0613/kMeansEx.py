
# coding=UTF-8

import urllib
import json
from geopy import geocoders
from math import *
#输入地址（字符串），返回地点对应的坐标（经度纬度）字典

from  kMeans import *
import requests
 #申请的key
#提供一个地址返回一个坐标
ak='f30c9d52b003c2b3ac089e2672e18baf'   #高德地图申请的API的KEY

 #传入地址，返回对应地址的经纬度信息
def address(address):   #输入的是一个字符串地址　如：address＝'北京市'
    url = "http://restapi.amap.com/v3/geocode/geo?key=%s&address=%s" % (ak,address)
    data = requests.get(url)
    contest = data.json()
    contest = contest["geocodes"][0]['location']  #取其中的坐标
    return contest  #返回的是坐标　unicode 类型的　如：116.407526,39.904030　　
if __name__ == '__main__':
    print(address('北京市'))

from time import sleep     #sleep(1)
def massPlaceFind(addressFileName ='placeAddress.txt', placeFileName = 'placeName.txt'):
    #地址文件名找到各个地址的坐标并存放在列表中返回 将地址信息存放在place.txt中
    fw = open(addressFileName, 'w')
    cityList = list()
    addressList = list()
    xAddressList = list()
    yAddressList = list()
    for city in open(placeFileName).readlines():
        #print "city is :", city
        city = city.strip() #去掉空格
        city = city.split('\n')#去掉换行
        cityList.extend(city) #将城市添加到列表中取
    #print "整个城市列表是：", cityList
    num = len(cityList)
    for i in range(num):
        #sleep(1)
        addressList.append(address(cityList[i]))
        #print "第%2d个城市" % i,cityList[i],"的坐标是:",address(cityList[i])
        #print "type(cityList[i]):",type(cityList[i])
    for i in range(num):
        abc = list(addressList[i].split(','))
        fw.write('%s\t%f\t%f\n' % (cityList[i], float(abc[0]), float(abc[1]) ))
        xAddressList.append(float(abc[0]))
        yAddressList.append(float(abc[1]))
    fw.close()
    return xAddressList,yAddressList, cityList
if __name__ == '__main__':
    massPlaceFind()

#上面是自己写的
##########################################3
def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy

import matplotlib.pyplot as plt


def clusterClubs(xAddressList, yAddressList, cityList, numClust = 3, addressFileName ='placeAddress.txt'):
    datList = []
    placeList = []
    for line in open(addressFileName).readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[1]), float(lineArr[2])])
        placeList.append(lineArr[0])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('chinaMap.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    for i in range(len(xAddressList)):
        ax1.annotate(cityList[i].decode('utf-8'),(xAddressList[i], yAddressList[i]))

    #ax1.annotate("asfasdf",(110,30))
    #ax1.annotate("萨德发的", (118, 35))

    plt.xlabel(u'经度')
    plt.ylabel(u'维度')
    print "簇心坐标是：\n", myCentroids
    plt.axis([102.5, 121, 16, 39])
    plt.show()


