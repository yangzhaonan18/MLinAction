#The following is  in the  page 19 of ML book
# coding=utf-8

from numpy import *

#The following is  in the  page 21 of ML book
#将文件转化成矩阵
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():                 #一行一行的读取
        line = line.strip()                     #.strip()去掉按行读取的一行字符字符串中首尾的 回车和空格
        listFromLine = line.split('\t')         #将上一步得到的整行数据 用tab字符分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]              #前三个
        classLabelVector.append(int(listFromLine[-1]))      #最后一个
        index += 1
    return returnMat,classLabelVector


