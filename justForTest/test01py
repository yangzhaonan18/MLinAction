#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time   : 2017/10/17 21:46
# @Author : lijunjiang
# @File   : test.py

from numpy  import *

classLabels = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
print classLabels
labelMat = mat(classLabels).transpose()
print "\nlabelMat is:\n",labelMat
m = 10
alphas = mat(zeros((m,1)))
print  "\nalphas is:\n",alphas
print "\nmultiply(alphas,labelMat) is :\n",multiply(alphas,labelMat)    #对应相乘即可
print "\nmultiply(alphas,labelMat).T is :\n",multiply(alphas,labelMat).T
print "\nmultiply(alphas,labelMat).T + 1 is :\n",multiply(alphas,labelMat).T +1

print "\nmultiply(alphas,labelMat) is :\n",multiply(alphas,labelMat.T)    #对应相乘即可



