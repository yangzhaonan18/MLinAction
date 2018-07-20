# coding=utf-8

#程序说明如下
#标题：page 29 kNN 手写识别系统
#内容：训练集约有2000个数据，测试集约有200个数据，输出预测的结果和预测的错误率
#时间：2018年5月29日 6月6日添加备注
'''
运行结果：
/home/yzn/PycharmProjects/kNN_Write_0529_5/venv/bin/python /home/yzn/PycharmProjects/kNN_Write_0529_5/main.py
in 8_36.txt the classifier came back with: 1, the real answer is: 8
in 3_42.txt the classifier came back with: 8, the real answer is: 3
in 5_42.txt the classifier came back with: 3, the real answer is: 5
in 9_63.txt the classifier came back with: 5, the real answer is: 9
in 5_43.txt the classifier came back with: 4, the real answer is: 5
in 8_11.txt the classifier came back with: 6, the real answer is: 8
in 1_86.txt the classifier came back with: 7, the real answer is: 1
in 8_45.txt the classifier came back with: 1, the real answer is: 8
in 3_43.txt the classifier came back with: 2, the real answer is: 3
in 8_20.txt the classifier came back with: 2, the real answer is: 8
in 5_33.txt the classifier came back with: 9, the real answer is: 5
in 3_11.txt the classifier came back with: 9, the real answer is: 3
in 8_23.txt the classifier came back with: 3, the real answer is: 8

the total number of errors is: 13
the total test number is: 946
the total error rate is: 13/946 = 0.013742
the total train number is: 1934

Process finished with exit code 0
'''

#文档的名称代表文档内数字的真实label 如9_200.txt:数字9的第200个例子的文档名
import  kNN
kNN.handwritingClassTest()



