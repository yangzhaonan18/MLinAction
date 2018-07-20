#coding=UTF-8

# 程序内容:用Tkinter创建GUI  Matplotlib和Tkiner的代码集成

import numpy
import Tkinter #3.0之前Tkinder，3.0之后（包括3.0）tkinder
import matplotlib
import regTrees

matplotlib.use('TkAgg') #将matplotlib后端设置为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#重新绘制图形（以用户输入的终止条件为参数绘图）
#@ tolS:合并叶子节点要求的最低误差降低值
#@ tolN:叶子节点的最小数目
def reDraw(tolS, tolN):#图形是回归树还是模型树在程序内部判断
    reDraw.f.clf()  # 清空图像
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get(): #检查复选框（Model Tree）是否选中,选中就是模型树（叶子节点是线性模型）
        if tolN < 2: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat,regTrees.modelTreeEval)
    else: #没有选中就是回归型（叶子节点就是常数型）
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    print "\n××××××××××××reDraw.rawDat的值是：\n",type(numpy.mat(reDraw.rawDat[:,0]))
    reDraw.a.scatter(list(reDraw.rawDat[:,0]), list(reDraw.rawDat[:,1]),c='r')  # 绘制原始数据的散点图（真实值）
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 使用预测数据yHat绘制折线图（预测值）
                                # （如果叶子节点是常数绘制出的图形就是方正的，如果叶子节点的模型是线性的绘制出的图形拟合程度就比较好）
    reDraw.canvas.show() #绘制图像

#获取输入框的数据
def getInputs():
    try: #从文本输入框中获取树创建终止条件，没有则用默认值
        #读取不到数据时会出现异常，因此使用try:…except:…
        tolN = int(tolNentry.get()) #叶子节点的最小数目，整型的
    except:
        tolN = 10#从文本输入框中获取树创建终止条件，没有则用默认值
        print "enter Integer for tolN"#从文本输入框中获取树创建终止条件，没有则用默认值
        tolNentry.delete(0, Tkinter.END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get()) #要求的最低误差降低值，浮点型
    except:
        tolS = 1.0
        print "enter Float for tolS"
        tolSentry.delete(0, Tkinter.END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()  #读取输入框的数据
    reDraw(tolS, tolN)  #绘制图像


root = Tkinter.Tk()

reDraw.f = Figure(figsize=(5,4), dpi=140)  # 创建canvas画布 图的长和宽大小 图片的分辨率
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# tolN输入框
Tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
# tolS输入框
Tkinter.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
#ReDraw按钮
Tkinter.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = Tkinter.IntVar()
chkBtn = Tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = numpy.mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = numpy.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop() #启动事件循环