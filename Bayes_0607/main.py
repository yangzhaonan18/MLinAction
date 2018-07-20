# coding=utf-8

#标题：第四章 朴素贝叶斯 分类器（过滤器）
#程序说明：共三个例子;
# 1,根据程序里提供的文档信息 预测文档/帖子是否具有攻击性。（训练集和测试集都是直接给的，不存在随机性，不可变）
# 2,根据两个文件夹中的文档信息 预测邮件是否属于垃圾邮件。（从总的数据集中随机选取部分作为测试集：这种方法叫留存交叉验证）
# 3,根据两个RSS源提供的文档信息 预测随机选择的文档是否来只于原来的源。（留存交叉验证）
#内容：
#时间：2018年6月7日 星期四 20:52

import feedparser
import bayes

#第一个例子 恶意留言区分
print "\n　　　　　　　　　　　　　　　　第一个过滤器例子： 恶意留言区分"
bayes.testingNB()

#第二个例子 垃圾邮件区分
print "\n　　　　　　　　　　　　　　　　第二个过滤器例子： 垃圾邮件区分"
bayes.spamTest()

#第三个例子 个人广告中录取区域倾向
#书中的RSS不能读取到信息，相关参数：书中的例子RSS len=60，将20个作为测试样本，其余40个作为训练样本，去掉的是频数前30个词。
# 本程序中使用的例子len=20，将5个个作为测试样本，其余15个作为训练样本，去掉频数为个位数是效果最好，这里暂时取3。
print "\n　　　　　　　　　　　　　　　　第三个例子:个人广告RSS中录取区域倾向"
nasa = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')  #len=60  NASA 航天新闻
ft = feedparser.parse('http://www.ftchinese.com/rss/news')      #len=20，FT中文网（正式官方新闻）政治 经济 全球新闻
#sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')   #len=6
#sf = feedparser.parse('http://rss.yule.sohu.vocabSetcom/rss/yuletoutiao.xml')  #搜狐娱乐（娱乐新闻）有时候len=30 有时却异常

'''
print "第一个的长度是：",len(nasa['entries'])
print "第一个的内容是：",nasa['entries']
print "第二个的长度是：",len(ft['entries'])
print "第二个的内容是：",ft['entries']
print "运行第一次的结果："

'''
#将两个RSS中的数据用来训练和预测
bayes.localWords(nasa, ft)  #程序中已经完成了所有的操作，包括预测错误率的计算。


print "\n运行第二次的结果："
bayes.getTopWords(nasa, ft)









