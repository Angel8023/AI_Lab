# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 21:17:45 2019

@author: Administrator
"""

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #nrows
    #为所有的分类类目创建字典
    labelCounts ={}
    for featVec in dataSet:
        currentLable=featVec[-1] #取得最后一列数据
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0
        labelCounts[currentLable]+=1
    #计算香农熵
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value : 
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet #返回不含划分特征的子集

def chooseBestFeatureToSplit(dataSet):
    numFeature=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)#香农熵
    bestInforGain=0
    bestFeature=-1
    for i in range(numFeature):
        featList=[number[i] for number in dataSet] #得到某个特征下所有值（某列）
        uniqualVals=set(featList) #set无重复的属性特征值
        newEntropy=0
        for value in uniqualVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet)) #即p(t)
            newEntropy+=prob*calcShannonEnt(subDataSet)#对各子集香农熵求和
        infoGain=baseEntropy-newEntropy #计算信息增益
        #最大信息增益
        if (infoGain>bestInforGain):
            bestInforGain=infoGain
            bestFeature=i
    return bestFeature #返回特征值

import operater
#投票表决代码
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items,key=operator.itemgetter(1),reversed=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #类别相同，停止划分
    if classList.count(classList[-1])==len(classList):
        return classList[-1]
    #长度为1，返回出现次数最多的类别
    if len(classList[0])==1:
        return majorityCnt(classList)
    #按照信息增益最高选取分类特征属性
    bestFeat=chooseBestFeatureToSplit(dataSet)#返回分类的特征序号
    bestFeatLable=labels[bestFeat] #该特征的label
    myTree={bestFeatLable:{}} #构建树的字典
    del(labels[bestFeat]) #从labels的list中删除该label
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLables=labels[:] #子集合
        #构建数据的子集合，并进行递归
        myTree[bestFeatLable][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLables)
    return myTree

