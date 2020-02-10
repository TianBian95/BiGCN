# -*- coding: utf-8 -*-
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
cwd=os.getcwd()
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main(obj):
    treePath = os.path.join(cwd, 'data/' + obj + '/data.TD_RvNN.vol_5000.txt')
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, "data/" + obj + "/" + obj + "_label_All.txt")
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_f:
            labelDic[eid]=1
            l2 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
    print(len(labelDic))
    print(l1, l2, l3, l4)

    def loadEid(event,id,y):
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            x_x = getfeature(x_word, x_index)
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            np.savez( os.path.join(cwd, 'data/'+obj+'graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            return None
    print("loading dataset", )
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    return

if __name__ == '__main__':
    obj= sys.argv[1]
    main(obj)