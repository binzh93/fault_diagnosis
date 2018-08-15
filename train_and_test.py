# -*- coding: utf-8 -*-
from sklearn.svm import SVC
from sklearn import metrics
from data_preprocess import raw_data_make, get_shuffle_feature_and_laebl, get_train_validation_test

from Wavelet import Wavelet_analyze
import numpy as np
import time
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression



def svm_test(train_X, train_Y, test_X, test_Y):
    train_Y = np.argmax(train_Y, axis=1)
    #train_Y = np.array(train_Y, dtype='float32')
    test_Y = np.argmax(test_Y, axis=1)

    #for i in range(train_Y.shape[0]):
        #train_Y[i] = int(train_Y[i])
    #for i in range(test_Y.shape[0]):
        #test_Y[i] = int(test_Y[i])

    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='rbf', coef0=0.0, probability=True)
    #clf = SVC(kernel='rbf', C=1e3, gamma=0.1)
    clf = SVC(C=1e3)


    #clf.fit(train_X, train_Y, eval_metric='auc')
    clf.fit(train_X, train_Y)

    test_pre = clf.predict(test_X)
    print metrics.accuracy_score(test_Y, test_pre)


def get_wavelet_packet_decomposition_feature(feature):
    fea = []
    for i in xrange(4000):
        wave = Wavelet_analyze(feature[i], fs=12000)
        wave.wavelet_tree()
        val = wave.wprvector()
        fea.append(val)
    fea = np.array(fea)
    return fea


def xgboost_test(train_X, train_Y, test_X, test_Y):
    clf = XGBClassifier(
        learning_rate =0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,   # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言, 
                                #假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                                #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        gamma=0,              # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.8,          # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=0.8,   # 生成树时进行的列采样 
        objective= 'binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        seed=27)
    train_Y = np.argmax(train_Y, axis=1)
    test_Y = np.argmax(test_Y, axis=1)


    clf.fit(train_X, train_Y)

    test_pre = clf.predict(test_X)
    print metrics.accuracy_score(test_Y, test_pre)
    

def main():
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print '-------mat to arr----------------'
    all_data, label, sparse_label = raw_data_make()
    #feature_shuffle, label_shuffle = get_shuffle_train_test(all_data, label)
    feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    '''
        with open("/home/binzh/fault diagnosis/regular.csv", "wb") as f:
        for i in xrange(4000):
            for j in xrange(50):
                f.write(str(all_data[i][j]) + ',')
                if j == 49:
                    f.write(str(label[i][0]) + '\n')
    with open("/home/binzh/fault diagnosis/shuffle.csv", "wb") as f:
        for i in xrange(4000):
            for j in xrange(50):
                f.write(str(feature_shuffle[i][j]) + ',')
                if j == 49:
                    f.write(str(label_shuffle[i][0]) + '\n')

    print 'end'
    '''
    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print '-------feature extract----------------'
    feature = get_wavelet_packet_decomposition_feature(feature_shuffle)

    #train_X, train_Y, test_X, test_Y = get_train_validation_test(feature, sparse_label)
    #train_X, train_Y = get_shuffle_train_test(train_X, train_Y)


    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print '-------split train and test----------------'
    #'''
    train_X = feature[:3600]
    train_Y = sparse_label_shuffle[:3600]
    test_X = feature[3600:]
    test_Y = sparse_label_shuffle[3600:]

    #'''
    print train_X.shape
    print train_Y.shape
    print test_X.shape
    print test_Y.shape

    print time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print '----------train and test----------------'
    svm_test(train_X, train_Y, test_X, test_Y)
    xgboost_test(train_X, train_Y, test_X, test_Y)






if __name__ == "__main__":
    '''
    a = np.array([[0,1,0,0], [1, 0,0,0], [0,0,0,1]])
    print a
    a = np.argmax(a, axis=1)
    print a
    '''
    main()
    # import random

    # a = [1, 2, 3, 4, 5]
    # np.random.sample()
    # random.shuffle(a, random.uniform)
    # print a
