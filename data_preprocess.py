# -*- coding: utf-8 -*-
'''
split the raw data to 1200 sample points
Noraml: 400*1200   400(sample nums)  1200(sample points)
Ball: 3*400*1200   3(falut category) 400(sample nums)  1200(sample points)
Inner Race: 3*400*1200   3(falut category) 400(sample nums)  1200(sample points)
Outer Race: 3*400*1200   3(falut category) 400(sample nums)  1200(sample points)
'''

import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import os


os.chdir("/Users/bin/Desktop/fault diagnosis/毕业论文相关/未命名文件夹/fault diagnosis/fault_data")


def raw_data_make():
    a = []
    list_dir = os.listdir("Norm")
    list_dir.remove(".DS_Store")    # bad mac
    # get Noraml feature

    for i in xrange(len(list_dir)):
        path = os.path.join("Norm", list_dir[i])
        data = sio.loadmat(path)
        val_name = list_dir[i].split(".")[-2]
        if len(val_name)==2:
            val = data['X0' + val_name + "_DE_time"].reshape(1, -1)
        else:
            val = data['X' + val_name + "_DE_time"].reshape(1, -1)
        for j in range(100):
            # print len(val[0][j * 1200: (j + 1) * 1200])
            a.append(val[0][j * 1200: (j + 1) * 1200])

    # get fault feature
    cat_list = ["Ball", "Inner Race", "Outer Race"]
    for k in cat_list:
        list_dir = os.listdir(str(k) + "/0.007")
        list_dir.remove(".DS_Store")   # bad mac
        for i in xrange(len(list_dir)):
            path = os.path.join(str(k) + "/0.007", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)
            # print type(val)
            # print list[i]
            # val = data[]
            for j in range(100):
                # print len(val[0][j * 1200: (j + 1) * 1200])
                a.append(val[0][j * 1200: (j + 1) * 1200])

        list_dir = os.listdir(str(k) + "/0.014")
        list_dir.remove(".DS_Store")   # bad mac
        for i in xrange(len(list_dir)):
            path = os.path.join(str(k) + "/0.014", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)
            # print type(val)
            # print list[i]
            for j in range(100):
                # print len(val[0][j * 1200: (j + 1) * 1200])
                a.append(val[0][j * 1200: (j + 1) * 1200])

        list_dir = os.listdir(str(k) + "/0.021")
        list_dir.remove(".DS_Store")   # bad mac
        for i in xrange(len(list_dir)):
            path = os.path.join(str(k) + "/0.021", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)
            # print type(val)
            # print list[i]
            for j in range(100):
                # print len(val[0][j * 1200: (j + 1) * 1200])
                a.append(val[0][j * 1200: (j + 1) * 1200])
    all_data = np.array(a)  # 全部特征
    #print all_data.shape

    # get single value label
    label = []
    for i in xrange(10):
        tmp = [i for j in xrange(400)]
        label.extend(tmp)
    label = np.array(label).reshape(-1, 1)  # 标签 0 1 2 3 4 5 6 7 8 9
    #print label.shape

    # get one hot label
    enc = preprocessing.OneHotEncoder()
    enc.fit(label)
    arr = enc.transform(label).toarray()
    sparse_label = np.array(arr, dtype="int64")  # 稀疏标签
    #print sparse_label.shape

    return all_data, label, sparse_label


def get_shuffle_feature_and_laebl(feature, label):
    data = np.concatenate((feature, label), axis=1)
    np.random.shuffle(data)
    feature = data[:, 0: 1200]
    label = data[:, 1200:]
    #label = np.array(data[:, 1200:], dtype="int32")
    return feature, label


def get_train_validation_test(all_data, sparse_label):
    for i in range(10):
        if i == 0:
            train_tmp = all_data[0: 280]
            validation_tmp = all_data[280: 360]
            test_tmp = all_data[360: 400]

            train_label_tmp = sparse_label[0: 280]
            validation_label_tmp = sparse_label[280: 360]
            test_label_tmp = sparse_label[360: 400]


        else:
            train_tmp = np.concatenate((train_tmp, all_data[i * 400: i * 400 + 280]), axis=0)
            validation_tmp = np.concatenate((validation_tmp, all_data[i * 400 + 280: i * 400 + 360]), axis=0)
            test_tmp = np.concatenate((test_tmp, all_data[i * 400 + 360: (i + 1) * 400]), axis=0)

            train_label_tmp = np.concatenate((train_label_tmp, sparse_label[i * 400: i * 400 + 280]), axis=0)
            validation_label_tmp = np.concatenate((validation_label_tmp, sparse_label[i * 400 + 280: i * 400 + 360]),
                                                  axis=0)
            test_label_tmp = np.concatenate((test_label_tmp, sparse_label[i * 400 + 360: (i + 1) * 400]), axis=0)

    train = np.array(train_tmp)
    validation = np.array(validation_tmp)
    test = np.array(test_tmp)

    train_label = np.array(train_label_tmp)
    validation_label = np.array(validation_label_tmp)
    test_label = np.array(test_label_tmp)
    return train, train_label, test, test_label

if __name__ == "__main__":
    all_data, label, sparse_label = raw_data_make()
    feature_shuffle, label_shuffle = get_shuffle_train_test(all_data, label)
    feature_shuffle, sparse_label_shuffle = get_shuffle_train_test(all_data, sparse_label)

























