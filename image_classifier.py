import glob
import random
import cv2
import joblib
import numpy as np
import os
from scipy.cluster.vq import *   # kmeans clustering
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



data_path = 'data'
category = ['airport', 'auditorium', 'bedroom', 'campus', 'desert', 'football_stadium', 'landscape', 'rainforest']
no_clusters = 100

def get_cat_num(img_path):    # 将类别映射为数字
    cat = img_path.split('\\')[1]
    if cat == 'airport':
        return 0
    elif cat == 'auditorium':
        return 1
    elif cat == 'bedroom':
        return 2
    elif cat == 'campus':
        return 3
    elif cat == 'desert':
        return 4
    elif cat == 'football_stadium':
        return 5
    elif cat == 'landscape':
        return 6
    elif cat == 'rainforest':
        return 7

def get_image_paths(path,categories):
    train_image_path = []
    test_image_path = []
    train_label = []
    test_label = []
    images = []

    for i, cat in enumerate(categories):   # 枚举获得索引序列
        image = glob.glob(os.path.join(path, cat, '*.jpg'))   # 获取每个类别中每张图片的地址
        for d in range(len(image)):
            images.append(image[d])   # 将所有图片地址加入images数组

    random.shuffle(images)  # 打乱数据顺序
    for j in range(len(images)):
        if j < len(images) * 0.8:    # 划分训练集和测试集
            train_image_path.append(images[j])     # 地址和标签一一对应
            train_label.append(get_cat_num(images[j]))
        else:
            test_image_path.append(images[j])
            test_label.append(get_cat_num(images[j]))

    return train_image_path, test_image_path, train_label, test_label

def get_images_descriptors(detector, image_path_array, ori_labels):
    descriptors = []
    labels = []
    i = 0
    for image_path in image_path_array:
        image = cv2.imread(image_path)
        if image is None:  # 标出不存在的图片地址
            print("图片"+image_path+"不存在!")
            continue
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp = detector.detect(gray,None)
        # photo = cv2.drawKeypoints(gray,kp,image)   # 依次展示画出关键点的原图
        # cv2.imshow('photo',photo)
        # cv2.waitKey(0)
        kp, des = detector.compute(gray,kp)
        if des is not None:
            descriptors.append(des)  # 返回n*dim的三维列表，n为特征子个数，dim为特征子维度
            labels.append(ori_labels[i])
        else:
            print("图片"+image_path+"未检测出特征点!")  # 标出未检测出特征点的图片地址
        i = i + 1

    return descriptors, labels

def vstack_descriptors(descriptors_list):    # 垂直拼接特征获得二维数组
    v_descriptors = descriptors_list[0]
    for descriptor in descriptors_list[1:]:
        v_descriptors = np.vstack((v_descriptors, descriptor))

    return v_descriptors

def cluster_descriptors(descriptors, no_clusters):   # knn

    voc, variance = kmeans(descriptors, no_clusters, 1)

    return voc,variance

def extract_features(kmeans, descriptors_list, no_clusters):    # 获取频次集合数组，每一张图片对应一个单元
    image_count = len(descriptors_list)
    im_features = np.zeros((len(descriptors_list),no_clusters),'float32')
    for i in range(image_count):
        words, distance = vq(descriptors_list[i],kmeans)
        for word in words:
            im_features[i][word] += 1

    return im_features

def train_SVC(features, train_labels):
    svm = LinearSVC(max_iter=500000)     # 定义一个较大正数，否则有警告
    svm.fit(features,np.array(train_labels))

    return svm


if __name__ == '__main__':
    # 预备操作：加载数据和对应的数字标签，并切分训练集和测试集
    train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, category)
    print("训练集测试集以及对应的标签!")
    print(train_image_paths, test_image_paths, train_labels, test_labels)  # 查看数据情况
    # print(len(train_image_paths))
    # print(len(test_image_paths))
    # print(len(train_labels))
    # print(len(test_labels))
    # 构建视觉码本 ============>
    sift = cv2.SIFT_create()    # 这里选择SIFT特征算子实例化
    # 在图片中绘制特征点
    test_image_path = 'data/desert/sun_acqlitnnratfsrsk.jpg'
    image = cv2.imread(test_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    kp = sift.detect(gray, None)   # 检测关键点
    photo = cv2.drawKeypoints(gray, kp, image)    # 在原图上画出关键点
    cv2.imshow('descriptors', photo)
    cv2.waitKey(0)

    # 批量提取训练集图片特征子
    descriptors, labels = get_images_descriptors(sift, train_image_paths, train_labels)
    print("特征子集和!")
    print(descriptors)
    descriptors_list = descriptors
    # 堆叠特征子
    descriptors = vstack_descriptors(descriptors)
    print("堆叠特征子!")
    print(descriptors)
    # print(labels)
    # print(len(descriptors))
    # print(len(labels))
    # 聚合特征子
    voc, var = cluster_descriptors(descriptors,no_clusters)
    joblib.dump(voc,"kmeans")   # 保存到本地可通过joblib.load("kmeans")调用
    # print(voc)
    # print(len(voc))  # 分类器数量
    # print(var)    # 偏差

    # 构建图片分类器 ============>

    # 提取图片特征子在码本中的分布
    im_features = extract_features(voc, descriptors_list, no_clusters)
    print("频次集合数组!")
    print(im_features)

    # 对数据标准化处理
    stdSlr = StandardScaler().fit(im_features)    # 获取均值和标准差
    im_features = stdSlr.transform(im_features)   # 将特征值正态化
    # im_features = StandardScaler().fit_transform(im_features)  # 将特征值正态化
    print("标准化处理!")
    print(im_features)

    # 训练分类器
    model = train_SVC(im_features, labels)
    joblib.dump((model), "bof.pkl", compress=3)

    # 评估分类器 ============>
    print("提取测试集特征!")
    test_descriptors, test_labels = get_images_descriptors(sift, test_image_paths, test_labels)
    test_im_features = extract_features(voc, test_descriptors, no_clusters)
    test_im_features = stdSlr.transform(test_im_features)  # 用训练集的标准化对象将特征值正态化
    # 计算整体正确率
    print("计算准确率!")
    predict = model.predict(test_im_features)
    acc = np.count_nonzero(predict == test_labels) / len(test_labels) * 100
    print("准确率为:",acc)
    # 绘制混淆矩阵
    confusion = confusion_matrix(predict, test_labels)   # 混淆矩阵
    # print(confusion)
    plt.imshow(confusion,cmap=plt.cm.Blues)   # 绘制混淆矩阵
    indices = range(len(confusion))    # 刻度
    plt.xticks(indices, category, rotation=320)
    plt.yticks(indices, category)
    plt.colorbar()   # 设置渐变色
    for first_index in range(len(confusion)):
        sum = 0    # 计算每一类被预测为各类的占比
        for second_index in range(len(confusion[first_index])):
            sum += confusion[first_index][second_index]
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, round(confusion[first_index][second_index]/sum,2))  # 保留两位小数
    plt.show()
