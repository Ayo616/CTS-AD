# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sopt.SGA import SGA

from utils.MyCount import C


def Compare(a,b):
    length = len(a)
    Mysum = 0
    for i in range(length):
        temp= (a[i]-b[i])*(a[i]-b[i])
        Mysum += temp
    com = math.sqrt(Mysum)
    return com

def algorithm_function(beta):
    anomaly = []
    normal = []
    Beta = beta[0]
    for i in dataset:
        temp = []
        for item in centroids:
            temp.append(Compare(i,item))
        if 10*min(temp) > Beta:
            anomaly.append(i)
        if 10*min(temp) < Beta:
            normal.append(i)

    score = len(normal)/(len(normal)+len(anomaly))
    print(score)
    Main_anomaly = anomaly
    Main_normal = normal
    mycount.vaildCount(anomaly,normal)
    if 0 < score <= 0.96:
        return score
    elif score > 0.97:
        return 0.9
    else:
        return 0.1

if __name__ == "__main__":
    mycount = C()
    # dataset = np.random.rand(500, 2)  # 随机生成500个二维[0,1)平面点
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = 0.3 * np.random.randn(100, 2)

    Y = 0.5 * np.random.randn(100, 2)

    # Z = 0.8 * np.random.randn(100, 2)
    # Y = np.append(X-2,Y,axis=0)
    # Y = np.append(X+1.2,Z-2.5,axis=0)
    X_train = np.r_[X + 2.3, Y -1]



    dataset = np.append(X_train,X_outliers,axis=0)

    for i in dataset:
        plt.scatter(i[0],i[1], c='g')
    plt.xlim(xmax=5,xmin=-5)
    plt.ylim(ymax=5,ymin=-5)
    plt.show()


    # estimator = KMeans(n_clusters=3)#构造聚类器
    # estimator.fit(dataset)#聚类
    # label_pred = estimator.labels_ #获取聚类标签
    # centroids = estimator.cluster_centers_ #获取聚类中心
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,10))
    # colors = ['b', 'g','r']
    # markers = ['o', 's','d']
    # for i,l in enumerate(label_pred):
    #     plt.plot(dataset[i][0],dataset[i][1],color=colors[l],marker=markers[l],ls='None')
    # print(centroids)
    # plt.plot(centroids[0][0],centroids[0][1],color = 'r',marker = 'o',ls='None')
    # plt.plot(centroids[1][0],centroids[1][1],color = 'r',marker = 'o',ls='None')
    # plt.show()


    # import matplotlib.pyplot as plt
    # from scipy.spatial.distance import cdist
    # K=range(1,10)
    # meandistortions=[]
    # for k in K:
    #     kmeans=KMeans(n_clusters=k)
    #     kmeans.fit(dataset)
    #     meandistortions.append(sum(np.min(cdist(
    #         X,kmeans.cluster_centers_,"euclidean"),axis=1))/X.shape[0])
    # K_list = {}
    # print(meandistortions)
    # for i in range(0,len(meandistortions)):
    #     K_list[i] = meandistortions[i-1]- meandistortions[i]
    #
    # K_list = sorted(K_list.items(), key=lambda item:item[1], reverse=True)
    # a = K_list[0]
    # plt.plot(K,meandistortions,'bx-')
    # plt.xlabel('k')
    # plt.ylabel(u'平均畸变程度')
    # plt.title('best k is '+ str(a[0]+1))
    # plt.show()




    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Main_anomaly = []
    Main_normal = []
    sga = SGA.SGA(func = algorithm_function,func_type = 'max',variables_num = 2,
                  lower_bound = [1,2],upper_bound = [10,5],generations = 2,
                  binary_code_length = 10)
    sga.run()
    algorithm_function(sga.global_best_point)

    # normal_df, outlier_df = mycount.getInfo()
    Main_anomaly = mycount.final_anomalys_DF
    Main_normal = mycount.final_nomalys_DF
    # print('anomaly',Main_anomaly)
    # print('normal', Main_normal)


    import matplotlib.pyplot as plt
    plt.figure()
    for i in Main_anomaly:
        print(i)
        plt.scatter(i[0],i[1],c='r')
    for i in Main_normal:
        plt.scatter(i[0],i[1],c='g')
    plt.show()