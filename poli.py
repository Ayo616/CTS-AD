import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# import numpy as np
# raw = pd.read_csv('./shiyan3.csv')
# # print(raw.columns)
# # Carbon dioxide,Temporary,Humidity,Oxygen
# x = raw['Temporary'].values.tolist()
# y = raw['Humidity'].values.tolist()
# z = raw['Carbon dioxide'].values.tolist()
# q = raw['Oxygen'].values.tolist()
# ax = plt.figure().add_subplot(221,projection = '3d')
# ax.scatter(x,y,z,c='g')
# ax.set_xlabel('Temporary')
# ax.set_ylabel('Humidity')
# ax.set_zlabel('Carbon dioxide')
#
# ax = plt.subplot(222,projection = '3d')
# ax.scatter(x,y,q,c='g')
# ax.set_xlabel('Temporary')
# ax.set_ylabel('Humidity')
# ax.set_zlabel('Oxygen')
#
# ax = plt.subplot(223,projection = '3d')
# ax.scatter(z,y,q,c='g')
# ax.set_xlabel('Carbon dioxide')
# ax.set_ylabel('Humidity')
# ax.set_zlabel('Oxygen')
#
# ax = plt.subplot(224,projection = '3d')
# ax.scatter(z,x,q,c='g')
# ax.set_xlabel('Carbon dioxide')
# ax.set_ylabel('Temporary')
# ax.set_zlabel('Oxygen')
# plt.show()


# def draw3D(normal_DF,anormal_DF,right,name):
#     # print(normal_DF.columns.values.tolist())
#     # print(normal_DF)
#     features = normal_DF.columns.values.tolist()
#     # label = features.pop(-1)
#     # print(label)
#     b = random.sample(features,3)
#     # b = features
#     print(b)
#     aab = normal_DF
#     # print(aab)
#     x,y,z = aab[b[0]],aab[b[1]],aab[b[2]]
#     # ax = plt.figure().add_subplot(221,projection = '3d')
#     ax = plt.figure(projection = '3d')
#     ax.scatter(x,y,z,c='g')
#     plt.show()



from utils.MyCount import C
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mycount = C()

anomaly = pd.read_csv('./MultiObject_Algorithm/outlier_df.csv')
normal = pd.read_csv('./MultiObject_Algorithm/normal_df.csv')

raw = pd.concat([anomaly,normal],axis=0).sample(frac=1)
content = raw.iloc[:,2:]

features = content.columns.values.tolist()
b = random.sample(features,2)
x,y = content[b[0]],content[b[1]]
plt.scatter(x,y,c='g')

# from sklearn.cluster import KMeans
# y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(content)
# plt.scatter(content.iloc[:, 0], content.iloc[:, 1], c=y_pred)



# right = content[raw['CLASS']== 1]
#
# features = content.columns.values.tolist()
#
# b = random.sample(features,2)
# aab = right
# # normal green
# x,y = aab[b[0]],aab[b[1]]
# plt.scatter(x,y,c='g')
#
# # abnormal blue
# result = anomaly
# rx,ry = result[b[0]],result[b[1]]
# plt.scatter(rx,ry,c ='r')

plt.show()
