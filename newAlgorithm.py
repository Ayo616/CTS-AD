
import pandas as pd
import numpy as np

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]  # 全是正常的，预测值应为1
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]  # 全是正常的，预测值应为1
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))  # 全是异常的，预测值应为-1


X_outliers = np.append(X_outliers,values=np.full((20,1),-1),axis=1)
anomaly = pd.DataFrame(X_outliers,columns=['f1','f2','CLASS'])
# print(anomaly)

X_train = np.append(X_train,values=np.full((200,1),1),axis=1)
normal = pd.DataFrame(X_train,columns=['f1','f2','CLASS'])
# print(normal)

raw = pd.concat([anomaly,normal],axis=0).sample(frac=1)
content = raw.iloc[:,:-1]


