import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from polt.draw import draw3D,demo_test


content = pd.read_csv('../data/medical/EEG(done).csv',header=None)
data = content.iloc[:,0:-1]
# print(content.iloc[:,-1])
# print(content.shape)

clf = LocalOutlierFactor()
result = clf.fit_predict(data)

outlier_df = pd.DataFrame()
normal_df = pd.DataFrame()

for i in range(len(result)):
    if result[i] == -1:
        outlier_df = outlier_df.append(content.iloc[i])
    else:
        normal_df = normal_df.append(content.iloc[i])

print('================= outlier ===================')

print(outlier_df.iloc[:,-1].value_counts())
print('================= normal ===================')
print(normal_df.iloc[:,-1].value_counts())
print(clf)
# right = data[content['label']== True]
# print(content[content['label']== True])
# draw3D(normal_df,outlier_df,right,'')


# PR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）
# FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）
# fpr, tpr, = 0,0
# from sklearn.metrics import auc
# from sklearn import metrics
# metrics.auc(fpr, tpr)