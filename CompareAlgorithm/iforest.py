import pandas as pd
from sklearn.ensemble import IsolationForest

# ../data/single_Co2_data.csv'
# locationActivity.csv
# MutiDim_nontime/locationActivity.csv
from polt.draw import draw3D,demo_test

content = pd.read_csv('../data/medical/EEG(done).csv',header=None)
data = content.iloc[:,0:-1]

ilf = IsolationForest()
ilf.fit(data)
result = ilf.predict(data)

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

# demo_test(outlier_df)
right = data[content.iloc[:,-1]== -1.0]
# print(content[content['label']== True])
draw3D(normal_df,outlier_df,right,'')

# PR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）
# FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）
# from sklearn import metrics
#
# fpr, tpr, = 43/(43+1097),0/79
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
# from sklearn.metrics import auc
# print(metrics.auc(fpr, tpr))
print(ilf)