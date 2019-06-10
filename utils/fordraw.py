from polt.draw import draw3D,draw2D,demo_test

import pandas as pd
# raw = pd.read_csv('../data/medical/diabetes(done).csv')
# normal_df = pd.read_csv('diabetes_normal_df.csv')
# outlier_df = pd.read_csv('diabetes_outlier_df.csv')
# normal_df = normal_df.iloc[:,0:-1]
# outlier_df = outlier_df.iloc[:,0:-1]
# content = raw.iloc[:,0:-1]
#
# right = content[raw['Outcome']== -1]
# # # print(content[content['label']== True])
# draw3D(normal_df,outlier_df,right,'')


# ================================================================== #
#                         gvc                                        #
# ================================================================== #
feature = ['AF_ESP','AF_EXAC','AF_TGP','CADD_PHRED','CADD_RAW','CDS_position','Protein_position','cDNA_position','cDNA_position.1']
raw = pd.read_csv('../data/medical/Genetic Variant Classifications(done).csv')
normal_df = pd.read_csv('Genetic Variant Classifications_normal_df.csv')
outlier_df = pd.read_csv('Genetic Variant Classifications_outlier_df.csv')
normal_df = normal_df[feature]
outlier_df = outlier_df[feature]
content = raw[feature]

right = content[raw['CLASS']== 1.0]
# # print(content[content['label']== True])
draw3D(normal_df,outlier_df,right,'')