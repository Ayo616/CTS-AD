import pandas as pd

import random
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list




# content = pd.read_csv('./Genetic Variant Classifications.csv')
# # feature = content.iloc[:,:-1]
# # label = content.iloc[:,-1]
# c = ['AF_ESP','AF_EXAC','AF_TGP','cDNA_position',
#      'cDNA_position','CDS_position','Protein_position','CADD_PHRED','CADD_RAW','CLASS']
# content = pd.DataFrame(content,columns=c)
#
#
# print(content["CLASS"].value_counts())
# content = content.dropna()
# import re
#
# def tip(txt):
#
#     re1='((?:(?:[1]{1}\\d{1}\\d{1}\\d{1})|(?:[2]{1}\\d{3})))(?![\\d])'	# Year 1
#     rg = re.compile(re1,re.IGNORECASE|re.DOTALL)
#     m = rg.search(txt)
#     return m.group(1)
#
#
# content = content.applymap(lambda x:x.split('-')[0] if (isinstance(x, str)) else x )
# # print(content.head())
# # content = content.applymap(lambda x:int(x) if (isinstance(x, str)) else x )
#
# posti = content[content["CLASS"]==0]
# negi = content[content["CLASS"]==1]
#
# negi = negi.sample(n=100)
# posti = posti.sample(n=2000)
# print(posti.shape)
# print(negi.shape)
#
#
# # print(v)
# content = pd.concat([posti,negi])
# print(content.shape)
# content = content.sample(frac=1).reset_index(drop=True)
# print(content["CLASS"].value_counts())
#
# content.to_csv("Genetic Variant Classifications(done).csv")

# ================================================================== #
#                                                                    #
# ================================================================== #

# content = pd.read_csv('./ptbdb_normal.csv')
# g = len(content)
# label = [[1] for x in range(g)]
# label = pd.DataFrame(label,columns=None)
# normal_content = pd.concat([content,label],1)
# # print(normal_content.head())
#
# content = pd.read_csv('./ptbdb_abnormal.csv',header=None)
# g = len(content)
# label = [[-1] for x in range(g)]
# label = pd.DataFrame(label,columns=None)
# abnormal_content = pd.concat([content,label],1)
# # print(abnormal_content.head())
#
#
#
# posti = normal_content
# negi = abnormal_content
#
# negi = negi.sample(n=100)
# posti = posti.sample(n=2000)
# print(posti.shape)
# print(negi.shape)
# negi.to_csv("./temp/Genetic Variant Classifications(negi).csv")
# posti.to_csv("./temp/Genetic Variant Classifications(posti).csv")
#
# # print(v)
# content = pd.concat([posti,negi],0)
# print(content.shape)

# print(content.iloc[:,-1].value_counts())
#
# content.to_csv("Genetic Variant Classifications(done).csv")


content = pd.read_csv('./temp/Genetic Variant Classifications(posti).csv')
content = content.sample(frac=1).reset_index(drop=True)
print(content.shape)
print(content.iloc[:,-1].value_counts())
# content.to_csv("EEG(done).csv")
