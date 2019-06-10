from glue.utils import matplotlib
from sopt.SGA import SGA
import numpy as np
import pandas as pd
import math

from MultiObject_Algorithm.main import getDistanceFromAvarage, algorithm_function
from utils.MyCount import C
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

mycount = C()

anomaly = pd.read_csv('outlier_df.csv')
normal = pd.read_csv('normal_df.csv')

raw = pd.concat([anomaly,normal],axis=0).sample(frac=1)
content = raw.iloc[:,2:]
# print(content)

right = content[raw['CLASS']== 1]

features = content.columns.values.tolist()

b = random.sample(features,2)
aab = right
# normal green
x,y = aab[b[0]],aab[b[1]]


plt.scatter(x,y,c='g')
# abnormal blue
result = anomaly
rx,ry = right[b[0]],right[b[1]]
plt.scatter(rx,ry,c ='b')

plt.show()
