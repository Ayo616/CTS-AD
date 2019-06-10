
# coding:utf-8

"""
Author: roguesir
Date: 2017/8/30
GitHub: https://roguesir.github.com
Blog: http://blog.csdn.net/roguesir
"""

import numpy as np
import matplotlib.pyplot as plt

def sgn(value):
    if value > 95:
        return 90
    elif 0 < value < 95:
        return value
    else:
        return 0
plt.figure(figsize=(6,4))
x = np.linspace(0, 100, 100)
y = np.array([])
for v in x:
    print(x)
    y = np.append(y,sgn(v))
l=plt.plot(x,y,'b',label='type')
plt.legend()
plt.show()
