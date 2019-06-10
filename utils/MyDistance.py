
import numpy as np
import pandas as pd
import math

from utils.MyCount import C


def getDistanceFromAvarage(df):
    df  = df.values
    average = np.mean(df,axis=0)
    df = df -average
    # 将数据加入到第一行
    # df = np.insert(df,0,values=average,axis=0)
    # 平法-相加-开更号
    df = np.square(df)
    df = np.sum(df,axis=1)
    df = np.sqrt(df)
    df = np.reshape(df,[-1,1])
    content = pd.DataFrame(df,columns=['Distance'])

    return content,average

def getCompareDistanceFromNeighbor(data,index,IsAnormaly_index,n = 2):
    data = data.values
    SumDistance = 0
    # 比较当前，与前n个
    for i in range(n):
        # 如果比较的前个在黑名单中（异常），使用前一个值代替
        n = 1
        while(True):
            if(index-i-n) not in IsAnormaly_index:
                temp = Compare(data[index-i-n],data[index])
                break
            else:
                n += 1

        # if(index-i-1) in IsAnormaly_index:
        #     temp = Compare(average,data[index])
        # else:
        #     temp = Compare(data[index-i-1],data[index])
        # temp = Compare(data[index-i-1],data[index]) #调试专用

    SumDistance += temp

    return SumDistance


def Compare(a,b):
    length = len(a)
    Mysum = 0
    for i in range(length):
        temp= (a[i]-b[i])*(a[i]-b[i])
        Mysum += temp
    com = math.sqrt(Mysum)
    return com


def getSpaceDistance(content,index):
    data = content.values
    space_list = np.random.randint(low=0,high=len(content),size=15)
    space_distance_map = {}
    for item in space_list:
        distance = Compare(data[index],data[item])
        space_distance_map[item] = distance
    space_distance_map = sorted(space_distance_map.items(), key=lambda item:item[1])

    # 只取最小前三个
    # print(space_distance_map)
    return calculate(space_distance_map[0][1],space_distance_map[1][1],space_distance_map[2][1],
                     space_distance_map[3][1],space_distance_map[4][1],space_distance_map[5][1])

def calculate(*args):
    sum = 0
    for i in args:
        sum += i*i
    math.sqrt(sum)
    # print(sum)
    return sum




def algorithm_function(beta):
    Beta = beta[0]
    number = int(beta[1])
    AllDistance = 0
    # 设计function
    anomalys = 0
    anomalys_DF = pd.DataFrame()
    normaly = 0
    normaly_DF = pd.DataFrame()
    IsAnormaly_index = []
    # 遍历数据,这里换成数据长度
    for index,item in content.iterrows():
        # 从第三位开始
        realindex = index +number
        # 防止越界
        if realindex <= len(content):
            Neighbor_distance = getCompareDistanceFromNeighbor(content,index,IsAnormaly_index,n=number)
            # print(Neighbor_distance)
            # print(Average_distance.iloc[realindex].values[0])
        #
        # （时间）邻居距离 + 平均距离 + 空间距离
        #
        AllDistance = Neighbor_distance + Average_distance.iloc[index].values[0] + getSpaceDistance(content,index)
        # print('AllDistance',AllDistance)
        if AllDistance > Beta:
            anomalys += 1
            anomalys_DF = anomalys_DF.append(raw.loc[index])
            IsAnormaly_index.append(index)
            # print("outlier :",item)
        else:
            normaly += 1
            normaly_DF = normaly_DF.append(raw.loc[index])
        # print(raw.loc[index])
    #
    #
    # 控制在正态分布内
    #
    #
    score = normaly/(normaly+anomalys)
    print(score)
    # 保存当前分数的集合状态
    mycount.IsAnomalys = IsAnormaly_index
    mycount.vaildCount(anomalys_DF,normaly_DF)
    if 0 < score < 0.93:
        return score
    elif score > 0.93:
        return 0.9
    else:
        return 0.1


from sopt.SGA import SGA


if __name__ == '__main__':

    mycount = C()
    raw = pd.read_csv('../data/medical/Genetic Variant Classifications(done).csv')
    # 不要标签
    content = raw.iloc[:,:-1]

    # content.reset_index(drop=True, inplace=True)
    # print(content)
    # print(content['value'].value_counts())
    # content = raw.iloc[:,0:-1]
    # ================================================================== #
    #                         Space Distance                             #
    # ================================================================== #

    # Space_distance = getSpaceDistance(content,5)

    # ================================================================== #
    #                         Average Distance                           #
    # ================================================================== #
    Average_distance,average = getDistanceFromAvarage(content)

    # ================================================================== #
    #                         Neighbor Distance                          #
    # ================================================================== #
    # Neighbor_distance = getCompareDistanceFromNeighbor(content,2,n = 4)
    # print('Neighbor_distance : ',Neighbor_distance)

    # ================================================================== #
    #                         Init function                              #
    # ================================================================== #
    sga = SGA.SGA(func = algorithm_function,func_type = 'max',variables_num = 2,
                  lower_bound = [1000000,2],upper_bound = [999999999,6],generations = 2,
                  binary_code_length = 10)
    sga.run()
    # show the SGA optimization result in figure
    # sga.save_plot()
    # print the result
    sga.show_result()

    algorithm_function(sga.global_best_point)
    normal_df, outlier_df = mycount.getInfo()
    # print('anomalys index : ',mycount.IsAnomalys)


    from polt.draw import draw3D,demo_test
    right = content[raw['CLASS']== 1]
    # print(content[content['label']== True])
    draw3D(normal_df,outlier_df,right,'')
    normal_df.to_csv('Genetic Variant Classifications_normal_df.csv')
    outlier_df.to_csv('Genetic Variant Classifications_outlier_df.csv')
