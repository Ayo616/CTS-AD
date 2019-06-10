import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
raw = pd.read_csv('../data/MutiDim_time/locationActivity.csv')


def draw3D(normal_DF,anormal_DF,right,name):
    # print(normal_DF.columns.values.tolist())
    # print(normal_DF)
    features = normal_DF.columns.values.tolist()
    # label = features.pop(-1)
    # print(label)
    b = random.sample(features,3)
    # b = features
    print(b)
    aab = normal_DF
    # print(aab)
    x,y,z = aab[b[0]],aab[b[1]],aab[b[2]]
    # ax = plt.figure().add_subplot(221,projection = '3d')
    ax = plt.figure().add_subplot(221,projection = '3d')
    ax.scatter(x,y,z,c='g')
    result = anormal_DF
    rx,ry,rz = right[b[0]],right[b[1]],right[b[2]]
    ax.scatter(rx,ry,rz,c ='b')
    ox,oy,oz = result[b[0]],result[b[1]],result[b[2]]
    ax.scatter(ox,oy,oz,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])
    ax.set_zlabel(b[2])

    b = random.sample(features,3)
    ax = plt.subplot(222,projection = '3d')
    ax.scatter(x,y,z,c='g')
    rx,ry,rz = right[b[0]],right[b[1]],right[b[2]]
    ax.scatter(rx,ry,rz,c ='b')
    ox,oy,oz = result[b[0]],result[b[1]],result[b[2]]
    ax.scatter(ox,oy,oz,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])
    ax.set_zlabel(b[2])

    b = random.sample(features,3)
    ax = plt.subplot(223,projection = '3d')
    ax.scatter(x,y,z,c='g')
    rx,ry,rz = right[b[0]],right[b[1]],right[b[2]]
    ax.scatter(rx,ry,rz,c ='b')
    ox,oy,oz = result[b[0]],result[b[1]],result[b[2]]
    ax.scatter(ox,oy,oz,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])
    ax.set_zlabel(b[2])

    b = random.sample(features,3)
    ax = plt.subplot(224,projection = '3d')
    ax.scatter(x,y,z,c='g')
    rx,ry,rz = right[b[0]],right[b[1]],right[b[2]]
    ax.scatter(rx,ry,rz,c ='b')
    ox,oy,oz = result[b[0]],result[b[1]],result[b[2]]
    ax.scatter(ox,oy,oz,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])
    ax.set_zlabel(b[2])




    plt.show()
    # plt.savefig((name +'3d.png'))




def demo_test(anormal_DF):
    # art_load_balancer_spikes
    # art_increase_spike_density
    raw = pd.read_csv('../data/OneDim_time/art_increase_spike_density.csv')
    print(raw.index.tolist())
    # indx=anormal_DF.index.tolist()
    # print('indx',indx)
    indx = anormal_DF
    o221 = []
    o222 = []
    o223 = []
    o224 = []
    for i in indx:
        if  60 < i < 110:
            o221.append(i)
        elif 150 < i < 200:
            o222.append(i)
        elif 300 < i < 350:
            o223.append(i)
        elif 450 < i < 500:
            o224.append(i)
        else:
            None
    a = raw.iloc[60:110,1:2]['value']
    # a = raw['value']
    # print(a)
    # max_indx=np.argmax(a)#max value index
    
    plt.subplot(221)
    plt.plot(a,'r-o',color='#054E9F')
    plt.grid(True)
    plt.xlabel('Sequential instance')
    plt.ylabel('Value of instance')
    plt.title('One-dimensional Sequential Data')
    plt.plot(indx,a[indx],'gs',color = '#FF0000')

    # plt.plot(max_indx,a[max_indx],'ks',color = '#FF0000')
    # show_max='['+str(max_indx)+' '+str(a[max_indx])+']'
    # plt.annotate(show_max,xytext=(max_indx,a[max_indx]),xy=(max_indx,a[max_indx]))
    # plt.plot(min_indx,a[min_indx],'gs',color = '#FF0000')

    plt.subplot(222)
    a = raw.iloc[150:200,1:2]['value']
    plt.plot(a,'r-o',color='#054E9F')
    plt.grid(True)
    plt.xlabel('Sequential instance')
    plt.ylabel('Value of instance')
    plt.title('One-dimensional Sequential Data')
    plt.plot(indx,a[indx],'gs',color = '#FF0000')


    plt.subplot(223)
    a = raw.iloc[300:350,1:2]['value']
    plt.plot(a,'r-o',color='#054E9F')
    plt.grid(True)
    plt.xlabel('Sequential instance')
    plt.ylabel('Value of instance')
    plt.title('One-dimensional Sequential Data')
    plt.plot(indx,a[indx],'gs',color = '#FF0000')


    plt.subplot(224)
    a = raw.iloc[450:500,1:2]['value']
    plt.plot(a,'r-o',color='#054E9F')
    plt.grid(True)
    plt.xlabel('Sequential instance')
    plt.ylabel('Value of instance')
    plt.title('One-dimensional Sequential Data')
    plt.subplots_adjust(wspace =0.2, hspace =0.5)
    plt.plot(indx,a[indx],'gs',color = '#FF0000')
    # plt.savefig('art_load_balancer_spikes.png')
    plt.show()

def draw2D(normal_DF,anormal_DF,right,name):
    # print(normal_DF.columns.values.tolist())
    # print(normal_DF)
    features = normal_DF.columns.values.tolist()
    # label = features.pop(-1)
    # print(label)
    b = random.sample(features,2)
    # b = features
    print(b)
    aab = normal_DF
    # print(aab)
    # normal green
    x,y = aab[b[0]],aab[b[1]]
    # ax = plt.figure().add_subplot(221,projection = '3d')
    ax = plt.figure().add_subplot(131)
    ax.scatter(x,y,c='g')

    #
    result = anormal_DF
    rx,ry = right[b[0]],right[b[1]]
    ax.scatter(rx,ry,c ='b')

    # abnormal red
    ox,oy = result[b[0]],result[b[1]]
    ax.scatter(ox,oy,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])


    b = random.sample(features,2)
    ax = plt.subplot(132)
    ax.scatter(x,y,c='g')
    rx,ry = right[b[0]],right[b[1]]
    ax.scatter(rx,ry,c ='b')
    ox,oy = result[b[0]],result[b[1]]
    ax.scatter(ox,oy,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])

    b = random.sample(features,2)
    ax = plt.subplot(133)
    ax.scatter(x,y,c='g')
    rx,ry = right[b[0]],right[b[1]]
    ax.scatter(rx,ry,c ='b')
    ox,oy = result[b[0]],result[b[1]]
    ax.scatter(ox,oy,c ='r')
    ax.set_xlabel(b[0])
    ax.set_ylabel(b[1])


    # b = random.sample(features,2)
    # ax = plt.subplot(224)
    # ax.scatter(x,y,c='g')
    # rx,ry = right[b[0]],right[b[1]]
    # ax.scatter(rx,ry,c ='b')
    # ox,oy = result[b[0]],result[b[1]]
    # ax.scatter(ox,oy,c ='r')
    # ax.set_xlabel(b[0])
    # ax.set_ylabel(b[1])





    plt.show()
    # plt.savefig((name +'3d.png'))



if __name__=="__main__":
    # art_load_balancer_spikes
    a = [69, 70, 98, 99, 154, 155, 180, 181, 183, 184, 214, 215, 230, 231, 239, 240, 241, 261, 262, 293, 294, 332, 350, 351, 399, 400, 413, 414, 436, 437, 438, 439, 455, 456, 460, 461, 471, 475, 476, 478, 479, 486, 487, 537, 538, 546, 547, 549, 550]
    # art_increase_spike_density.csv
    b = [0, 6, 100, 106, 200, 206, 300, 306, 400, 406, 500, 506]
    demo_test(a)

