import numpy as np
import math
import random
import uuid
class circle():
    def __init__(self,feature,r):
        self.id = uuid.uuid1()
        self.feature = feature
        self.r = r
        self.life = True
        self.marry = False


    def fusion(self,circleObj):
        self.feature = (self.feature + circleObj.feature)/2
        self.r = (self.r+ circleObj.r)/2+0.3
        circleObj.life = False

    def makelove(self,circleObj):
        # 移动方向
        move = (circleObj.feature - self.feature )/100
        # print('move',move)
        feature = self.feature - move

        if random.choice([True, False]):
            move = (circleObj.feature - self.feature )/100
            feature = self.feature + move

        area = math.pi * self.r * self.r
        new_area = math.pi * circleObj.r * circleObj.r
        newR = math.sqrt((area+new_area)/math.pi)
        r = newR*5
        print((area+new_area)/math.pi)

        # 双亲失效
        circleObj.life = False
        self.life = False

        self.marry = True
        circleObj.marry = True

        return circle(feature,r)

    def cross(self,circleObj):
        if circleObj.life == True:
            dist = np.linalg.norm(self.feature-circleObj.feature)
            #  有交差
            if (self.r + circleObj.r) > dist and dist > abs(self.r - circleObj.r):
                return True
            else:
                return False
        else:
            return False


class CirclePool():
    def __init__(self):
        self.circleList = []
        self.biggestCircle = None
        self.tag = True
        self.newbaby=[0]

    def initRawData(self,data):
        circles = []
        for i in data:
            r = np.random.uniform(0,0.2)
            circles.append(circle(i,r))
            print(r)
        print('初始化半径结束')
        self.circleList = circles


    def recover(self):
        for i in self.circleList:
            i.life = True

    def finish(self):
        print('zhixing finish')
        self.biggestCircle = 0
        for i in self.circleList:
            if i.r > self.biggestCircle:
                self.biggestCircle = i.r
        print(self.newbaby[-1],self.newbaby[-2])
        if self.biggestCircle > 4 and self.newbaby[-1] == self.newbaby[-1000] \
                and self.newbaby[-2] == self.newbaby[-959]:
            print("finish sore is enough!")
            return True
        else:
            return False

    def process(self):
        index = 1
        while (True):
            print(index)
            print(len(self.circleList))
            if len(self.circleList) == 0:
                break
            else:
                temp = random.sample(self.circleList,1)[0]
            print(temp.marry)
            RemoveList = set()
            if temp.life == True:
                for j in range(len(self.circleList)):
                    if self.circleList[j].life== True and temp.cross(self.circleList[j]) and self.circleList[j].id != temp.id:
                        # temp.fusion(self.circleList[j])
                        son = temp.makelove(self.circleList[j])
                        self.newbaby.append(self.newbaby[-1]+1)
                        self.circleList.append(son)
                        RemoveList.add(self.circleList[j])
                        RemoveList.add(temp)
                        # print(temp.r)
                        # break
                self.newbaby.append(self.newbaby[-1])
                index = index+1


            # for i in RemoveList:
            #     self.circleList.remove(i)

            print('=========')
            # self.recover()
            if index%1000 == 0:
                self.move()

            if index%2000 == 0:
                if(self.finish()):
                    break


        #     plt.ion()
        #     if index%200 == 0:
        #         plt.cla()
        #         for i in self.circleList:
        #             if i.life == True:
        #                 plt.xlim((-4, 4))
        #                 plt.ylim((-4, 4))
        #                 plt.scatter(i.feature[0],i.feature[1],edgecolors='g',s=i.r,alpha = 0.3)
        #         plt.pause(0.1)
        # plt.ioff()
        # plt.show()


    def move(self):
        d1 = max([x.feature[0] for x in self.circleList])-min([x.feature[0] for x in self.circleList])
        d2 = max([x.feature[1] for x in self.circleList])-min([x.feature[1] for x in self.circleList])

        for i in self.circleList:
            # if i.r > 14000:
            #     move_variable = np.random.uniform(-1,1,len(i.feature))
            #     i.feature = i.feature + move_variable
            if i.r > d1/10  or i.r> d2/10:
                i.r = i.r+ np.random.uniform(-4,0.1)
            elif i.r > d1/4 or i.r> d2/4:
                i.r = i.r% max(d1,d2)
            else:
                i.r = i.r+ np.random.uniform(-0.1,0.4)



def WithOutliers(data,center):
    mydata = []
    X = []
    for i in data:
        distance = 9999999
        for j in center:
            distance = min(np.sqrt(np.sum((i.feature-j)**2)),distance)  # 计算欧氏距离
        i.r = distance
        mydata.append(i)
        X.append(distance)
    # from sklearn.cluster import KMeans
    # estimator = KMeans(n_clusters=2)#构造聚类器
    # estimator.fit(X)#聚类
    # label_pred = estimator.labels_ #获取聚类标签
    ave = np.average(np.array(X))
    for i in mydata:
        if i.r > 1.8*ave:
            plt.scatter(i.feature[0],i.feature[1],c='red')
        else:
            plt.scatter(i.feature[0],i.feature[1],c='green')
    # plt.plot(sorted(X))
    # print(label_pred)
    plt.show()


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color







if __name__ == '__main__':
    import matplotlib.pyplot as plt
    outliers =  np.random.uniform(low=-4,high=4,size=(20,2))
    normals =  0.5 * np.random.randn(100,2)
    data = np.r_[normals+2,normals-2]
    data = np.append(data,outliers,axis=0)
    plt.figure(figsize=(10,10))
    for i in data:
        plt.subplot(231).scatter(i[0],i[1],c='g')
    # plt.show()

    c = CirclePool()
    c.initRawData(data)
    circles = c.circleList
    for i in circles:
        if i.marry == False :
            plt.xlim((-4, 4))
            plt.ylim((-4, 4))
            plt.subplot(236).scatter(i.feature[0],i.feature[1],edgecolors='g',s=i.r*8000)

    c.process()

    circles = c.circleList
    for i in circles:

        plt.subplot(232).scatter(i.feature[0],i.feature[1],edgecolors='g',s=i.r)


    for i in circles:
        if i.life == True:
            plt.xlim((-4, 4))
            plt.ylim((-4, 4))
            plt.subplot(233).scatter(i.feature[0],i.feature[1],edgecolors='g',s=i.r,alpha = 0.3)



    for i in circles:
        if i.life == True :
            plt.xlim((-4, 4))
            plt.ylim((-4, 4))
            plt.subplot(234).scatter(i.feature[0],i.feature[1],edgecolors='g')

    # plt.xlim((-4, 4))
    # plt.ylim((-4, 4))
    # plt.subplot(235).scatter(circles[-1].feature[0],circles[-1].feature[1],edgecolors='g',s=circles[-1].r)

    for i in circles:
        if i.marry == True :
            plt.xlim((-4, 4))
            plt.ylim((-4, 4))
            plt.subplot(235).scatter(i.feature[0],i.feature[1],edgecolors='g')

    plt.show()

    plt.plot(c.newbaby)

    plt.show()

    index = 0
    maxR = 0
    maxRObject = 0
    Rlist = []
    CenterList = []
    for item in circles:
        if item.life == True:
            CenterList.append(item.feature)
            Rlist.append(item.r)
            index += 1
            if item.r > maxR:
                maxR = item.r
                maxRObject = item
    print("life point",index)
    print("max R",maxR)
    print("maxRObject",maxRObject.feature)
    print("finnal center",np.mean(np.array(CenterList),0))
    datda = np.mean(np.array(CenterList),0)

    plt.plot(sorted(Rlist))

    plt.show()

    centersObject = []
    centersObject_feature = []
    for i in circles:
        if i.life == True and i.r > maxR*0.5:
            centersObject.append(i)
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        plt.scatter(i.feature[0],i.feature[1],edgecolors='g')
    for i in centersObject:
        plt.scatter(i.feature[0],i.feature[1],edgecolors='g',s=i.r)
        centersObject_feature.append(i.feature)
    plt.show()


    WithOutliers(circles,centersObject_feature)