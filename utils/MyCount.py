


class C():

    def __init__(self):
        self.final_anomalys_DF = None
        self.final_nomalys_DF =None
        self._IsAnomalys = None

    def vaildCount(self,anomalys_DF,nomalys_DF):
        # print('============ anomaly list =============')
        # print(anomalys_DF['label'].value_counts())
        # print('============ anomaly list =============')
        # print(nomalys_DF['label'].value_counts())
        self.final_anomalys_DF = anomalys_DF
        self.final_nomalys_DF = nomalys_DF
        return None



    def getInfo(self):
        print('============ anomaly list =============')
        # print(self.final_anomalys_DF['CLASS'].value_counts())
        # print(self.final_anomalys_DF)
        from polt.draw import draw3D,demo_test
        # demo_test(self.final_anomalys_DF)
        # print(self.final_anomalys_DF)
        print('============ nomaly list =============')
        # print(self.final_nomalys_DF['CLASS'].value_counts())

        # print(self.final_nomalys_DF)
        return self.final_nomalys_DF, self.final_anomalys_DF


