import pandas as pd

from utils.MyCount import C
from utils.MyDistance import getDistanceFromAvarage, algorithm_function


from sopt.SGA import SGA
if __name__ == '__main__':

    mycount = C()
    raw = pd.read_csv('./data/MutiDim_time/locationActivity.csv')
    # 不要标签
    content = raw.iloc[:,:-1]
    print(content)
    # ================================================================== #
    #                         Average Distance                           #
    # ================================================================== #
    Average_distance,average = getDistanceFromAvarage(content)

    sga = SGA.SGA(func = algorithm_function,func_type = 'max',variables_num = 2,
                  lower_bound = [10,2],upper_bound = [200,5],generations = 2,
                  binary_code_length = 10)
    sga.run()
    # show the SGA optimization result in figure
    # sga.save_plot()
    # print the result
    sga.show_result()
    algorithm_function(sga.global_best_point)
    mycount.getInfo()


