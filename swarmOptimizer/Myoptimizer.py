
from sopt.SGA import SGA
from math import sin

def func1(x):
    return (x[0]-1)**2 + (sin(x[1])-0.5)**4 + 2

if __name__ == '__main__':
    sga = SGA.SGA(func = func1,func_type = 'min',variables_num = 2,
                  lower_bound = 0,upper_bound = 2,generations = 20,
                  binary_code_length = 10)
    # run SGA
    sga.run()
    # show the SGA optimization result in figure
    # sga.save_plot()
    # print the result
    sga.show_result()
    print('sf',sga.global_best_point)
