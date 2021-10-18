import numpy as np
import math
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

def load_data(data_num, k,dt):
    # Load data
    x = []  # data

    np.random.seed(0)
    for i in range(k):
        x.extend(np.random.normal(loc=[i*dt,i*dt], scale=0.5, size=(data_num, 2)).tolist()) # *2.5 데이터 거리 구분 위함 
        if i == 0:
            real_answer = [0] * data_num
        else:
            real_answer.extend([i]*data_num)
    
    x = np.array(x)
    #sns.scatterplot(x=x[:,0], y=x[:,1]) #기존 그래프 그리기

    return x, real_answer


def cal_distance(x, y): 
    '''
    유클리디안 거리 계산 함수
    '''
    sum = 0
    for a, b in zip(x,y):
        sum += (a-b)**2
    return math.sqrt(sum)

def means_center(graph):
    '''
    calculate center(means) 
    '''
    np_graph = np.array(graph)
    
    return np_graph.mean(axis=0)


# def median_center(graph):
#     '''
#     calculate center(median) 
#     '''
#     np_graph = np.array(graph)
#     list_median = list((np.median(np_graph[:,0]),np.median(np_graph[:,1])))
#     return np.array(list_median)

def visualize(iter, log, real_answer, data_num, cluster_num):
    plt.figure(figsize=(15,8))
    for i in range(iter):
        plt.subplot(2, iter//2+1, i+1)  # row, col, index
        df = pd.DataFrame(log[i])
        df.columns = ['x1', 'x2', 'group']
        print(f"iteration {i+1}'s accuracy : {round(sum(np.array(real_answer) == df.group) / len(real_answer) * 100,2)}%")
        sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = 'group').set_title(f'iter : {i}')
    
    plt.savefig(f"results/data_{data_num}, cluster_{cluster_num}.png")
    print("Visualization png saved in results!!")
