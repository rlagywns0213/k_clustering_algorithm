import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import random

x = []  # data
k = 3  # hyper paramters
np.random.seed(0)
x.extend(np.random.normal(loc=[0,0], scale=0.5, size=(10, 2)).tolist())
x.extend(np.random.normal(loc=[3,3], scale=0.5, size=(10, 2)).tolist())
x.extend(np.random.normal(loc=[-3,-3], scale=0.5, size=(10, 2)).tolist())
x = np.array(x)
real_answer = [0] * 10
real_answer.extend([1]*10)
real_answer.extend([2]*10)
sns.scatterplot(x=x[:,0], y=x[:,1])

log=[]

def cal_distance(x, y): 
    '''
    유클리디안 거리 계산 함수
    '''
    sum = 0
    for a, b in zip(x,y):
        sum += (a-b)**2
    return math.sqrt(sum)

centers = x[np.random.choice(len(x), size=3, replace=False)] # center 출력

def means_center(graph):
    '''
    calculate center(means) '''
    np_graph = np.array(graph)
    return np_graph.mean(axis=0)

for epoch in range(10):
    group = {}
    for i in range(k):
        group[i] = []
    # find nearest center
    for row in x:
        temp = []
        for i in range(k):
            temp.append(cal_distance(centers[i], row))
        group[np.argmin(temp)].append(row.tolist())
    
    
    # plot data store
    for i in range(k):
        group_temp = np.array(group[i])
        group_temp = np.c_[group_temp, np.full(len(group_temp), i)]
        if i == 0:
            store_data = group_temp
        else:
            store_data = np.append(store_data, group_temp, axis = 0)

    # update center
    update_centers = []
    for i in range(k):
        update_centers.append(means_center(group[i]).tolist())
    update_centers = np.array(update_centers)

    if np.array_equal(update_centers,centers):
        print(f"{epoch} epoch early stopped!")
        early_stop_iter = epoch
        break
    else:
        centers = update_centers
        log.append(store_data)

plt.figure(figsize=(15,8))
for i in range(4):
    plt.subplot(2, 10//2+1, i+1)  # row, col, index
    df = pd.DataFrame(log[i])
    df.columns = ['x1', 'x2', 'group']
    print(f"iteration {i+1}'s accuracy : {round(sum(np.array(real_answer) == df.group) / len(real_answer) * 100,2)}%")
    sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = 'group').set_title(f'iter : {i+1}')