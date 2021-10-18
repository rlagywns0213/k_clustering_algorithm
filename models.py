import numpy as np
from utils import cal_distance, means_center
import time

def cluster_modeling(x,k,iter_num, centers):
    log=[]
    for iter in range(iter_num):
        time_1 = time.time()
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
            print(f"{iter} iter early stopped!")
            break
        else:
            centers = update_centers
            log.append(store_data)
        print('Iteration: {:04d}'.format(iter+1),
            'time: {:.4f}s'.format(time.time() - time_1))
    return group, log, iter