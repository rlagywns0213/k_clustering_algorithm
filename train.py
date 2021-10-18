import time
import argparse
import numpy as np
import pandas as pd
import random
from models import cluster_modeling
import yaml
from utils import load_data, visualize
# config
with open('configuration.yaml') as f:
  configuration = yaml.load(f)

iter_num = configuration['iter_num']
cluster_num = configuration['cluster_num']
data_num = configuration['data_num']
#hidden = configuration['hidden']


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--iter_num', type=int, default=iter_num,
                    help='Number of iteration to train.')
parser.add_argument('--cluster_num', type=int, default=cluster_num,
                    help='Number of cluster_num to train.')
parser.add_argument('--data_num', type=int, default=data_num,
                    help='Number of data_num to train.')
parser.add_argument('--center', required = False, default='means',
                    help='how to calculate center')

args = parser.parse_args()
np.random.seed(args.seed)


# Load data
data_num = args.data_num
k = args.cluster_num  # hyper paramters
x, real_answer =load_data(data_num, k)
centers = x[np.random.choice(len(x), size=k, replace=False)] #center: data point 중 무작위로 출력

# train
t = time.time()
group, log, iter = cluster_modeling(x, k, iter_num, centers)
print('All clustering time: {:.4f}s'.format(time.time() - t))

# figure
print("classification & Visualize Start")
visualize(iter, log, real_answer, data_num, cluster_num)

