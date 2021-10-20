# Implementation of k_means clustering
====

## Data

- I used `np.random.normal`
- And i increase some distance between clusters.`loc=[i*dt,i*dt], scale=0.5, size=(data_num, 2)`

## Initialize : how to reduce time

1. Just Random Sample Centroid
2. 1 Random Sample => Max( mean Distance of centroid )

## Usage

```bash
python train.py
```

> initial configuration  
iter_num: 100<br>
cluster_num: 3<br>
data_num: 100<br>

## Example  
```bash
python train.py --data_num 100000 --cluster 3 --distance 0
```

## Terminal_Results

```bash
$ python train.py --data_num 100000 --cluster 3
Iteration: 0001 time: 4.9780s
Iteration: 0002 time: 5.0096s
Iteration: 0003 time: 5.1221s
3 iter early stopped!
All clustering time: 21.3724s
classification & Visualize Start
iteration 1's accuracy : 99.89%
iteration 2's accuracy : 100.0%
iteration 3's accuracy : 100.0%
Visualization png saved in results!!
```

## Visualization by iteration

####  Distance : 2.5


  - data : 10,000 , 4 cluster, distance : 2.5
  <center><img src="results/data_10000, cluster_3.png" width="50%" height="50%"></center>

####  Distance : 0
  - data : 100 , 3 cluster, distance : 0
  <center><img src="results/data_100, cluster_3.png" width="50%" height="50%"></center>

  ```bash
  $ python train.py --data_num 100 --cluster 3
  Iteration: 0001 time: 0.0060s
  Iteration: 0002 time: 0.0050s
  Iteration: 0003 time: 0.0050s
  Iteration: 0005 time: 0.0050s
  Iteration: 0006 time: 0.0050s
  Iteration: 0007 time: 0.0050s
  Iteration: 0008 time: 0.0060s
  8 iter early stopped!
  All clustering time: 0.0509s
  classification & Visualize Start
  iteration 1's accuracy : 74.67%
  iteration 2's accuracy : 79.33%
  iteration 3's accuracy : 86.33%
  iteration 4's accuracy : 92.0%
  iteration 5's accuracy : 95.33%
  iteration 6's accuracy : 95.0%
  iteration 7's accuracy : 95.67%
  iteration 8's accuracy : 95.67%
  Visualization png saved in results!!
  ```

  - data : 100,000 , cluster : 3, distance : 0
  <center><img src="results/data_100000, cluster_3.png" width="50%" height="50%"></center>

  ```bash
  Iteration: 0001 time: 5.0046s
  Iteration: 0002 time: 5.1961s
  Iteration: 0003 time: 5.4633s
  Iteration: 0004 time: 5.3149s
  Iteration: 0005 time: 4.9385s
  Iteration: 0006 time: 5.1649s
  Iteration: 0007 time: 5.6810s
  Iteration: 0008 time: 5.4391s
  Iteration: 0009 time: 5.4926s
  Iteration: 0010 time: 5.2374s
  Iteration: 0011 time: 5.3019s
  Iteration: 0012 time: 5.2025s
  Iteration: 0013 time: 5.2201s
  Iteration: 0014 time: 5.1711s
  Iteration: 0015 time: 5.0116s
  Iteration: 0016 time: 5.1381s
  Iteration: 0017 time: 5.2156s
  Iteration: 0018 time: 5.2689s
  Iteration: 0019 time: 5.2497s
  Iteration: 0020 time: 5.1666s
  All clustering time: 104.8845s
  classification & Visualize Start
  iteration 1's accuracy : 69.94%
  iteration 2's accuracy : 80.44%
  iteration 3's accuracy : 88.36%
  iteration 4's accuracy : 93.28%
  iteration 5's accuracy : 96.13%
  iteration 6's accuracy : 97.81%
  iteration 7's accuracy : 98.77%
  iteration 8's accuracy : 99.33%
  iteration 9's accuracy : 99.63%
  iteration 10's accuracy : 99.8%
  iteration 11's accuracy : 99.87%
  iteration 12's accuracy : 99.85%
  iteration 14's accuracy : 99.84%
  iteration 15's accuracy : 99.83%
  iteration 16's accuracy : 99.83%
  iteration 17's accuracy : 99.83%
  iteration 18's accuracy : 99.83%
  iteration 19's accuracy : 99.83%
  Visualization png saved in results!!
  ```

## My Idea

  - data : 200,000 , cluster : 10, distance : 3.5

  ![image](https://user-images.githubusercontent.com/28617444/138086380-f5506967-3c0b-4b51-aeb9-1789a3e88602.png)

#### log

  ```bash
  Iteration: 0001 time: 9.3414s
  Iteration: 0002 time: 9.7300s
  2 iter early stopped!
  All clustering time: 29.1654s   
  classification & Visualize Start
  iteration 1's accuracy : 98.8%
  iteration 2's accuracy : 100.0%
  ```
  
#### centroid 결과
  ![image](https://user-images.githubusercontent.com/28617444/138086560-0a2d7c2a-fe28-44dc-8dcc-c90b97934674.png)

#### Clustering 결과

  <center><img src="experiments\data_20000, cluster_10,distance_3.5,init_new.png" width="50%" height="50%"></center>

  - The most efficient Centroids initialize
