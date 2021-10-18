# Implementation of k_means clustering
====

## Data

- I used `np.random.normal`
- And i increase some distance between clusters.`loc=[i*2.5,i*2.5], scale=0.5, size=(data_num, 2)`

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
python train.py --data_num 100000 --cluster 3
```

## Terminal_Results

```bash
$ python train.py   
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

####  1. PCA

  - data : 100,000 , 3 cluster 
  <center><img src="results/data_100000, cluster_3.png" width="50%" height="50%"></center>
