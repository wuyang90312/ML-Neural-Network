'''
With different values of K,
we can compare different validation result
The result will be the loss function
'''

import numpy as np
import tensorflow as tf
import k_mean as km
import plot_generator as plot

def k_mean_validation(K):
    #K = 5 # Define 3 clusters
    D = 2 #len(mean) # numbers of element per each dataset
    B = 10000
                    
    KM = km.k_mean("data2D.npy")
    # Required argument: numbers of clusters, dimensions of points, numbers of points
    _, segment_ids, X_tmp, mu= KM.cluster(K, D, B)
    
    
    data = tf.ones(shape = [B,])
    
    division = tf.unsorted_segment_sum(data, segment_ids, K, name=None)
    with tf.Session():
        print division.eval()/10000
    
    
for i in range(1, 6):
    k_mean_validation
