'''
With different values of K,
we can compare the percentage for each cluster
'''

import numpy as np
import tensorflow as tf
import k_mean as km
import plot_generator as plot

def k_comparison(K):
    D = 2
    B = 10000
                    
    KM = km.k_mean("data2D.npy")
    _, segment_ids, X_data, mu= KM.cluster(K, D, B)
    
    data = tf.ones(shape = [B,])
    division = tf.unsorted_segment_sum(data, segment_ids, K, name=None)
    
    with tf.Session():
        print "K =",K,":",division.eval()/10000
        plot.plot_cluster(segment_ids, X_data, mu, K)
    
for i in range(1, 6):
    k_comparison(i)