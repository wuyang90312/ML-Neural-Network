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
    D = 2 
    B = 10000
                    
    KM = km.k_mean("data2D.npy")
    # Required argument: numbers of clusters, dimensions of points, numbers of points
    _, segment_ids, X_data, mu= KM.cluster(K, D, B, 1.0/3.0)
    
    # Take the validation set as input to calculate the loss from cluster centers
    loss,_ = KM.cal_loss(KM.validation.astype(np.float32), mu, D)
    with tf.Session():
        print "K =",K,":",loss.eval()
    
for i in range(1, 6):
    k_mean_validation(i)
