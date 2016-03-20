'''
K-mean algorithm:
Find K cluster centers each of which has their own
corresponding points. Those points should be close to 
its cluster center rather than to the other centers 
'''

import numpy as np
import tensorflow as tf
import math as mt
import Euclid_Distance as ed # import functions from local
from matplotlib.pyplot import *
import itertools
 
class k_mean:

    def __init__(self, file_name):
        self.file_name = file_name
    
    def cluster(self, K, D, B):  
        # Read training data
        X_data= np.load(self.file_name)
        mean = X_data.mean()
        dev = X_data.std()
        # Normailize the training data
        X_tmp = (X_data - mean)/ dev 

        '''
        Initialize the value of centers of 3 clusters 
        in 2 indep normal distribution
        '''

        X = tf.placeholder(tf.float32, [None, D], name='dataset')

        Y = tf.Variable(tf.random_normal(shape = (K,D), stddev = 1.0, dtype=tf.float32))

        #with sess.as_default():
        #print Y.eval()
        ED = ed.Euclid_Distance(X, Y, D)
        dist = ED.cal_Euclid_dis()
        cluster = tf.argmin(dist, 1)
        correspond_cluster = tf.gather(Y,cluster)
        offset = tf.sub(X, correspond_cluster)
        loss = tf.reduce_mean(tf.square(offset))
        '''
        print cluster.eval()
        print offset.eval()
        print loss.eval()
        '''

        # Set up the hyperparameters
        learning_rate =  0.01	
        epsilon = 1e-5
        beta1 = 0.9
        beta2 = 0.99
        training_epochs = 500

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        train_op = optimizer.minimize(loss)

        sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        sess.run(init)

        res_loss = []
        min_idx = []
        record, loss_prv = 0, 0
        for epoch in range(training_epochs):
            loss_np, min_idx, _ = sess.run([loss, cluster, train_op], feed_dict={X: X_tmp})
            if record == 20:
            	break
            elif loss_prv == loss_np:
            	record += 1
            else:
            	loss_prv = loss_np
            res_loss.append(loss_np)
            if(epoch % 20 == 0):
            	print loss_np
            	print Y.eval()

        fig1 = figure(1)
        plot(res_loss, 'b')
        show()

        min_idx = np.array(min_idx)
        fig2 = figure(2)
        colors = itertools.cycle(["r","b","g"])
        for i in range(3):
	        myc = next(colors)
	        data = X_tmp[np.where(min_idx == i), :]
	        data = data[0, :]
	        scatter(data[:, 0], data[:, 1], c = myc, alpha = 0.2)
        show()
        
K = 3 # Define 3 clusters
D = 2 #len(mean) # numbers of element per each dataset
B = 10000
                
km = k_mean("data2D.npy")
km.cluster(K, D, B)

