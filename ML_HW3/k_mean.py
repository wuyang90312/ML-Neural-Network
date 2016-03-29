'''
K-mean algorithm:
Find K cluster centers each of which has their own
corresponding points. Those points should be close to 
its cluster center rather than to the other centers 
'''

import numpy as np
import tensorflow as tf
import Euclid_Distance as ed # import functions from local
import plot_generator as plot
 
class k_mean:

    def __init__(self, file_name):
        self.file_name = file_name
    
    def cal_loss(self, X, Y, D):
        ED = ed.Euclid_Distance(X, Y, D)
        dist = ED.cal_Euclid_dis()
        cluster = tf.argmin(dist, 1)
        correspond_cluster = tf.gather(Y,cluster)
        offset = tf.sub(X, correspond_cluster)
        loss = tf.reduce_sum(tf.square(offset))
        return loss, cluster
    
    def cluster(self, K, D, B, portion=0):  
        # Read training data
        X_data= np.load(self.file_name)
        '''
        Take a certain percentage of data as a validation set
        The rest will be used as a training set
        '''
        seperation = int((1-portion)*B)
        self.validation = X_data[seperation:,:]
        X_train = X_data[:seperation,:]
        # print portion,seperation, X_data.shape, validation.shape
        
        '''
        Initialize the value of centers of 3 clusters 
        in 2 indep normal distribution
        '''

        X = tf.placeholder(tf.float32, [None, D], name='dataset')
        Y = tf.Variable(tf.random_normal(shape = (K,D), dtype=tf.float32))

        loss, cluster = self.cal_loss( X, Y, D)

        # Set up the hyperparameters
        learning_rate =  0.001
        epsilon = 1e-5
        beta1 = 0.9
        beta2 = 0.99
        training_epochs = 3000

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        train_op = optimizer.minimize(loss)

        sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        sess.run(init)

        res_loss = []
        min_idx = []
        record, loss_prv = 0, 0
        for epoch in range(training_epochs):
            loss_np, min_idx, mu, _ = sess.run([loss, cluster, Y, train_op], feed_dict={X: X_train})
            if record == 200:
            	break
            elif loss_prv == loss_np:
            	record += 1
            else:
            	loss_prv = loss_np
            res_loss.append(loss_np)
            '''
            if(epoch % 200 == 0):
            	print epoch, ':' ,loss_np
            	# print Y.eval()
            '''
        return res_loss, min_idx, X_data, mu


K = 3 # Define 3 clusters
D = 2 #len(mean) # numbers of element per each dataset
B = 10000
                
km = k_mean("data2D.npy")
# Required argument: numbers of clusters, dimensions of points, numbers of points
res_loss, min_idx, X_data, mu= km.cluster(K, D, B, 1.0/3.0)
print mu

plot.plot_loss(res_loss)
plot.plot_cluster(min_idx, X_data, mu, K)
