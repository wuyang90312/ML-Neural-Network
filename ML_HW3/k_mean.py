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

'''
Calculate the means of 10,000 datasets on both dimensions
'''   
def cal_mean(x):
    mean = [0,0]
    for n in X:
        mean = mean + n
    
    mean = mean / len(X)
    return mean  

'''
Calculate the deviations of 10,000 datasets on both dimensions
'''
def cal_dev(x, mean):
    dev = [0,0]
    for n in X:
        dev = dev + pow(n - mean, 2)

    for n in range(len(dev)):
        dev[n] = mt.sqrt(dev[n] / len(X))
    return dev
    
# Read training data
X_tmp= np.load('data2D.npy')


# calculate the means and deviations for later initilization
'''
mean = cal_mean(X)
dev = cal_dev(X, mean)
'''

# Y= np.array([[0,1], [1,2],[2,3]], dtype = np.float64)

K = 3 # Define 3 clusters
D = 2 #len(mean) # numbers of element per each dataset
B = 10000
'''
Initialize the value of centers of 3 clusters 
in 2 indep normal distribution
'''
'''
Y = np.zeros(shape = [K, D])
for i in range(D):
    Y[0:K, i] = dev[i]*np.random.randn(K) + mean[i]
'''
X = tf.placeholder(tf.float32, [None, D], name='dataset')

Y = tf.Variable(tf.random_normal(shape = (K,D), stddev = 1.0, dtype=tf.float32))

#with sess.as_default():
#print Y.eval()
ED = ed.Euclid_Distance(X, Y, K, B)
dist = ED.cal_Euclid_dis()
cluster = tf.argmin(dist, 1)
correspond_cluster = tf.gather(Y,cluster)
offset = tf.sub(X, correspond_cluster)
loss = tf.reduce_sum(tf.square(offset))
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
training_epochs = 100

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

for epoch in range(training_epochs):
    loss_np, _ = sess.run([loss, train_op], feed_dict={X: X_tmp})
    print loss_np

