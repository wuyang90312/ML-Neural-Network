import numpy as np
import tensorflow as tf
import plot_generator as plot
import Euclid_Distance as ed # import functions from local
from utils import *

class MoG:
    def __init__(self, file_name):
        self.file_name = file_name

    def cal_loss(self, K, D, B):
        X_data = np.load('data2D.npy')
        # Normailize the training data        
        mean = X_data.mean()
        dev = X_data.std()
        X_tmp = (X_data - mean)/ dev 

        X = tf.placeholder(tf.float32, [None, D], name='dataset')
        Y = tf.Variable(tf.random_normal(shape = (K,D), stddev = 1.0, dtype=tf.float32))


        # not sure how to initialie pi and sigma??????
        # Carry me God Yang!!!!!!!!!!!!!!!!!!
        log_pi = logsoftmax(tf.constant([[0.2, 0.3, 0.4]], dtype=tf.float32))
        # log_pi = logsoftmax(tf.random_normal(shape = (1,K), mean = 0.2, stddev = 0.01, dtype=tf.float32))
        # exp_sigma = tf.exp(tf.random_normal(shape = (1,K), mean = 0.3, stddev = 0.01, dtype=tf.float32))
        exp_sigma = tf.exp(tf.constant([[0.3,0.4,0.3]],dtype=tf.float32))

        ED = ed.Euclid_Distance(X, Y, D)
        dist = ED.cal_Euclid_dis()
        cost = reduce_logsumexp(-0.5 * tf.div(dist, exp_sigma) + log_pi - tf.log(tf.sqrt(2 * pi * exp_sigma)))
        loss = -tf.reduce_sum(cost)


        learning_rate =  0.01
        epsilon = 1e-5
        beta1 = 0.9
        beta2 = 0.99
        training_epochs = 2000

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon)
        train_op = optimizer.minimize(loss)

        sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        sess.run(init)

        res_loss = []
        record, loss_prv = 0, 0
        for epoch in range(training_epochs):
            loss_np, _ , mu= sess.run([loss, train_op, Y], feed_dict={X: X_tmp})
            res_loss.append(loss_np)
            if record == 200:
                break
            elif loss_prv == loss_np:
                record += 1
            else:
                loss_prv = loss_np
            if(epoch % 20 == 0):
                print 'epoch', epoch
                print loss_np
                print mu

        return res_loss

K = 3
B = 10000
D = 2
pi = np.pi

mog = MoG("data2D.npy")
res_loss = mog.cal_loss(K, D, B)

plot.plot_loss(res_loss)
