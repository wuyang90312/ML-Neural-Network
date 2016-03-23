import numpy as np
import tensorflow as tf
import plot_generator as plot
import Euclid_Distance as ed # import functions from local
from utils import *

class MoG:
    def __init__(self, file_name):
        self.file_name = file_name

    def cal_min_idx(self, X, Y, D):
        X_data = tf.constant(X, dtype=tf.float32)
        Y_final = tf.constant(Y, dtype=tf.float32)
        ED_min = ed.Euclid_Distance(X_data, Y_final, D)
        ed_min = ED_min.cal_Euclid_dis()
        min_idx = tf.argmin(ed_min, 1)
        return min_idx

    def cal_loss(self, K, D, B):
        X_data = np.load('data2D.npy')
        # Normailize the training data        
        mean = X_data.mean()
        dev = X_data.std()
        X_norm = (X_data - mean)/ dev 

        # Initialize centoroid, pi_k and sigma
        X = tf.placeholder(tf.float32, [None, D], name='dataset')
        Y = tf.Variable(tf.random_normal(shape = (K,D), stddev=1.0, dtype=tf.float32))
        pi_k = tf.Variable(tf.random_normal(shape=(1,K), dtype=tf.float32))
        sigma = tf.Variable(tf.random_normal(shape=(1,K), dtype=tf.float32))

        log_pi = logsoftmax(pi_k)
        exp_sigma = tf.exp(sigma)

        ED = ed.Euclid_Distance(X, Y, D)
        dist = ED.cal_Euclid_dis()
        cost = reduce_logsumexp(tf.div(-dist, 2*exp_sigma) + log_pi - tf.log(tf.sqrt(2 * pi * exp_sigma)))
        loss = -tf.reduce_mean(cost)

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
            _, loss_np, mu_norm, pi_np, sigma_square = sess.run([train_op, loss, Y, pi_k, exp_sigma], feed_dict={X: X_norm})
            res_loss.append(loss_np)
            if record == 100:
                break
            elif loss_prv == loss_np:
                record += 1
            else:
                loss_prv = loss_np
  
            if(epoch % 100 == 0):
                print 'epoch', epoch
                print 'loss', loss_np

        print 'loss:', loss_np
        print 'centoroid:', mu_norm        
        pi_np = tf.exp(pi_np) / tf.reduce_sum(tf.exp(pi_np))
        print 'pi_k:', pi_np.eval()
        print 'sigma_square:', sigma_square

        min_idx = self.cal_min_idx(X_norm, mu_norm, D).eval()

        return res_loss, X_data,mu_norm*dev+mean, min_idx

K = 3
B = 10000
D = 2
pi = np.pi

mog = MoG("data2D.npy")
res_loss, X_plot, mu_plot, min_idx = mog.cal_loss(K, D, B)

plot.plot_loss(res_loss)
plot.plot_cluster(min_idx, X_plot, mu_plot, K)
