import numpy as np
import tensorflow as tf
import plot_generator as plot
import Euclid_Distance as ed # import functions from local
from utils import *

pi = np.pi
class MoG:
    def __init__(self, file_name):
        self.file_name = file_name

    def cal_min_idx(self, X, Y, D):
        X_input = tf.constant(X, dtype=tf.float32)
        Y_input = tf.constant(Y, dtype=tf.float32)
        ED_min = ed.Euclid_Distance(X_input, Y_input, D)
        ed_min = ED_min.cal_Euclid_dis()
        min_idx = tf.argmin(ed_min, 1)
        return min_idx

    def cal_loss(self, X, Y, D, log_pi, exp_sigma):
        ED = ed.Euclid_Distance(X, Y, D)
        dist = ED.cal_Euclid_dis()
        cost = reduce_logsumexp(tf.div(-dist, 2*exp_sigma) + log_pi - tf.log(tf.sqrt(2 * pi * exp_sigma)))
        loss = -tf.reduce_sum(cost)
        return loss

    def cluster(self, K, D, B, portion=0):
        X_data = np.load('data2D.npy')  

        seperation = int((1-portion)*B)
        self.validation = X_data[seperation:,:]
        X_train = X_data[:seperation,:]      

        # Initialize centoroid, pi_k and sigma
        X = tf.placeholder(tf.float32, [None, D], name='dataset')
        Y = tf.Variable(tf.random_normal(shape = (K,D), dtype=tf.float32))
        pi_k = tf.Variable(tf.random_normal(shape=(1,K), dtype=tf.float32))
        sigma = tf.Variable(tf.random_normal(shape=(1,K), dtype=tf.float32))

        log_pi = logsoftmax(pi_k)
        exp_sigma = tf.exp(sigma)

        loss = self.cal_loss(X, Y, D, log_pi, exp_sigma)

        learning_rate =  0.005
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
        record, loss_prv = 0, 0
        for epoch in range(training_epochs):
            _, loss_np, mu_final, pi_np, sigma_square, pi_log= sess.run([train_op, loss, Y, pi_k, exp_sigma, log_pi], feed_dict={X: X_train})
            res_loss.append(loss_np)
            if record == 200:
                break
            elif loss_prv == loss_np:
                record += 1
            else:
                loss_prv = loss_np
            '''
            if(epoch % 100 == 0):
                print 'epoch', epoch
                print 'loss', loss_np
            '''
        '''
        print 'loss_training:', loss_np
        print 'centoroid:', mu_final        
        pi_np = tf.exp(pi_np) / tf.reduce_sum(tf.exp(pi_np))
        print 'pi_k:', pi_np.eval()
        print 'sigma_square:', sigma_square
        '''
        min_idx = self.cal_min_idx(X_train, mu_final, D).eval()

        return res_loss, X_data, mu_final, min_idx, sigma_square, pi_log

'''
K = 3
B = 10000
D = 2

mog = MoG("data2D.npy")
res_loss, X_plot, mu_plot, min_idx, _, _ = mog.cluster(K, D, B)
print X_plot.shape, min_idx.shape
plot.plot_loss(res_loss)
plot.plot_cluster(min_idx, X_plot, mu_plot, K)
'''
