import numpy as np
import tensorflow as tf
import plot_generator as plot
import Euclid_Distance as ed 
from utils import *
import Log_Posterior as LPosterior
import Log_Probability as LProbability

pi = np.pi
class MoG:
    def __init__(self, file_name):
        self.file_name = file_name

    def cal_min_idx(self, X, Y, sigma, pi_k, D):
        X_input = tf.constant(X, dtype=tf.float32)
        Y_input = tf.constant(Y, dtype=tf.float32)
        sigma_input = tf.constant(sigma, dtype=tf.float32)
        LP = LPosterior.Log_Posterior(X_input, Y_input, sigma_input, pi_k, D)
        post = LP.cal_log_posterior() 
        min_idx = tf.argmax(post, 1)   
        return min_idx

    def cal_loss(self, X, Y, D, log_pi, exp_sigma):
        ED = ed.Euclid_Distance(X, Y, D)
        dist = ED.cal_Euclid_dis()
        cost = reduce_logsumexp(tf.div(-dist, 2*exp_sigma) + log_pi - (D/2)* tf.log(2 * pi * exp_sigma))
        loss = -tf.reduce_sum(cost)
        return loss

    def cluster(self, K, D, B, portion=0):
        X_data = np.load(self.file_name).astype(np.float32) 

        seperation = int((1-portion)*B)
        self.validation = X_data[seperation:,:]
        self.train = X_data[:seperation,:]

        # Initialize centroid, pi_k and sigma
        X = tf.placeholder(tf.float32, [None, D], name='dataset')
        Y = tf.Variable(tf.random_normal(shape = (K,D), dtype=tf.float32))
        pi_k = tf.Variable(tf.random_normal(shape=(1,K), dtype=tf.float32))
        sigma = tf.Variable(tf.random_normal(shape=(1,K), dtype=tf.float32))

        log_pi = logsoftmax(pi_k)
        exp_sigma = tf.exp(sigma)
        loss = self.cal_loss(X, Y, D, log_pi, exp_sigma)

        # setting the hyperparameter for gradient descent
        learning_rate =  0.01
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
            loss_train, _, mu_final, pi_final, sigma_square, pi_log = sess.run([loss, train_op, Y, pi_k, exp_sigma, log_pi], feed_dict={X: self.train})
            res_loss.append(loss_train)
            if record == 500:
                break
            elif loss_prv == loss_train:
                record += 1
            else:
                loss_prv = loss_train
            
            if(epoch % 100 == 0):
                print 'epoch', epoch
                print 'loss', loss_train
            
        print 'K =', K
        print 'loss_training:', loss_train
        print 'centroid:', mu_final        
        print 'pi_k:', tf.exp(pi_log).eval()
        print 'sigma_square:', sigma_square
        
        min_idx = self.cal_min_idx(self.train, mu_final, np.sqrt(sigma_square), tf.exp(pi_log), D).eval()

        return res_loss, X_data, mu_final, min_idx, sigma_square, pi_log, tf.exp(pi_log).eval()

# When K = 3, compute the loss function for the whole data 
K = 3
B = 10000
D = 2

mog = MoG("data2D.npy")
res_loss, X_plot, mu_plot, min_idx, _, _, pi_1 = mog.cluster(K, D, B)

plot.plot_loss(res_loss)
plot.plot_cluster(min_idx, X_plot, mu_plot, K)
