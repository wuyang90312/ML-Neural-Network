import tensorflow as tf
import numpy as np
from utils import *

pi = np.pi

'''
The log probability of posterior can be calculated by:
logP(z|x) = log( (P(x|z)P(z))/ sum(P(x|z)P(z)))
          = log(pi_k/sqrt(2*pi*sigma^2))     
               - 1/(2*sigma^2) * ((x-mu)(x-mu).T) 
               - log-sum-exp(log(pi_k/sqrt(2*pi*sigma^2)) - 1/(2*sigma^2) * ((x-mu)(x-mu).T)
'''

class Log_Posterior:
    def __init__(self, X, Y, sigma, pi_k, D):
        self.X = X
        self.Y = Y
        self.sigma = sigma
        self.pi_k = pi_k
        self.D = D
        
    def cal_Euclid_dis(self):
        x2 = self.cal_square(self.X)
        y2 = self.cal_square(self.Y)
        xy = self.cal_XY(self.X,self.Y)
        
        Euclid_dist = x2 + tf.transpose(y2) - 2*xy
        return Euclid_dist
        
    def cal_square(self, X):
        square = tf.square(X)
        result = tf.matmul(square, tf.ones(shape=[self.D ,1], dtype=tf.float32))
        return result

    def cal_XY(self, X, Y):
        result = tf.matmul(X,Y, False, True)
        return result

    def cal_term1(self, pi_k, sigma):
        return tf.log(pi_k) -  self.D/2 * tf.log(2 * pi * tf.square(sigma))

    def cal_term2(self, sigma):
        ed = self.cal_Euclid_dis()
        return tf.mul(-0.5, tf.div(ed, tf.square(sigma)))

    def cal_term3(self):
        my_tensor = self.cal_term2(self.sigma) + self.cal_term1(self.pi_k, self.sigma)
        log_sum = reduce_logsumexp(my_tensor, 1, True)
        return log_sum

    def cal_log_posterior(self):
        res = self.cal_term1(self.pi_k, self.sigma) + self.cal_term2(self.sigma) - self.cal_term3()
        return res


'''
#test case
X = np.load('data2D.npy').astype(np.float32)
Y = tf.constant([[0,1], [1,2]], dtype = tf.float32)
sigma = tf.constant([[0.3, 0.4]], dtype = tf.float32)
pi_k = tf.constant([[0.4, 0.6]], dtype = tf.float32)    

with tf.Session():
    LP = Log_Posterior(X,Y,sigma,pi_k,2)
    res = LP.cal_log_posterior()
    print res.eval()
'''