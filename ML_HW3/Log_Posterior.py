import tensorflow as tf
import numpy as np
from utils import *

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
        return tf.log(tf.div(pi_k, tf.sqrt(2 * pi * tf.square(sigma))))

    def cal_term2(self, sigma):
        ed = self.cal_Euclid_dis()
        # d = -0.5 * tf.div(ed, tf.square(sigma))
        d = tf.mul(-0.5, tf.div(ed, tf.square(sigma)))
        return d

    def cal_term3(self):
        my_tensor = self.cal_term2(self.sigma) + self.cal_term1(self.pi_k, self.sigma)
        log_sum = reduce_logsumexp(my_tensor, 1, True)
        return log_sum

    def cal_log_posterior(self):
        res = self.cal_term1(self.pi_k, self.sigma) + self.cal_term2(self.sigma) - self.cal_term3()
        return res

'''
#test case
# X = tf.constant([[1,2], [2,3],[3,4]], dtype = tf.float32)
X_data = np.load('data2D.npy')
X = tf.placeholder(tf.float32, [None, 2], name='dataset')
Y = tf.constant([[0,1], [1,2], [2,3]], dtype = tf.float32)
sigma = tf.constant([[0.3, 0.2, 0.5]], dtype = tf.float32)
pi_k = tf.constant([[0.4, 0.3, 0.3]], dtype = tf.float32)    
pi = np.pi

LP = Log_Posterior(X, Y, sigma, pi_k, 2)
result = LP.cal_log_posterior()

sess = tf.Session()
res = sess.run([result], feed_dict={X:X_data})
print res

# with tf.Session():
#     LP = Log_Posterior(X,Y,sigma,pi_k,2)
#     res = LP.cal_log_posterior()
#     print res.eval()
'''
