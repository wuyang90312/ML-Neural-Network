import tensorflow as tf
import numpy as np

class Log_Probability:
    def __init__(self, X, Y, sigma, D):
        self.X = X
        self.Y = Y
        self.sigma = sigma
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

    def cal_Term1(self, sigma):
        return -tf.log(tf.sqrt(2 * pi * tf.square(sigma)))

    def cal_Term2(self, ed, sigma):
        return -0.5 * tf.div(ed, tf.square(sigma))

    def cal_log_probability(self):
    	ed = self.cal_Euclid_dis()
    	log_prob = self.cal_Term1(sigma) + self.cal_Term2(ed, sigma)
    	return log_prob
        '''
        The format of the return value(solution):
        |  logN(x1;u1,sig1^2) logN(x1;u2,sig2^2) ... logN(x1;uK,sigK^2)   |
        |  logN(x2;u1,sig1^2) logN(x2;u2,sig2^2) ... logN(x2;uK,sigK^2)   |
        |         ...                ...         ...         ...          |   
        |  logN(xB;u1,sig1^2) logN(xB;u2,sig2^2) ... logN(xB;uK,sig1K^2)  |
        '''
'''
# test case
# logN(x;mu, sig^2) = - log(square(sqrt(2*pi*sig^2))) - 1/2*sgi^2 * ((x-mu)(x-mu).T)
# X = np.array([[1,2], [2,3], [3,4]], dtype = np.float32)
X_data = np.load('data2D.npy')
X = tf.placeholder(tf.float32, [None, 2], name='dataset')
Y = np.array([[0,1], [1,2],[0,2]], dtype = np.float32)
sigma = tf.constant([[0.4, 0.5, 0.3]], dtype = tf.float32)	
pi = np.pi

LP = Log_Probability(X, Y, sigma, 2)
result = LP.cal_log_probability()

sess = tf.Session()
res = sess.run([result], feed_dict={X:X_data})
print res

# with tf.Session():
# 	LP = Log_Probability(X, Y, sigma, 2)
# 	print LP.cal_log_probability().eval()
'''