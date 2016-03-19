'''
Calculate the Squared Euclidean Distance b/t 
pairs of points by the vectorized tensorflow,
Without utilization of loop
'''

import numpy as np
import tensorflow as tf

X= np.array([[1,2], [2,3], [3,4]], dtype = np.float32)
Y= np.array([[0,1], [1,2]], dtype = np.float32)


class Euclid_Distance:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def cal_Euclid_dis(self):
        x2 = self.cal_square(self.X, self.Y.shape[0])
        y2 = self.cal_square(self.Y, self.X.shape[0])
        xy = self.cal_XY(self.X,self.Y)
        
        Euclid_dist = x2 + tf.transpose(y2) - 2*xy
        #print Euclid_dist.eval()
        return Euclid_dist

    def cal_square(self, X,size_Y):
        square = tf.matmul(X, X, False, True)
        diagonal = tf.diag(np.diag(square.eval()))
        ones = np.ones(shape = [X.shape[0], size_Y], dtype = np.float32)
        result = tf.matmul(diagonal, ones, True)
        #print result.eval()
        return result
        
    def cal_XY(self, X, Y):
        result = tf.matmul(X,Y, False, True)
        #print result.eval()
        return result

with tf.Session():
    ED = Euclid_Distance(X, Y)
    print ED.cal_Euclid_dis().eval()



