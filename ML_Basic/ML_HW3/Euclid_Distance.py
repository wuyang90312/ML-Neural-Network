'''
Calculate the Squared Euclidean Distance b/t 
pairs of points by the vectorized tensorflow,
Without utilization of loop
'''

import numpy as np
import tensorflow as tf

'''
The format of the return value(solution):
|  D(x1, y1) D(x1, y2) ... D(x1, yK)   |
|  D(x2, y1) D(x2, y2) ... D(x2, yK)   |
|  ...          ...    ...    ...      |   
|  D(xB, y1) D(xB, y2) ... D(xB, yK)   |
'''
class Euclid_Distance:
    def __init__(self, X, Y, D):
        self.X = X
        self.Y = Y
        self.D = D
        
    def cal_Euclid_dis(self):
        x2 = self.cal_square(self.X)
        y2 = self.cal_square(self.Y)
        xy = self.cal_XY(self.X,self.Y)
        
        Euclid_dist = x2 + tf.transpose(y2) - 2*xy
        return Euclid_dist

    def cal_square(self, X):
        square = tf.square(X)
        result = tf.matmul(square, tf.ones(shape=[self.D ,1]))
        return result

    def cal_XY(self, X, Y):
        result = tf.matmul(X,Y, False, True)
        return result

'''
#test case
X= np.array([[1,2], [2,3], [3,4]], dtype = np.float32)
Y= np.array([[0,1], [1,2]], dtype = np.float32)

with tf.Session():
    ED = Euclid_Distance(X, Y, 2)
    print ED.cal_Euclid_dis().eval()
'''