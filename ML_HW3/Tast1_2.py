import numpy as np
import tensorflow as tf
import Euclid_Distance as ed

max_num_updates = 1000
X = np.load('data2D.npy')
K = 3
D = X[0].size
Y = np.zeros(shape=[K, D])
Y[0:3,0] = 1.15 * np.random.randn(3) + 0.10
Y[0:3,1] = 1.59 * np.random.randn(3) - 1.51

print Y

# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)

# print clus




with tf.Session():
    ED = ed.Euclid_Distance(X, Y)
    # cluster = tf.argmin(ED.cal_Euclid_dis().eval(), 1)
    # print cluster.get_shape()

    print ED.cal_Euclid_dis().eval()