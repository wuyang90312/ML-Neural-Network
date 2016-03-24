import numpy as np
import tensorflow as tf
import plot_generator as plot
import Euclid_Distance as ed # import functions from local
from utils import *
import MoG as mog

def MoG_validation(K):
	MoG_valid = mog.MoG("data2D.npy")
	_, X_data, mu, _, sigma_2, log_pi = MoG_valid.cluster(K, D, B, 1.0/3.0)

	loss_valid = MoG_valid.cal_loss(MoG_valid.validation.astype(np.float32), mu, D, log_pi, sigma_2)
	min_idx = MoG_valid.cal_min_idx(X_data, mu, D)

	data = tf.ones(shape = [B,])
	division = tf.unsorted_segment_sum(data, min_idx, K, name=None)
	with tf.Session():
		print 'K =', K, ',loss_validation:', loss_valid.eval(), "Proportion:",division.eval()/10000
		plot.plot_cluster(min_idx.eval(), X_data, mu, K)

B = 10000
D = 2
for i in range(1, 6):
	MoG_validation(i)