import numpy as np
import tensorflow as tf
import plot_generator as plot
import Euclid_Distance as ed 
from utils import *
import MoG as mog

def MoG_validation(K):
	MoG_valid = mog.MoG("data100D.npy")
	# _, X_data, mu, _, sigma_2, log_pi, pi_np = MoG_valid.cluster(K, D, B, 1.0/3.0)
	_, X_data, mu, _, sigma_2, log_pi, pi_np = MoG_valid.cluster(K, D, B)


	loss_valid = MoG_valid.cal_loss(MoG_valid.validation.astype(np.float32), mu, D, log_pi, sigma_2)
	min_idx = MoG_valid.cal_min_idx(X_data, mu, np.sqrt(sigma_2), pi_np, D)

	data = tf.ones(shape = [B,])
	division = tf.unsorted_segment_sum(data, min_idx, K, name=None)

	with tf.Session():
		print 'loss_validation:', loss_valid.eval()
		print 'Total Proportion:', division.eval()/10000
		# print 'Total Proportion:', division.eval()
		plot.plot_cluster(min_idx.eval(), X_data, mu, K)
		
B = 10000
D = 100
for i in range(1, 6):
	MoG_validation(i)