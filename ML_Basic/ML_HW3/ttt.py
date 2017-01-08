import tensorflow as tf

X = tf.constant([[1,2],[2,3],[3,4]], dtype = tf.float32)
X2 = tf.square(X)
diag = tf.matmul(X2, tf.ones([2,1]))

res = tf.diag(diag)
res = tf.reshape(res, [3,3])

with tf.Session():
	print diag.get_shape()
	print res.eval()