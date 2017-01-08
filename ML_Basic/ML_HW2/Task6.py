import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from datetime import datetime

class NN:
    def __init__(self, images, labels, drop_out):
        self.images = images
        self.labels = labels
        self.drop_out = drop_out

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])        

    def constru_NN(self, images_val, labels_val, images_test, labels_test, deg_learn, layer, unit, momentum_rate):
        # Place the input image and corresponding label into the placeholders
        X = tf.placeholder(tf.float32, [None, 784], name='image')
        Y = tf.placeholder(tf.float32, [None, 10], name='label')

        dropout_rate = tf.placeholder("float")
        
        # Set up the weight matrices: bias/input->hidden unit, hidden unit->hidden unit, bias/hidden unit->output
        b1 = tf.Variable(tf.zeros([unit[0]]), name='weight_b2h')
        w1 = tf.Variable(tf.truncated_normal([784, unit[0]], stddev=0.1), name='weight_i2h1')
        
        # If the number of layer > 2, set the weight and bias as variable, otherwise set them as 0 and identity matrix   
        if layer >= 2:
            b2 = tf.Variable(tf.zeros([unit[1]]), name='weight_h12h2')
            w2 = tf.Variable(tf.truncated_normal([unit[0], unit[1]], stddev=0.1), name='weight_h12h2')
        else:
            b2 = tf.zeros([unit[1]])
            w2 = tf.diag(tf.ones([unit[1]]))

        # If the number of layer > 3, set the weight and bias as variable, otherwise set them as 0 and identity matrix   
        if layer >= 3:
            b3 = tf.Variable(tf.zeros([unit[2]]), name='weight_2h22h3') 
            w3 = tf.Variable(tf.truncated_normal([unit[1], unit[2]], stddev=0.1), name='weight_h22h3')
        else:
            b3 = tf.zeros([unit[2]])
            w3 = tf.diag(tf.ones([unit[2]]))         

        b4 = tf.Variable(tf.zeros([10]), name='weight_h32o')
        w4 = tf.Variable(tf.truncated_normal([unit[2], 10], stddev=0.1), name='weight_h32o')
               
        # Set up the hyperparameters
        learning_rate = pow(10, deg_learn)
        training_epochs = 1500
        momentum = momentum_rate

        # Set up math operations: input -> hidden unit, hidden unit -> hidden unit and hidden unit -> output
        layer1_result = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))
        dropout_rate_extra = dropout_rate
        
        layer1_result_dropout = tf.nn.dropout(layer1_result, dropout_rate_extra)
        layer2_result = tf.nn.relu(tf.add(tf.matmul(layer1_result_dropout, w2), b2))
        if layer < 2:
            dropout_rate_extra = 1.0      
            
        layer2_result_dropout = tf.nn.dropout(layer2_result, dropout_rate_extra)
        layer3_result = tf.nn.relu(tf.add(tf.matmul(layer2_result_dropout, w3), b3))
         
        if layer < 3:
            dropout_rate_extra = 1.0   
        layer3_result_dropout = tf.nn.dropout(layer3_result, dropout_rate)
            
        logits = tf.add(tf.matmul(layer3_result_dropout, w4), b4)
        # start to train the model, maximize the log-likelihood
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        cost = tf.reduce_mean(cost_batch)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        train_step = optimizer.minimize(cost)
        
        # Compute the log-likelihood of training data and validation data
        log_likelihood_train = -tf.reduce_mean(cost_batch)
        layer1_result_valid = tf.nn.relu(tf.add(tf.matmul(images_val, w1), b1))
        layer2_result_valid = tf.nn.relu(tf.add(tf.matmul(layer1_result_valid, w2), b2))
        layer3_result_valid = tf.nn.relu(tf.add(tf.matmul(layer2_result_valid, w3), b3))
        logits_valid = tf.add(tf.matmul(layer3_result_valid, w4), b4)
        log_likelihood_valid = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_valid, labels=labels_val))
        
        # Compute the prediciton by using the softmax
        train_predition = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(logits_valid)
        layer1_result_test = tf.nn.relu(tf.add(tf.matmul(images_test, w1), b1))
        layer2_result_test = tf.nn.relu(tf.add(tf.matmul(layer1_result_test, w2), b2))
        layer3_result_test = tf.nn.relu(tf.add(tf.matmul(layer2_result_test, w3), b3))
        logits_test = tf.add(tf.matmul(layer3_result_test, w4), b4)
        test_prediction = tf.nn.softmax(logits_test)
        
        sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        sess.run(init)

        result = np.zeros(shape=(5, training_epochs))
        lilelihood_overfitting  = False
        log_likelihood_max = -float("inf")
        oscillation = 0
        
        for epoch in range(training_epochs):
            for i in xrange(300):
                batch_xs = self.images[i * 50: (i + 1) * 50]
                batch_ys = self.labels[i * 50: (i + 1) * 50]
                cost_np, _ = sess.run([cost,train_step], 
                    feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: self.drop_out})
        
            likelihood_train, predictions_train = sess.run([log_likelihood_train, train_predition], 
                feed_dict={X: self.images, Y:self.labels, dropout_rate: 1.0})
            accuracy_train = self.accuracy(predictions_train, self.labels)
            likelihood_val = log_likelihood_valid.eval()
            accuracy_val = self.accuracy(valid_prediction.eval(), labels_val)

            result[0, epoch] = epoch + 1
            result[1, epoch] = (100 - accuracy_val) * 1000 / 100
            result[2, epoch] = (100 - accuracy_train) * 15000 / 100
            result[3, epoch] = likelihood_train
            result[4, epoch] = likelihood_val  
            
            if(epoch % 50 == 0):
                print ("Epoch:%04d, Train Accuracy=%0.4f, Eval Accuracy=%0.4f, log_likelyhood_train:%0.6f, log_likelyhood_val:%0.6f" % 
                    (epoch+1, accuracy_train, accuracy_val, likelihood_train, likelihood_val))

            # When the log-likelihood reaches at maximum, use the early stopping to stop the training
            if(log_likelihood_max < likelihood_val):
                log_likelihood_max = likelihood_val
                lilelihood_oscillation = 0
            elif(log_likelihood_max - likelihood_val > 0):
	        	lilelihood_oscillation += 1

            if(lilelihood_overfitting == False):
	        	if(lilelihood_oscillation >= 15):
	        	    lilelihood_overfitting = True
	        	    print "Likelihood overfitting"
	        	    print ("Epoch:%04d, Train Accuracy=%0.4f, Eval Accuracy=%0.4f, log_likelyhood_train:%0.6f, log_likelyhood_val:%0.6f" % 
		               (epoch+1, accuracy_train, accuracy_val, likelihood_train, likelihood_val))
	        	    accuracy_test = self.accuracy(test_prediction.eval(), labels_test)
	        	    print ("Test Accuracy=%0.4f" % (accuracy_test))
	        	    print 'Test_Error: ' + repr((100 - accuracy_test) * 2720 / 100)
	        	    print ("Validation Accuracy=%0.4f" % (accuracy_val))
	        	    print 'Validation_Error: ' + repr((100 - accuracy_val) * 1000 / 100)

	        	    plt.plot(result[0,:epoch], result[1,:epoch],'b-', label='Validation Error')
	        	    plt.plot(result[0,:epoch], result[2,:epoch],'r-', label = 'Training Error')
	        	    plt.legend(loc = 'upper right', numpoints = 1)
	        	    plt.xlabel('Number of Epochs')
	        	    plt.ylabel('Number of Errors')
	        	    plt.savefig('err_early')
	        	    plt.close()
                    	        	    
	        	    plt.plot(result[0,:epoch], result[3,:epoch],'b-', label = 'Log-likelihood_Training')
	        	    plt.plot(result[0,:epoch], result[4,:epoch],'r-', label = 'Log-likelihood_Validation')
	        	    plt.legend(loc = 'lower right', numpoints = 1)
	        	    plt.xlabel('Number of Epochs')
	        	    plt.ylabel('Log-likelihood')
	        	    plt.savefig('log_early')
	        	    plt.close()
	            
        # Compute the test errors after the complete training process
        print "Complete Training:"
        accuracy_test = self.accuracy(test_prediction.eval(), labels_test)
        print ("Test Accuracy=%0.4f" % (accuracy_test))
        print 'Test_Error: ' + repr((100 - accuracy_test) * 2720 / 100)
        print ("Validation Accuracy=%0.4f" % (accuracy_val))
        print 'Validation_Error: ' + repr((100 - accuracy_val) * 1000 / 100)

    	plt.plot(result[0,:], result[1,:],'b-', label='Validation Error')
    	plt.plot(result[0,:], result[2,:],'r-', label = 'Training Error')
    	plt.legend(loc = 'upper right', numpoints = 1)
    	plt.xlabel('Number of Epochs')
    	plt.ylabel('Number of Errors')
        plt.savefig('err')
        plt.close()

    	plt.plot(result[0,:], result[3,:],'b-', label = 'Log-likelihood_Training')
    	plt.plot(result[0,:], result[4,:],'r-', label = 'Log-likelihood_Validation')
    	plt.legend(loc = 'lower right', numpoints = 1)
    	plt.xlabel('Number of Epochs')
    	plt.ylabel('Log-likelihood')
        plt.savefig('log')
        plt.close()
  


with np.load("notMNIST.npz") as data:
    images , labels = data["images"], data["labels"]

# Reshape the imput data 
images_in = images.reshape(784,18720).T.astype("float32")
labels_in = np.zeros([18720,10]).astype("float32")
index = 0
for i in labels:
    labels_in[index, i] = 1
    index +=1

# Random seed on time, generate the hyperparameters
random.seed(datetime.now())
drop_out = random.randint(1,2) * 0.5        # drop-out rate is either 0.5 (drop) or 1.0 (non-drop)
deg_learn = random.uniform(-4,-2)           # exponent of the learning rate based on 10
layer = random.randint(1,3)                 # uniformly select the number of the NN layer from 1 to 3
momentum_rate = random.randint(3,5) * 0.1   # uniformly select the momentum rate from 0.3 to 0.5

unit_amount=[0,0,0]                         # Generate the number of units at each layer according to the total layers
for i in range(3):
    if i < layer:
        unit_amount[i] = random.randint(100, 500);
    else:
        unit_amount[i] = unit_amount[i-1]

# Print the hyperparameter settings:
print "log of learning rate: %f" % (deg_learn)
print "number of layers: %d" % (layer)
if layer < 2:
    print "number of hidden units per layer: %d" % (unit_amount[0])
elif layer < 3:
    print "number of hidden units per layer: %d %d" % (unit_amount[0], unit_amount[1])
else:
    print "number of hidden units per layer: %d %d %d" % (unit_amount[0], unit_amount[1], unit_amount[2])   
print "dropout rate:" + repr(drop_out)
print "momentum rate:" + repr(momentum_rate)

nn = NN(images_in[:15000,:]/255, labels_in[:15000,:], drop_out)
nn.constru_NN(images_in[15000:16000,:]/255, labels_in[15000:16000,:], images_in[16000:18720,:]/255, labels_in[16000:18720,:], deg_learn, layer, unit_amount, momentum_rate)
