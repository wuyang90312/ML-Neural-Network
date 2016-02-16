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

    def constru_NN(self, images_val, labels_val, images_test, labels_test, deg_learn, layer, unit):
        # Place the input image and corresponding label into the placeholders
        X = tf.placeholder(tf.float32, [None, 784], name='image')
        Y = tf.placeholder(tf.float32, [None, 10], name='label')

        dropout_rate = tf.placeholder("float")


        
        # Set up the 4 weight matrices: bias/input->hidden unit, bias/hidden unit->output
        b1 = tf.Variable(tf.zeros([unit[0]]), name='weight_b2h')
        w1 = tf.Variable(tf.truncated_normal([784, unit[0]], stddev=0.1), name='weight_i2h1')
        
        
        if layer >= 2:
            b2 = tf.Variable(tf.zeros([unit[1]]), name='weight_h12h2')
            w2 = tf.Variable(tf.truncated_normal([unit[0], unit[1]], stddev=0.1), name='weight_h12h2')
        else:
            b2 = tf.zeros([unit[1]])
            w2 = tf.diag(tf.ones([unit[1]]))

         
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
        training_epochs = 2000



        # Set up 2 math operations: input -> hidden unit, hidden unit -> output
        layer1_result = tf.nn.relu(tf.add(tf.matmul(X/255, w1), b1))
        dropout_rate_extra = dropout_rate
        
        layer1_result_dropout = tf.nn.dropout(layer1_result, dropout_rate)
        layer2_result = tf.nn.relu(tf.add(tf.matmul(layer1_result_dropout, w2), b2))
        
            
        if layer < 2:
            dropout_rate_extra = 1.0
        layer2_result_dropout = tf.nn.dropout(layer2_result, dropout_rate_extra)
        layer3_result = tf.nn.relu(tf.add(tf.matmul(layer2_result_dropout, w3), b3))

        if layer < 3:
            dropout_rate_extra = 1.0    
        layer3_result_dropout = tf.nn.dropout(layer3_result, dropout_rate_extra)
            
        logits = tf.add(tf.matmul(layer3_result_dropout, w4), b4)




        # start to train the model, maximize the log-likelihood
        cost_batch = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        cost = tf.reduce_mean(cost_batch)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cost)
        


        log_likelihood_train = -tf.reduce_sum(cost_batch)
        layer1_result_valid = tf.nn.relu(tf.add(tf.matmul(images_val/255, w1), b1))
        layer2_result_valid = tf.nn.relu(tf.add(tf.matmul(layer1_result_valid, w2), b2))
        layer3_result_valid = tf.nn.relu(tf.add(tf.matmul(layer2_result_valid, w3), b3))
        logits_valid = tf.add(tf.matmul(layer3_result_valid, w4), b4)
        log_likelihood_valid = -tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits_valid, labels=labels_val))
        
        train_predition = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(logits_valid)
        layer1_result_valid = tf.nn.relu(tf.add(tf.matmul(images_test/255, w1), b1))
        layer2_result_valid = tf.nn.relu(tf.add(tf.matmul(layer1_result_valid, w2), b2))
        layer3_result_valid = tf.nn.relu(tf.add(tf.matmul(layer2_result_valid, w3), b3))
        logits_test = tf.add(tf.matmul(layer3_result_valid, w4), b4)
        test_prediction = tf.nn.softmax(logits_test)
        
        sess = tf.InteractiveSession()
        init = tf.initialize_all_variables()
        sess.run(init)
        result = np.zeros(shape=(5, training_epochs))
        
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
            
            if(epoch % 1 == 0):
                print ("Epoch:%04d, Train Accuracy=%0.4f, Eval Accuracy=%0.4f, log_likelyhood_train:%0.6f, log_likelyhood_val:%0.6f" % 
                    (epoch+1, accuracy_train, accuracy_val, likelihood_train, likelihood_val))

        print sess.run(w3)
        print sess.run(b3)
        print sess.run(w2)
        print sess.run(b2)
        print sess.run(w1)
        print sess.run(b1)
        #Predict Test Error
        accuracy_test = self.accuracy(test_prediction.eval(), labels_test)
        print ("Test Accuracy=%0.4f" % (accuracy_test))
        print 'Test_Error: ' + repr((100 - accuracy_test) * 2720 / 100)

        plt.plot(result[0,:], result[1,:],'b-', label='Validation Error')
        plt.plot(result[0,:], result[2,:],'r-', label = 'Training Error')
        plt.legend(loc = 'upper right', numpoints = 1)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Number of Errors')
        plt.show()

        plt.plot(result[0,:], result[3,:],'b-', label = 'Log-likelihood_Training')
        plt.plot(result[0,:], result[4,:],'r-', label = 'Log-likelihood_Validation')
        plt.legend(loc = 'lower right', numpoints = 1)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Log-likelihood')
        plt.show()
  
with np.load("notMNIST.npz") as data:
    images , labels = data["images"], data["labels"]

images_in = images.reshape(784,18720).T.astype("float32")
labels_in = np.zeros([18720,10]).astype("float32")
index = 0
for i in labels:
    labels_in[index, i] = 1
    index +=1

# Random seed on time, generate the hyperparameters
random.seed(datetime.now())
drop_out = random.randint(5,10)*0.1     # drop-out rate is either 0.5 - 0.9(drop) or 1.0 (non-drop)
deg_learn = random.uniform(-4,-2)       # exponent of the learning rate based on 10
layer = random.randint(1,3)             # uniformly select the number of the NN layer from 1 to 3

unit_amount=[0,0,0]                     # Generate the number of units at each layer according to the total layers
for i in range(3):
    if i < layer:
        unit_amount[i] = random.randint(100, 500);
    else:
        unit_amount[i] = unit_amount[i-1]
print drop_out, deg_learn, layer, unit_amount

cnn = NN(images_in[:15000,:], labels_in[:15000,:], drop_out)
cnn.constru_NN(images_in[15000:16000,:], labels_in[15000:16000,:], images_in[16000:18720,:], labels_in[16000:18720,:], deg_learn, layer, unit_amount)
