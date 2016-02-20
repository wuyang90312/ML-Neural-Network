import numpy as np
from collections import OrderedDict
from operator import itemgetter

class Training_Size:
    # Compare the accuracy of KNN algorithm based on different size of a training set
    def __init__(self, file_name):
        self.name = file_name

    def extract_zip(self, in1, in2, in3, in4):
        with np.load(self.name) as data:
            x, self.t = data[in1], data[in2]
            x_eval , self.t_eval = data[in3], data[in4]    
        return x, self.t, x_eval, self.t_eval
        
    def calculate_diff(self, matrix_1, matrix_2): 
		# calculate the error distance b/t two matrices 
        # elementwise subtraction and square, and then sum all the elements up
        result = np.sum(np.square(matrix_1 - matrix_2)) 
        return result;

    def label(self, k, unknown, samples):
    	# compare the validation elements to the sample space 
    	# and return the result with minimum value 

        store = {}
        # loop all the elements in sample space, store the offset
        for idx, matrix in enumerate(samples):  
            offset = self.calculate_diff(unknown, matrix)
            store[idx] = offset

        # sort the dictionary according to the value(offset)
        # choose the idx(key) of the dictionary element corresponding to the smallest offset, 
        # and return its related t value
        sort = OrderedDict(sorted(store.items(), key=itemgetter(1))) 
        return self.t[sort.keys()[0]]
    
    def find_near_neighbors(self, k, N, training, validation): 
        # look for the nearest neighbor and label the test sample
        estimates=[]

        # loop and label every training sample
        for matrix in validation: 
            # take the preceeding N elements as training samples
            estimates.append(self.label(k, matrix, training[:N]))  
            
        result = 0
        # loop and count the total amount of differences
        for idx, estimate in enumerate(estimates):  
            if(estimate != self.t_eval[idx]):
                result +=1
        return result
        
    def compare_size(self, k, training, validation): 
        # change the size of the training set
        N = [5, 50, 100, 200, 400, 800] 
        result = []

        for n in N:
            result.append([n, self.find_near_neighbors(k, n, training, validation)])

        print 'N' + '\t' + 'Validation Errors'
        for i in range(len(N)):
        	print repr(N[i]) + '\t' + repr(result[i][1])

def main():  

   	train = Training_Size("TINY_MNIST.npz");
   	x, t, x_eval, t_eval = train.extract_zip("x", "t", "x_eval", "t_eval")

   	k = 1
   	train.compare_size(k, x, x_eval)

main()