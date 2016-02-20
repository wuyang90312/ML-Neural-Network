import numpy as np
from collections import OrderedDict
from operator import itemgetter

class Neighbor_Size:
    # Compare the accuracy of KNN algorithm based on different value of k
    
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
        # and return the result with most popular value

        store = {}
        # loop all the elements in sample space, store the offset
        for idx, matrix in enumerate(samples): 
            offset = self.calculate_diff(unknown, matrix)
            store[idx] = offset
        
        # sort the dictionary according to the value(offset)
        sort = OrderedDict(sorted(store.items(), key=itemgetter(1))) 
       
        # sort the offset and choose the idx(key) of the dictionary element corresponding to the most popular data
        decision  = 0;
        for i in range(k):
            if(self.t[sort.keys()[i]] == self.t[sort.keys()[0]]):
                decision += 1
            else:
                decision -= 1
                result = self.t[sort.keys()[i]]       
        if(decision > 0):
            result = self.t[sort.keys()[0]]
        return result 

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
        
    def compare_size(self, training, validation): 
        # change the value of k
        K = [1, 3, 5, 7, 21, 101, 401]
        train_size = 800
        result = []
       
        for k in K:
            result.append([k, self.find_near_neighbors(k, train_size, training, validation)])
            
        print 'k' + '\t' + 'Validation Errors'
        for i in range(len(K)):
            print repr(K[i]) + '\t' + repr(result[i][1])

def main():
        
    neighbor = Neighbor_Size("TINY_MNIST.npz");
    x, t, x_eval, t_eval = neighbor.extract_zip("x", "t", "x_eval", "t_eval")
    neighbor.compare_size(x, x_eval)

main()