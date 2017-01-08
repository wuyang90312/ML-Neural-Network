import numpy as np
import matplotlib.pyplot as plt

class Eucld_Linear_Reg:

    def __init__(self, file_name):
        self.name = file_name
     
    def extract_zip(self, in1, in2, in3, in4):
        with np.load(self.name) as data:
            self.x, self.t = data[in1], data[in2]
            self.x_eval , self.t_eval = data[in3], data[in4]    
        
    def linear_reg(self, rate, b, N, epoch):
        result = np.zeros(shape = (3, epoch))
        X = self.x[:N, :]
        B = -self.t[:N, :] + b
        T = self.t[:N, :]        
        # the size of minibatch is 50
        size_mini = 50

        w = np.zeros(shape = (len(X[0,:]), 1)) # initialize the weight
        for j in range(epoch):
            for i in range(N / size_mini):
                # calculate the gradient of Euclidean Cost and update the weight for each minibatch
                grad = self.euclid_grad(X[i*size_mini:(i+1)*size_mini,:], B[i*size_mini:(i+1)*size_mini,:], w, size_mini) 
                w = w - grad * rate

            estimate_validation = self.estimation(w, self.x_eval)
            estimation_training = self.estimation(w, X)

            result[0,j] = j + 1
            result[1,j] = self.count_err(self.t_eval,estimate_validation)
            result[2,j] = self.count_err(T, estimation_training)

        return result
        
    def estimation(self, w, validation):
        estimate = validation.dot(w)
        result = (estimate >= 0)
        return result

    def euclid_grad(self, x, b, w, N):
        grad = (x.T).dot(b)+(x.T).dot(x).dot(w) / N
        return grad
    
    def count_err(self, t_eval, estimate):
        size = len(t_eval)
        compare = np.zeros((size,2))
        compare[:,:1] = t_eval
        compare[:,1:2] = estimate
        count = 0
        for i in range(size):
            if(compare[i,0] != compare[i,1] ):
                count += 1
        return count
   
def main():  

    train_size = 50
    b = 0.5
    numEpoch = 10000
    learningRate = 0.05

    linear_reg = Eucld_Linear_Reg("TINY_MNIST.npz")
    linear_reg.extract_zip("x", "t", "x_eval", "t_eval")
    result = linear_reg.linear_reg(learningRate, b, train_size, numEpoch)  

    plt.plot(result[0,:], result[1,:],'b-', label='Validation Error')
    plt.plot(result[0,:], result[2,:],'r-', label = 'Training Error')
    plt.legend(loc = 'upper right', numpoints = 1)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Number of Errors')
    plt.show()

main()
