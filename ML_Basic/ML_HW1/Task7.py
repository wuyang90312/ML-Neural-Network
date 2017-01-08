import numpy as np

class Eucld_Linear_Reg:

    def __init__(self, file_name):
        self.name = file_name
     
    def extract_zip(self, in1, in2, in3, in4):
        with np.load(self.name) as data:
            self.x, self.t = data[in1], data[in2]
            x_eval , t_eval = data[in3], data[in4]    
        return x_eval, t_eval
        
    def linear_reg(self, epoch, rate, threshold, N, lamda):
        x = self.x[:N, :]
        B = -self.t[:N, :] + threshold
        # the size of minibatch is 50
        size_mini = 50

        w = np.zeros(shape = (len(x[1, :]), 1)) # initialize the weight
        for i in range(epoch): # iterate towards to a small gradient
            for i in range(N / size_mini):
                # calculate the gradient of Euclidean Cost
                grad = self.euclid_grad(x[i*size_mini:(i+1)*size_mini,:], B[i*size_mini:(i+1)*size_mini,:], w, lamda, size_mini)
                w = w - grad * rate

        return w
        
    def estimation(self, w, validation):
        estimate = validation.dot(w)
        result = (estimate >= 0)
        return result

    def euclid_grad(self, x, b, w, lamda, N):
        grad = ((x.T).dot(b) + (x.T).dot(x).dot(w)) / N + lamda * w
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
 
    def compare_size(self, x_eval, t_eval): # change the size of the training set
        lamda = [0, 0.0001, 0.001, 0.01, 0.1, 0.5]
        learningRate = 0.05
        numEpoch = 2000
        threshold = 0.5
        train_size = 50

        print 'lambda'+ '\t' + 'Validation Error'
        for i in lamda:
            coefficient = self.linear_reg(numEpoch, learningRate, threshold, train_size, i)
            estimate = self.estimation(coefficient, x_eval)
            print repr(i) + '\t' + repr(self.count_err(t_eval, estimate))

def main():

    linear_reg = Eucld_Linear_Reg("TINY_MNIST.npz") # the feature space has a rank of 6
    x_eval, t_eval = linear_reg.extract_zip("x", "t", "x_eval", "t_eval")
    linear_reg.compare_size(x_eval, t_eval)

main()
