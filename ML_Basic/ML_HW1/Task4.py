import numpy as np
import matplotlib.pyplot as plt

class Linear_Reg:
    # With only multiple variables: 1, x, x^2, x^3, x^4, x^5 
   
    def __init__(self, x, y, size):
        self.x_input = x
        self.y = y
        self.size = size
        
    def linear_reg(self, itr, rate):
        length = len(self.x_input)
        # create a n x 6 matrix
        x = np.ones(shape = (length, self.size)) 
        for i in range(self.size):
            sample =  np.power(self.x_input, i)

            # Normalization
            mean = np.mean(sample)
            variant = np.std(sample)
            # with zero variant, the normalization can be omitted
            if variant == 0: 
                variant = 1
                mean = 0
            # column two are copied with elements of train_x, 
            # and normalize each dimension(column)
            x[:, i:i+1] = (sample - mean) * (1 / variant) 

        
        # Updata the weight by using the gradient descent method
        # w = np.random.rand(self.size, 1) #initialize the weight randomly
        w = np.zeros(shape = (self.size, 1)) # initialize the weight
        for i in range(itr):
            # calculate the gradient for the entire training set
            # and update the weight once in each iteration
            grad = 2 * ((x.T).dot(x).dot(w) - (x.T).dot(self.y)) / length 
            w = w - grad * rate 
        
        # Weight is achieved via the analytic method
        w1 = np.zeros(shape = (self.size, 1)) 
        w_local_min = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(self.y)) 
        
        train_y = x.dot(w)
        train_y_analytic = x.dot(w_local_min)
        return train_y, train_y_analytic
        
def main(): 

    train_x = np.linspace (1.0 , 10.0 , num =100) [:, np.newaxis]
    train_y = np.sin(train_x) +0.1 * np.power( train_x , 2) + 0.5 * np.random.randn (100 , 1)

    linear_reg = Linear_Reg(train_x, train_y, 6) # the feature space has a rank of 6
    
    numItr = 50000
    learningRate = 0.05
    train_y_fn, train_y_analytic = linear_reg.linear_reg(numItr, learningRate)

    # Plot the training data and the fitted curve
    plt.plot(train_x, train_y_fn, 'g-', label = 'Fitted Curve')
    plt.plot(train_x, train_y, 'bo', label = 'Training Data')
    plt.plot(train_x, train_y_analytic, 'r--', label = 'Analytic Curve')
    plt.xlabel('train_x')
    plt.ylabel('train_y')
    plt.legend(loc = 'upper left', numpoints = 1)
    plt.show()

main()