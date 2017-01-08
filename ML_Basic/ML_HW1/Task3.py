import numpy as np
import matplotlib.pyplot as plt

class Linear_Reg:
    # With only one variable
   
    def __init__(self, x, y, size):
        self.x_input = x
        self.y = y
        self.size = size
        
    def linear_reg(self, itr, rate):
        # create a n x 2 matrix: column one are all ones for w0
        # column two are copied with elements of train_x
        x = np.ones(shape = (len(self.x_input), self.size)) 
        x[:, 1:2] = self.x_input 
        
        w = np.zeros(shape=(self.size,1)) # initialize the weight
        for i in range(itr):
            # calculate the gradient and update weight in each epoch
            grad = 2 * ((x.T).dot(x).dot(w)-(x.T).dot(self.y)) / len(self.x_input) 
            w = w - grad * rate

        return w
        
def main():
           
    train_x = np.linspace (1.0 , 10.0 , num =100) [:, np.newaxis]
    train_y = np.sin(train_x) + 0.1 * np.power( train_x , 2) + 0.5 * np.random.randn (100 , 1)

    numEpoch = 10000
    learningRate = 0.005

    linear_reg = Linear_Reg(train_x, train_y, 2)
    weight = linear_reg.linear_reg(numEpoch, learningRate)
    train_y_fn = weight[0,0] + weight[1, 0] * train_x

    plt.plot(train_x, train_y_fn, label = 'Fitted Line')
    plt.plot(train_x, train_y,'ro', label = 'Training Data')
    plt.xlabel('train_x')
    plt.ylabel('train_y')
    plt.legend(loc = 'upper left', numpoints = 1)
    plt.show()

main()