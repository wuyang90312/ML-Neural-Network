import numpy as np
from matplotlib.pyplot import *

def plot_loss(loss):
    fig = figure(1)
    plot(loss, 'b')
    xlabel('Number of updates')
    ylabel('Loss')
    show()

def plot_cluster(min_idx, X_data, mu, K):
    fig = figure(2)
    fig_name = repr(K) + '_cluster'
    colors = ["r","b","g","y","m"]
    for i in range(K):
        col = colors[i]
        data = X_data[np.where(min_idx == i), :]
        data = data[0, :]
        scatter(data[:, 0], data[:, 1], c = col, alpha = 0.3)
        
    for i in range(K):
        scatter(mu[i, 0], mu[i, 1], c = 'white', alpha = 1, marker = '+', s = 100)
    savefig(fig_name)
    close()

def plot_valid_cluster(min_idx, X_data, mu, K):
    fig = figure(2)
    fig_name = repr(K) + '_valid_cluster'
    colors = ["r","b","g","y","m"]
    for i in range(K):
        col = colors[i]
        data = X_data[np.where(min_idx == i), :]
        data = data[0, :]
        scatter(data[:, 0], data[:, 1], c = col, alpha = 0.3)
        
    for i in range(K):
        scatter(mu[i, 0], mu[i, 1], c = 'white', alpha = 1, marker = '+', s = 100)
    savefig(fig_name)
    close()
