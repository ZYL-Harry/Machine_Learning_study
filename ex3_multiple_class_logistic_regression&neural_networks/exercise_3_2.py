import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import scipy.optimize as opt

'''
2.  Neural Networks by the feedforward propagation algorithm
    3 layers – an input layer(400 input units), a hidden layer(25 units) and an output layer(10 output units)
'''

# 2.1 read the prepared dataset
def read_data():
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_3/ex3/ex3data1.mat'
    data1 = loadmat(path1)
    # print(data1.keys())
    X = data1['X']
    # print(X)
    y = data1['y']
    # print(y)
    return X, y

# 2.2 visiualize the dataset
def visuial_data(X, y):
    rand_indices = np.random.choice(X.shape[0], 100)
    # print(rand_indices.shape)
    select = X[rand_indices, :]
    # print(select.shape)
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
    # print(ax.shape)
    for r in range(10):
        for c in range(10):
            show = select[10*r+c, :].reshape((20, 20)).T
            ax[r, c].matshow(show, cmap=cm.binary)
            plt.xticks([])
            plt.yticks([])
    plt.show()

# 2.3 read the trained parameter θ
def read_weights():
    path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_3/ex3/ex3weights.mat'
    data2 = loadmat(path2)
    theta1 = data2['Theta1']
    theta2 = data2['Theta2']
    return theta1, theta2

# 3 feedforward propagation
# 3.1 sigmoid function
def sigmoidfunction(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    f = 1 / (1 + np.exp(- X * theta.T)) # (5000,401)*(401,25)=(5000,25); (5000,26)*(26,10)=(5000,10)
    return f
# 3.2 feedforward propagation classification
def classification(theta1, theta2, X):
    X_new = np.c_[np.matrix(np.zeros((X.shape[0], 1))), X]
    a1 = X_new  # (5000,401)
    # z2 = theta1 * a1.T    # (25,401)*(401,5000)
    a2 = np.c_[np.matrix(np.zeros((X.shape[0], 1))), sigmoidfunction(theta1, a1)]    # (5000,26)
    a3 = sigmoidfunction(theta2, a2)    #(5000,10)
    prediction_compute = np.argmax(a3, axis=1) + 1
    return prediction_compute
# 3.3 check the accuracy
def check_accuracy(y, prediction):
    correct = np.matrix(np.zeros((y.shape[0], 1)))
    for i in range(y.shape[0]):
        if prediction[i, 0] == y[i, 0]:
            correct[i, 0] = 1
        else:
            correct[i, 0] = 0
    accuracy_compute = (np.sum(correct) / correct.shape[0]) * 100
    return accuracy_compute

if __name__ == '__main__':
    # number of labels
    num_labels = 10 # label: 1-10
    # read dataset form file
    X, y = read_data()
    # visialize dataset
    visuial_data(X, y)
    # read the trained parameter θ
    theta1, theta2 = read_weights()
    # classification
    prediction = classification(theta1, theta2, X)
    # compute the accuracy
    accuracy = check_accuracy(y, prediction)
    print('accuracy is {0}%'.format(accuracy))

