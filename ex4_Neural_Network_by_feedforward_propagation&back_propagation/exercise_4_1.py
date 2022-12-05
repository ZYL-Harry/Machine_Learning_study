import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat

'''
1. Neural Networks by feedforward
'''

# 1.1 read the dataset
def read_data():
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_4/ex4/ex4data1.mat'
    data1 = loadmat(path1)
    X = data1['X']
    y = data1['y']
    return X, y

# 1.2 visiualize the dataset
def visiualize_data(X):
    rand_indices = np.random.choice(X.shape[0], 100)
    X_select = X[rand_indices, :]
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
    for r in range(10):
        for c in range(10):
            show = X_select[10*r+c, :].reshape((20, 20)).T
            ax[r, c].matshow(show, cmap=cm.binary)
            plt.xticks([])
            plt.yticks([])
    plt.show()

# 1.3 read the trained parameter θ
def read_weights():
    path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_4/ex4/ex4weights.mat'
    data2 = loadmat(path2)
    theta1 = data2['Theta1']
    theta2 = data2['Theta2']
    return theta1, theta2

# 1.4 feedforward propagation
# 1.4.1 sigmoid function
def sigmoidfunction(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    f = 1 / (1 + np.exp(- X * theta.T)) # (5000,401)*(401,25)=(5000,25), (5000,26)*(26,10)=(5000,10)
    return f
# 1.4.2 feedforward propagation
def feedforward_propagation(theta1, theta2, X):
    X_new = np.c_[np.matrix(np.ones((X.shape[0], 1))), X]
    a1 = X_new  # (5000,401)
    a2 = np.c_[np.matrix(np.ones((X.shape[0], 1))), sigmoidfunction(theta1, a1)]    # (5000,26)
    a3 = sigmoidfunction(theta2, a2)    # (5000,10)
    prediction_compute = np.argmax(a3, axis=1) + 1
    return a3, prediction_compute
# 1.4.3 check the accuracy
def check_accuracy(y, prediction):
    correct = np.matrix(np.zeros((y.shape[0], 1)))
    for i in range(y.shape[0]):
        if prediction[i, 0] == y[i, 0]:
            correct[i, 0] = 1
        else:
            correct[i, 0] = 0
    accuracy_compute = (np.sum(correct) / correct.shape[0]) * 100
    return accuracy_compute
# 1.4.4 cost function
def costfunction(theta1, theta2, a3, y, learning_rate):
    theta1 = np.matrix(theta1)
    theta2 = np.matrix(theta2)
    a3 = np.matrix(a3)
    y = np.matrix(y)
    y_vector = np.matrix(np.zeros((y.shape[0], 10)))    # transfer lables into 10-dimensional vectors
    for i in range(y_vector.shape[0]):
        y_vector[i, y[i, 0]-1] = 1
    first = np.sum(- np.multiply(y_vector, np.log(a3)) - np.multiply((1 - y_vector), np.log(1 - a3))) / a3.shape[0] # (5000,10)*(10,5000)=(5000,10)
    second = (learning_rate / (2 * a3.shape[0])) * (np.sum(np.power(theta1, 2)) + np.sum(np.power(theta2, 2)))
    J = first + second
    return J

if __name__ == '__main__':
    # number of labels
    num_labels = 10
    # read the dataset
    X, y = read_data()
    # visiualize the dataset
    visiualize_data(X)
    # read the trained parameter θ
    theta1, theta2 = read_weights()
    # train the neural network
    a3, prediction = feedforward_propagation(theta1, theta2, X)
    # compute the accuracy
    accuracy = check_accuracy(y, prediction)
    print('accuracy is {0}%'.format(accuracy))
    # compute the cost function
    learning_rate = 1
    J = costfunction(theta1, theta2, a3, y, learning_rate)
    print('J = ', J)
