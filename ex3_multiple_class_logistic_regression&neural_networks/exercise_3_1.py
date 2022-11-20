import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import scipy.optimize as opt

'''
1. Multi-class Classification by Logistic Regression
'''

# 1.1 read the prepared dataset
def read_data():
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_3/ex3/ex3data1.mat'
    data1 = loadmat(path1)
    # print(data1.keys())
    X = data1['X']
    # print(X)
    y = data1['y']
    # print(y)
    return X, y

# 1.2 visiualize the dataset
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

# 1.3 vectorize the parameters in logistic regression
def sigmoidfunction(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    f = 1 / (1 + np.exp(- X * theta.T)) # (5000,401)*(401,1)=(5000,1)
    return f
# 1.3.1 vectorize the cost function
def vector_costfunction(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = - y.T * np.log(sigmoidfunction(theta, X)) - (1 - y).T * np.log(1 - sigmoidfunction(theta, X))   # (1,5000)*(5000,1)=(1,1)
    reg = (learning_rate / (2 * X.shape[0])) * np.sum(np.power(theta, 2))
    J = np.sum(error) / X.shape[0] + reg  # np.sum() here is used to transfer to 1 dimension
    return J
# 1.3.2 vectorize the gradient descent
def vector_gradientdescent(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    partial = np.matrix(np.zeros((theta.shape)))
    error = sigmoidfunction(theta, X) - y   # (5000,1)
    for i in range(X.shape[1]):
        error_multiply = error.T * X[:, i]  # (1,5000)*(5000,1)
        partial_part = np.sum(error_multiply) / X.shape[0] # np.sum() here is used to transfer to 1 dimension
        if i == 0:
            partial[0, i] = partial_part
        else:
            reg_part = (learning_rate / X.shape[0]) * theta[0, i]
            partial[0, i] = partial_part + reg_part
    # print(partial.flatten().A[0])
    return partial.flatten().A[0]

# 1.4 one vs all clasiification
def onevsall(X, y, num_labels, learning_rate):
    X_new = np.c_[np.zeros((X.shape[0], 1)), X]
    theta_new = np.matrix(np.zeros((num_labels, X_new.shape[1])))
    # for each number, compute its related parameters θ(1 row)
    for i in range(1, num_labels+1):
        theta_initial = np.matrix(np.zeros(X_new.shape[1]))
        y_i = np.matrix(np.zeros((y.shape[0], 1)))
        for j in range(y.shape[0]):
            if y[j, 0] == i:
                y_i[j, 0] = 1
            else:
                y_i[j, 0] = 0
        # y_i:(5000,1); X_new:(5000, 401), theta_initial:(1,401)
        result = opt.minimize(fun=vector_costfunction, x0=theta_initial, args=(X_new, y_i, learning_rate), method='CG', jac=vector_gradientdescent)
        theta_new[i-1, :] = result.x
    return theta_new

# 1.5 one vs all prediction
def predictionfunction(theta, X):
    X_test = np.c_[np.zeros((X.shape[0], 1)), X]
    # compute the possibility for each item
    possibility = sigmoidfunction(theta, X_test) # (5000,401)*(401,10)=(5000,10)
    prediction_compute = np.argmax(possibility, axis=1) + 1 # (5000,)
    return prediction_compute

if __name__ == '__main__':
    # number of labels
    num_labels = 10 # label: 1-10
    # read dataset form file
    X, y = read_data()
    # visialize dataset
    visuial_data(X, y)
    # train the logistic regression
    learning_rate = 0.1
    theta = onevsall(X, y, num_labels, learning_rate)
    print('θ_optimum = ', theta)
    # test for prediction
    prediction = predictionfunction(theta, X)
    # print(prediction)
    # compute the accuracy
    correct = np.matrix(np.zeros((y.shape[0], 1)))
    for i in range(y.shape[0]):
        if prediction[i] == y[i, 0]:
            correct[i, 0] = 1
        else:
            correct[i, 0] = 0
    accuracy = (np.sum(correct) / correct.shape[0]) * 100
    print('accuracy is {0}%'.format(accuracy))
