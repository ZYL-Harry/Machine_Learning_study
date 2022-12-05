import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report

'''
2. Neural Networks by backpropagation
    architecture: 3 layers: 5000 input units, 25 hidden units, 10 output units
'''

# 2.1 read the dataset
def read_data():
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_4/ex4/ex4data1.mat'
    data1 = loadmat(path1)
    X = data1['X']
    y = data1['y']
    return X, y

# 2.2 visiualize the dataset
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

# 2.3 randomly initialize weights
def initialize_weights(input_units_size, hidden_units_size, output_units_size, epsilon):
    theta1_initial = np.matrix(np.random.random((hidden_units_size, (input_units_size + 1))) * 2 * epsilon - epsilon)    # (25,401)
    theta2_initial = np.matrix(np.random.random((output_units_size, (hidden_units_size + 1))) * 2 * epsilon - epsilon)    # (10,26)
    return theta1_initial, theta2_initial

# 2.4 implement forward propagation to get h(x) for any x
# 2.4.1 sigmoid function
def sigmoidfunction(theta, X):
    theta = np.matrix(theta)
    X = np.matrix(X)
    f = 1 / (1 + np.exp(- X * theta.T)) # (5000,401)*(401,25)=(5000,25), (5000,26)*(26,10)=(5000,10)
    return f
# 2.4.2 feedforward propagation
def feedforward_propagation(theta1, theta2, X):
    X_new = np.c_[np.matrix(np.ones((X.shape[0], 1))), X]
    a1 = X_new  # (5000,401)
    z2 = a1 * theta1.T  # (5000,401)*(401,25)=(5000,25)
    a2 = np.c_[np.matrix(np.ones((X.shape[0], 1))), sigmoidfunction(theta1, a1)]    # (5000,26)
    z3 = a2 * theta2.T  # (5000,26)*(26,10)=(5000,10)
    a3 = sigmoidfunction(theta2, a2)    # (5000,10)
    prediction_compute = np.argmax(a3, axis=1) + 1
    return a1, z2, a2, z3, a3, prediction_compute

# 2.5 compute cost function
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

# 2.6 implement backpropagation to compute partial derivatives
# 2.6.1 sigmoid gradient
def sigmoid_gradient(z):
    f = 1 / (1 + np.exp(- z))
    f_derivative = np.multiply(f, (1 - f))
    return f_derivative
# 2.6.2 obtain the gradients for the neural network cost function
'''
def backpropagation(thetas, input_units_size, hidden_units_size, output_units_size, y, a1, z2, a2, a3, learning_rate):
    theta1 = np.matrix(np.reshape(thetas[:(hidden_units_size * (input_units_size + 1))], (hidden_units_size, (input_units_size + 1))))
    theta2 = np.matrix(np.reshape(thetas[(hidden_units_size * (input_units_size + 1)):], (output_units_size, (hidden_units_size + 1))))
    a3 = np.matrix(a3)
    y = np.matrix(y)
    delta1 = np.matrix(np.zeros(theta1.shape))
    delta2 = np.matrix(np.zeros(theta2.shape))
    for t in range(y.shape[0]):
        delta3_s = a3[t, :] - y[t, :] # (1,10)
        a1_t = a1[t, :] # (1,401)
        z2_t = np.c_[np.matrix(np.zeros((1, 1))), z2[t, :]]    # (1,26)
        a2_t = a2[t, :] # (1,26)
        delta2_s = np.multiply((theta2.T * delta3_s.T).T, sigmoid_gradient(z2_t)) # (1,26)
        delta2 = delta2 + delta3_s.T * a2_t   # (10,1)*(1,26)=(10,26)
        delta1 = delta1 + delta2_s[1, 1:].T * a1_t   # (25,1)*(1,401)=(25,401)
    D1 = delta1 / y.shape[0]    # (25,401)
    D2 = delta2 / y.shape[0]    # (10,26)
    # regularized neural networks
    D1[:, 1:] = D1[:, 1:] + ((learning_rate) / (y.shape[0])) * np.power(theta1[:, 1:], 2)
    D2[:, 1:] = D2[:, 1:] + ((learning_rate) / (y.shape[0])) * np.power(theta2[:, 1:], 2)
    return D1, D2
'''
# backpropagation for training the neural networks——a comprehensive function
# in the backpropagation, this function does all the jobs including computing the costfunction and the gradients
def backpropagation_minimize(thetas, input_units_size, hidden_units_size, output_units_size, X, y, learning_rate):
    y = np.matrix(y)
    # separate the thetas into theta1 and theta2
    theta1 = np.matrix(np.reshape(thetas[:(hidden_units_size * (input_units_size + 1))], (hidden_units_size, (input_units_size + 1))))
    theta2 = np.matrix(np.reshape(thetas[(hidden_units_size * (input_units_size + 1)):], (output_units_size, (hidden_units_size + 1))))
    # implement the forward propagation
    a1, z2, a2, z3, a3, prediction_compute = feedforward_propagation(theta1, theta2, X)
    # compute the costfunction
    J = costfunction(theta1, theta2, a3, y, learning_rate)
    # initialize the delta1 and delta2
    delta1 = np.matrix(np.zeros(theta1.shape))
    delta2 = np.matrix(np.zeros(theta2.shape))
    for t in range(y.shape[0]):
        delta3_s = a3[t, :] - y[t, :]  # (1,10)
        a1_t = a1[t, :]  # (1,401)
        z2_t = np.c_[np.matrix(np.zeros((1, 1))), z2[t, :]]  # (1,26)
        a2_t = a2[t, :]  # (1,26)
        delta2_s = np.multiply((theta2.T * delta3_s.T).T, sigmoid_gradient(z2_t))  # (1,26)
        delta2 = delta2 + delta3_s.T * a2_t  # (10,1)*(1,26)=(10,26)
        delta1 = delta1 + delta2_s[0, 1:].T * a1_t  # (25,1)*(1,401)=(25,401)
    D1 = delta1 / y.shape[0]  # (25,401)
    D2 = delta2 / y.shape[0]  # (10,26)
    # regularized neural networks
    D1[:, 1:] = D1[:, 1:] + ((learning_rate) / (y.shape[0])) * np.power(theta1[:, 1:], 2)
    D2[:, 1:] = D2[:, 1:] + ((learning_rate) / (y.shape[0])) * np.power(theta2[:, 1:], 2)
    D_matrix = np.r_[np.reshape(theta1_initial, ((theta1_initial.shape[0] * theta1_initial.shape[1]), 1)), np.reshape(theta2_initial, ((theta2_initial.shape[0] *theta2_initial.shape[1]), 1))]
    # print(D_matrix.shape)
    D = np.ravel(D_matrix)
    return J, D

# 2.7 gradient checking
'''
# 2.7.1 compute the numerical gradients
def computenumericalgradients(thetas):
    epsilon_check = 0.0001
    thetas_check = np.matrix(np.zeros(thetas.shape))
    numgradients = np.matrix(np.zeros(thetas.shape))
    for i in range(thetas.shape[0]):
        thetas_check[i, 1] = epsilon_check
        # J_plus and J_minus haven't been completed due to the different parameters with the costfunction
        J_plus = costfunction((thetas + thetas_check))
        J_minus = costfunction((thetas - thetas_check))
        numgradients[i, 1] = (J_plus - J_minus) / (2 * epsilon_check)
        thetas_check[i, 1] = 0
    return numgradients
# 2.7.2 compute the difference
def checkgradients(theta1, theta2, D1, D2):
    thetas = np.c_[theta1[:], theta2[:]]
    gradients_backpropagation = np.c_[D1[:], D2[:]]
    numgradients = computenumericalgradients(thetas)
    difference = gradients_backpropagation - numgradients
    return difference
'''

# 2.8 gradient descent to find the optimum parameter θ
def training_function(thetas, input_units_size, hidden_units_size, output_units_size, X, y, learning_rate):
    # result = opt.minimize(fun=backpropagation_minimize, x0=(thetas), args=(input_units_size, hidden_units_size, output_units_size, X, y, learning_rate), method='CG', jac=True, options={'maxiter': 250})
    result = opt.minimize(fun=backpropagation_minimize, x0=(thetas),args=(input_units_size, hidden_units_size, output_units_size, X, y, learning_rate), method='CG', jac=True, options={'maxiter': 1000})
    print(result)
    theta = result.x
    theta1 = np.matrix(np.reshape(theta[:(hidden_units_size * (input_units_size + 1))], (hidden_units_size, (input_units_size + 1))))
    theta2 = np.matrix(np.reshape(theta[(hidden_units_size * (input_units_size + 1)):], (output_units_size, (hidden_units_size + 1))))
    return theta1, theta2

# 2.9 test for prediction
def prediction_function(theta1, theta2, X):
    a1, z2, a2, z3, a3, prediction_test = feedforward_propagation(theta1, theta2, X)
    return prediction_test

if __name__ == '__main__':
    # number of labels
    num_labels = 10
    # read the dataset
    X, y = read_data()
    # visiualize the dataset
    visiualize_data(X)
    # randomly initialize weights
    input_units_size = 400
    hidden_units_size = 25
    output_units_size = 10
    epsilon = 0.12
    theta1_initial, theta2_initial = initialize_weights(input_units_size, hidden_units_size, output_units_size, epsilon)
    # implement forward propagation
    a1_initial, z2_initial, a2_initial, z3_initial, a3_initial, prediction_initial = feedforward_propagation(theta1_initial, theta2_initial, X)
    # compute the cost function
    learning_rate = 1
    J_initial = costfunction(theta1_initial, theta2_initial, a3_initial, y, learning_rate)
    print('J_initial = ', J_initial)
    thetas = np.r_[np.reshape(theta1_initial, ((theta1_initial.shape[0] * theta1_initial.shape[1]), 1)), np.reshape(theta2_initial, ((theta2_initial.shape[0] *theta2_initial.shape[1]), 1))]
    # print('1: ', thetas.shape)
    '''
    # implement back-propagation
    D1, D2 = backpropagation(thetas, input_units_size, hidden_units_size, output_units_size, y, a1_initial, z2_initial, a2_initial, a3_initial, learning_rate)
    # gradient checking
    # checkgradients(theta1_initial, theta2_initial, D1, D2)
    # all the codes above in the main function are mainly used to check gradients and produce the initial thetas, the codes below are used to really train the neural networks
    '''
    # thetas = np.c_[D1[:], D2[:]]
    # gradient descent to find the optimum parameter θ
    theta1, theta2 = training_function(thetas, input_units_size, hidden_units_size, output_units_size, X, y, learning_rate)
    print('θ_1 = ', theta1)
    print('θ_2 = ', theta2)
    # test for prediction
    prediction_test = prediction_function(theta1, theta2, X)
    # compute the accuracy
    print(classification_report(y, prediction_test))
    correct = np.matrix(np.zeros((y.shape[0], 1)))
    for i in range(y.shape[0]):
        if prediction_test[i] == y[i, 0]:
            correct[i, 0] = 1
        else:
            correct[i, 0] = 0
    accuracy = (np.sum(correct) / correct.shape[0]) * 100
    print('accuracy is {0}%'.format(accuracy))


