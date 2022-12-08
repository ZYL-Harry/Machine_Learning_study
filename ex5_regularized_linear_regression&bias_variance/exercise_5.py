import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt

'''
1. Regularized linear regression
'''

# 1.1 read the dataset
def read_data():
    path = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_5/ex5/ex5data1.mat'
    data = loadmat(path)
    # training set
    X = data['X']
    y = data['y']
    # cross validation set
    Xval = data['Xval']
    yval = data['yval']
    # test set
    Xtest = data['Xtest']
    ytest = data['ytest']
    return X, y, Xval, yval, Xtest, ytest

# 1.2 visiualize the dataset
def visiualize_data(X, y, Xval, yval, Xtest, ytest):
    X_show = np.r_[X, Xval, Xtest]
    y_show = np.r_[y, yval, ytest]
    plt.figure()
    plt.scatter(x=X_show, y=y_show, color='r', marker='+')
    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.show()

# 1.3 regularized linear regression cost function
# 1.3.1 sigmoid function
def sigmoid_function(z):
    f = 1 / (1 + np.exp(-z))
    return z
# 1.3.2 cost function
def regularized_costfunction(theta, X, y, learning_rate):
    X = np.matrix(X)    # (12,1)
    X_new = np.c_[np.matrix(np.ones((X.shape[0], 1))), X]   # (12,2)
    y = np.matrix(y)    # (12,1)
    theta = np.matrix(theta)    # (1,2)
    error = sigmoid_function(X_new * theta.T) - y   # (12,2)*(2,1)=(12,1)
    first = (1 / (2 * y.shape[0])) * (np.sum(np.power(error, 2)))
    second = (learning_rate / (2 * y.shape[0])) * (np.sum(np.power(theta, 2)))
    J = first + second
    return J

# 1.4 regularized linear regression gradient
def regularized_gradient(theta, X, y, learning_rate):
    X = np.matrix(X)
    X_new = np.c_[np.matrix(np.ones((X.shape[0], 1))), X]
    y = np.matrix(y)
    theta = np.matrix(theta)
    partial_derivative = np.matrix(np.zeros(theta.shape))
    error = sigmoid_function(X_new * theta.T) - y # (12,1)
    for i in range(theta.shape[1]):
        error_multiply = error.T * X_new[:, i]    # (1,12)*(12,1)=(1,1)
        partial_part = (1 / y.shape[0]) * np.sum(error_multiply)    # (1,1)
        if i == 1:
            partial_derivative[0, i] = partial_part
        else:
            partial_derivative[0, i] = partial_part + (learning_rate / y.shape[0]) * theta[0, i]
    return partial_derivative.flatten().A[0]

# 1.5 fitting linear regression
def fitting(theta, X, y, learning_rate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    # learning_rate = 0   # set the regularization parameter to 0 temporarily
    # when apply opt.minimize() theta will be transfered into a (1,n) array
    result = opt.minimize(fun=regularized_costfunction, x0=theta, args=(X, y, learning_rate), jac=regularized_gradient, method='CG')
    theta_best = result.x
    return theta_best

# 1.6 plot the fitting linear regression
def plot_regression(theta_best, X, y, feature, degree, index):
    theta_best = np.matrix(theta_best)
    if index == 1:
        X_point = np.matrix(feature[:, 0]).flatten().A[0]
        X_plot = np.matrix(np.linspace(start=feature[:, 0].min(), stop=feature[:, 0].max(), num=100)).T  # In the polynomial regression situation, take the first column as the original feature
        X_poly = np.matrix(np.zeros((X_plot.shape[0], degree)))
        X_poly[:, 0] = X_plot
        for i in range(degree - 1):
            X_poly[:, (i + 1)] = np.matrix(np.power(X_plot, (i + 2)))   # (1,7)---(2,8)
        X_poly = feature_normalize(X_poly)
        X_new = np.c_[np.matrix(np.ones((X_plot.shape[0], 1))), X_poly]
        y_regression = sigmoid_function(X_new * theta_best.T)
    elif index == 0:
        X_point = X
        X_plot = np.matrix(np.linspace(start=X.min(), stop=X.max(), num=100)).T
        X_new = np.c_[np.matrix(np.ones((X_plot.shape[0], 1))), X_plot]   # (100,2)
        y_regression = sigmoid_function(X_new * theta_best.T) # (100,2)*(2,1)=(100,1)
    # print('X_point_shape: ', X_point.shape)
    plt.scatter(x=X_point, y=y, color='r', marker='+')
    plt.plot(X_plot, y_regression, color='b')
    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water flowing out of the bam(y)')
    plt.show()

# 2. learning curves
def learning_curves(X, y, Xval, yval, theta):
    # compute the error with 1-X.shape[0] training example(s)
    error_train = np.matrix(np.zeros((X.shape[0], 1)))
    error_val = np.matrix(np.zeros((X.shape[0], 1)))
    learning_rate = 0
    Xval_new = np.c_[np.matrix(np.ones((Xval.shape[0], 1))), Xval]
    for i in range(X.shape[0]):
        X_i = np.matrix(X[:i+1, :])   # X[:i,j] automatically becomes a row matrix
        # print(X_i.shape)
        X_i_new = np.c_[np.matrix(np.ones(((i + 1), 1))), X_i]
        y_i = np.matrix(y[:i+1, :])
        result_i = opt.minimize(fun=regularized_costfunction, x0=theta, args=(X_i, y_i, learning_rate), jac=regularized_gradient, method='CG')
        theta_i = np.matrix(result_i.x)
        # compute the error of training set
        y_train_i = sigmoid_function(X_i_new * theta_i.T)    # (i,2)*(2,1)=(i,1)
        error_train[i, 0] = (1 / (2 * X_i.shape[0])) * np.sum(np.power((y_train_i - y_i), 2))
        # compute the error of cross validation set
        y_val_i = sigmoid_function(Xval_new * theta_i.T)
        error_val[i, 0] = (1 / (2 * Xval.shape[0])) * np.sum(np.power((y_val_i - yval), 2))
    # plot the learning curves
    num_x = np.matrix(range(X.shape[0])).T + 1
    plt.figure()
    plt.plot(num_x, error_train, color='b', label='Train')
    plt.plot(num_x, error_val, color='g', label='Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

# 3. polynomial regression
# 3.1 feature normalization
def feature_normalize(feature):
    feature_normalized = np.matrix(np.zeros(feature.shape))
    for i in range(feature_normalized.shape[1]):
        feature_normalized[:, i] = (feature[:, i] - feature[:, i].mean()) / feature[:, i].std()
    # feature_normalized = (feature - feature.mean()) / feature.std()
    return feature_normalized
# 3.2 polynomial regression
def poly_regression(X, y, Xval, yval, Xtest, ytest, theta_poly, degree, learning_rate):
    # degree = 8
    feature = np.matrix(np.zeros((X.shape[0], degree)))
    feature[:, 0] = np.matrix(X[:, 0]).T # X is actually (12,1)
    feature_val = np.matrix(np.zeros((Xval.shape[0], degree)))
    feature_val[:, 0] = np.matrix(Xval[:, 0]).T
    feature_test = np.matrix(np.zeros((Xtest.shape[0], degree)))
    feature_test[:, 0] = np.matrix(Xtest[:, 0]).T
    for i in range(degree - 1):
        feature[:, (i + 1)] = np.matrix(np.power(np.matrix(X[:, 0]).T, (i + 2)))    # (1,7)---(2,8)
        feature_val[:, (i + 1)] = np.matrix(np.power(np.matrix(Xval[:, 0]).T, (i + 2)))
        feature_test[:, (i + 1)] = np.matrix(np.power(np.matrix(Xtest[:, 0]).T, (i + 2)))
    # normalize the features
    feature_normalized = feature_normalize(feature)
    feature_val_normalized = feature_normalize(feature_val)
    feature_test_normalized = feature_normalize(feature_test)
    # fitting
    theta_best_poly = fitting(theta_poly, feature_normalized, y, learning_rate)
    return feature, feature_normalized, feature_val_normalized, feature_test_normalized, theta_best_poly

# 3.2 select learning rate λ
def select_lambda(X, y, Xval, yval, Xtest, ytest, theta_poly, degree):
    lambdas = np.matrix([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T
    error_train = np.matrix(np.zeros((lambdas.shape[0], 1)))
    error_val = np.matrix(np.zeros((lambdas.shape[0], 1)))
    theta_poly_lambda = np.matrix(np.zeros((lambdas.shape[0], theta_poly.shape[1])))
    for i in range(lambdas.shape[0]):
        lambda_i = lambdas[i, 0]
        feature, feature_normalized, feature_val_normalized, feature_test_normalized, theta_best_poly = poly_regression(X, y , Xval, yval, Xtest, ytest, theta_poly, degree, lambda_i)
        theta_best_poly = np.matrix(theta_best_poly)    # (1,9)
        theta_poly_lambda[i, :] = theta_best_poly # (1,9)
        # compute the error of training set
        feature_normalized_new = np.c_[np.matrix(np.ones((feature_normalized.shape[0], 1))), feature_normalized]
        y_train_i = sigmoid_function(feature_normalized_new * theta_best_poly.T)
        error_train[i, 0] = (1 / (2 * y.shape[0])) * np.sum(np.power((y_train_i - y), 2))
        # compute the error of cross validation set
        feature_val_normalized_new = np.c_[np.matrix(np.ones((feature_val_normalized.shape[0], 1))), feature_val_normalized]
        y_val_i = sigmoid_function(feature_val_normalized_new * theta_best_poly.T)
        error_val[i, 0] = (1 / (2 * yval.shape[0])) * np.sum(np.power((y_val_i - yval), 2))
    plt.figure()
    plt.plot(lambdas, error_train, color='b', label='Train')
    plt.plot(lambdas, error_val, color='g', label='Cross Validation')
    plt.xlabel('λ')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    # select the best λ
    index = np.argmin(error_val)
    lambda_best = lambdas[index, 0]
    theta_best_lambda = theta_poly_lambda[index, :]
    return lambda_best, theta_best_lambda

if __name__ == '__main__':
    # read the dataset
    X, y, Xval, yval, Xtest, ytest = read_data()
    # visiualize the dataset
    visiualize_data(X, y, Xval, yval, Xtest, ytest)
    # compute the regularized linear regression cost function
    theta = np.matrix(np.ones((1, 2)))
    learning_rate = 1
    J = regularized_costfunction(theta, X, y, learning_rate)
    print('J_initial = ', J)
    # compute the regularized linear regression gradient
    gradient = regularized_gradient(theta, X, y, learning_rate)
    print('gradients_initial = ', gradient)
    # fitting linear regression
    theta_best = fitting(theta, X, y, learning_rate)
    print('θ_optimum = ', theta_best)
    # plot the fitting linear regression
    plot_regression(theta_best, X, y, 0, 0, 0)
    # use learning curves to diagnose the bias or variance problem
    learning_curves(X, y, Xval, yval, theta)
    # learning polynomial regression
    degree = 8
    learning_rate = 0
    theta_poly = np.matrix(np.ones((1, (degree + 1))))
    feature, feature_normalized, feature_val_normalized, feature_test_normalized, theta_best_poly = poly_regression(X, y, Xval, yval, Xtest, ytest, theta_poly, degree, learning_rate)
    print('θ_optimum_poly = ', theta_best_poly)
    plot_regression(theta_best_poly, feature_normalized, y, feature, degree, 1)
    learning_curves(feature_normalized, y, feature_val_normalized, yval, theta_poly)
    # select the learning rate λ
    lambda_best, theta_best_lambda = select_lambda(X, y, Xval, yval, Xtest, ytest, theta_poly, degree)
    print('λ_best = ', lambda_best)
    # estimate the performance with the best learning rate
    plot_regression(theta_best_lambda, feature_normalized, y, feature, degree, 1)
    # compute the error of test set
    feature_test_normalized_new = np.c_[np.matrix(np.ones((feature_test_normalized.shape[0], 1))), feature_test_normalized]
    y_test_i = sigmoid_function(feature_test_normalized_new * theta_best_lambda.T)
    error_test = (1 / (2 * ytest.shape[0])) * np.sum(np.power((y_test_i - ytest), 2))
    print('error_test = ', error_test)
