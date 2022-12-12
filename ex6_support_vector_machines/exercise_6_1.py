import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

'''
1. Support Vector Machines
'''

# 1.1 read the dataset
def read_data(path):
    data = loadmat(path)
    X = data['X']   # (51,2)
    y = data['y']   # (51,1)
    return X, y

# 1.2 visiualize the dataset
def visiualize_data(X, y):
    # print(np.argwhere(y == 1)[:, 0])
    positive = np.c_[X[np.argwhere(y == 1)[:, 0], :], y[np.argwhere(y == 1)[:, 0]]]
    negative = np.c_[X[np.argwhere(y == 0)[:, 0], :], y[np.argwhere(y == 0)[:, 0]]]
    plt.figure()
    plt.scatter(x=positive[:, 0], y=positive[:, 1], color='k', marker='+')
    plt.scatter(x=negative[:, 0], y=negative[:, 1], color='y', marker='o')
    plt.show()

# 1.3 take the SVM algorithm
def training_data(X1, y1, C, kernel, tol, max_iter, gamma):
    if kernel == 'linear':
        model = svm.SVC(C=C, kernel=kernel, tol=tol, max_iter=max_iter)
    elif kernel == 'rbf':
        model = svm.SVC(C=C, kernel=kernel, gamma=gamma, tol=tol, max_iter=max_iter)
    estimator = model.fit(X1, y1)
    accuracy = model.score(X1, y1)
    print('an instance of estimator is ', estimator)
    print('the mean accuracy is ', accuracy)
    return model

# 1.4 visiualize the dataset
# 1.4.1 visiualize the dataset---without kernel(linear kernel)
def find_decision_boundary_linear(model, X, y):
    # original examples
    positive = np.c_[X[np.argwhere(y == 1)[:, 0], :], y[np.argwhere(y == 1)[:, 0]]]
    negative = np.c_[X[np.argwhere(y == 0)[:, 0], :], y[np.argwhere(y == 0)[:, 0]]]
    # create the data for plotting
    w = np.matrix(model.coef_)
    b = np.matrix(model.intercept_)
    x_min = 0
    x_max = 4
    x_plot = np.matrix(np.linspace(start=x_min, stop=x_max, num=100)).T
    y_plot = - (w[0, 0] * x_plot + b) / w[0, 1]
    # plot
    plt.figure()
    plt.scatter(x=positive[:, 0], y=positive[:, 1], color='k', marker='+')
    plt.scatter(x=negative[:, 0], y=negative[:, 1], color='y', marker='o')
    plt.plot(x_plot, y_plot, color='b')
    plt.show()
    # x1_min = 0
    # x1_max = 4
    # x2_min = 2
    # x2_max = 4.5
    # x1 = np.linspace(start=x1_min, stop=x1_max, num=100)
    # x2 = np.linspace(start=x2_min, stop=x2_max, num=100)
    # x1_plot, x2_plot = np.meshgrid(x1, x2)
    # X_plot = np.array([x1_plot, x2_plot])
    # confidence_score = model.decision_function(X_plot)
    # print(confidence_score.shape)

# 1.4.2 viviualize the dataset---Gaussian kernel(rbf kernel)
def find_decision_boundary_gaussian(model, X, y):
    # original examples
    positive = np.c_[X[np.argwhere(y == 1)[:, 0], :], y[np.argwhere(y == 1)[:, 0]]]
    negative = np.c_[X[np.argwhere(y == 0)[:, 0], :], y[np.argwhere(y == 0)[:, 0]]]
    # create the data for plotting
    x1_plot = np.linspace(start=X[:, 0].min(), stop=X[:, 0].max(), num=100)
    x2_plot = np.linspace(start=X[:, 1].min(), stop=X[:, 1].max(), num=100)
    x1, x2 = np.meshgrid(x1_plot, x2_plot)
    x1 = np.matrix(x1).T
    x2 = np.matrix(x2).T
    input = np.c_[x1.reshape((x1.shape[0] * x1.shape[1]), 1), x2.reshape((x2.shape[0] * x2.shape[1]), 1)]
    values = np.matrix(model.predict(input)).T
    values_plot = values.reshape(x1.shape)
    # for i in range(x1.shape[0]):
    #     x_compute = np.c_[x1[:, i], x2[:, i]]
    #     values[:, i] = np.matrix(model.predict(x_compute)).T
    # plot
    plt.figure()
    plt.scatter(x=positive[:, 0], y=positive[:, 1], color='k', marker='+')
    plt.scatter(x=negative[:, 0], y=negative[:, 1], color='y', marker='o')
    plt.contour(x1, x2, values_plot, [0])
    plt.show()

# find the best parameters for svm
def find_best_params(Xval, yval):
    C_set = np.matrix([0.1, 0.3, 1, 3, 10, 30]).T
    sigma_set = np.matrix([0.1, 0.3, 1, 3, 10, 30]).T
    gamma_set = 1 / (2 * np.power(sigma_set, 2))
    best_score = 0
    best_C = 0
    best_sigma = 0
    for i in range(C_set.shape[0]):
        for j in range(sigma_set.shape[0]):
            model_temp = svm.SVC(C=C_set[i, 0], kernel='rbf', gamma=gamma_set[j, 0], tol=1e-3)
            estimator_temp = model_temp.fit(Xval, yval)
            accuracy_temp = model_temp.score(Xval, yval)
            if accuracy_temp > best_score:
                best_score = accuracy_temp
                best_C = C_set[i, 0]
                best_sigma = sigma_set[j, 0]
    return best_C, best_sigma, best_score

if __name__ == '__main__':
    '''1.1 SVM without kernel(linear kernel)'''
    # read the dataset
    path1 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_6/ex6/ex6data1.mat'
    X1, y1 = read_data(path1)
    # visiualize the datset
    visiualize_data(X1, y1)
    # take the SVM algorithm
    C1 = 100
    kernel1 = 'linear'
    tol1 = 1e-3
    max_iter1 = 100
    gamma1 = 0
    model1 = training_data(X1, y1, C1, kernel1, tol1, max_iter1, gamma1)
    # visiualize the classifier
    find_decision_boundary_linear(model1, X1, y1)
    '''1.2 SVM with Gaussian kernel'''
    path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_6/ex6/ex6data2.mat'
    X2, y2 = read_data(path2)
    # visiualize the dataset
    visiualize_data(X2, y2)
    # take the SVM algorithm
    C2 = 100
    kernel2 = 'rbf'
    tol2 = 1e-3
    max_iter2 = 1000
    sigma2 = 0.1 # the parameter in the gaussian kernel function
    gamma2 = 1 / (2 * np.power(sigma2, 2))   # the parameter used in the svm functions when using the gaussian kernel
    model2 = training_data(X2, y2, C2, kernel2, tol2, max_iter2, gamma2)
    # visiualize the classifier
    find_decision_boundary_gaussian(model2, X2, y2)
    '''1.3 SVM with Gaussian kernel'''
    # read the dataset
    path3 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_6/ex6/ex6data3.mat'
    data3 = loadmat(path3)
    X3 = data3['X']
    y3 = data3['y']
    Xval3 = data3['Xval']
    yval3 = data3['yval']
    # visiualize the dataset
    visiualize_data(X3, y3)
    # determine the parameters C and sigmma
    C3, sigma3, best_score = find_best_params(Xval3, yval3)
    print('C_best = ', C3)
    print('sigma_best = ', sigma3)
    # take the svm algorithm
    kernel3 = 'rbf'
    tol3 = 1e-3
    max_iter3 = 1000
    gamma3 = 1 / (2 * np.power(sigma3, 2))
    model3 = training_data(X3, y3, C3, kernel3, tol3, max_iter3, gamma3)
    # visiualize the calssifier
    find_decision_boundary_gaussian(model3, X3, y3)
