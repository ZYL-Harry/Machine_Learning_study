import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

'''
1. Logistic Regression
'''
# 1.1 Visualizing the data
# 1.1.1 get the data
path = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_2/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admission'])
# 1.1.2 classify the data
positive = data[data.Admission.isin([1])]
negative = data[data.Admission.isin([0])]
# 1.1.3 plot the data
plt.figure()
pos = plt.scatter(x=positive['Exam 1'], y=positive['Exam 2'], color='k', marker='+')
neg = plt.scatter(x=negative['Exam 1'], y=negative['Exam 2'], color='y', marker='o')
plt.legend([pos, neg], ['Admitted', 'Unadmitted'])
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.show()

# 1.2 Implementation
# 1.2.1 Warmup exercise: sigmoid function
def Sigmoid_Function(z):
    g = 1 / (1 + np.exp(-z))
    return g

# 1.2.2  Cost function
def Cost_Function(theta, x, y):
    first = - y.T * np.log(Sigmoid_Function(theta * x.T)).T
    # print(first)
    second = - (1 - y).T * np.log(1 - Sigmoid_Function(theta * x.T)).T
    # print(second)
    J = (1 / len(x)) * np.sum(first + second)
    return J

#1.2.3_1 Gradient descent
def Gradient_Descent(alpha, x, y, theta):
    temp = np.matrix(np.zeros(theta.shape))
    error = Sigmoid_Function(theta * x.T) - y.T
    for i in range(temp.shape[1]):
        partial_part = error * x[:, i]
        temp[0, i] =theta[0, i] - alpha * (1 / len(x)) * np.sum(partial_part)
    # partial_part = (Sigmoid_Function(theta * x.T) - y.T) * x
    # temp = theta - alpha * (1 / len(x)) * np.sum(partial_part)
    theta_new = temp
    return theta_new
# 1.2.3_2 Gradient descent used in established function(only for the partial part)
def Gradient_Descent2(theta, x, y):
    temp = np.matrix(np.zeros(theta.shape))
    error = Sigmoid_Function(theta * x.T) - y.T
    for i in range(temp.shape[1]):
        partial_part = error * x[:, i]
        temp[0, i] = (1 / len(x)) * np.sum(partial_part)
    theta_partial = temp
    return theta_partial

if __name__ == '__main__':
    data.insert(0, 'Ones', 1)
    x = data.loc[:, ['Ones', 'Exam 1', 'Exam 2']]
    x = np.matrix(x.values)
    print('the shape of x1 is ', x.shape)
    y = data.loc[:, ['Admission']]
    y = np.matrix(y.values)
    print('the shape of y1 is ', y.shape)
    theta = np.matrix(np.zeros(3))
    print('the shape of θ_initial is ', theta.shape)
    J0 = Cost_Function(theta, x, y)
    print('J_initial is ', J0)

    # 1.2.3(1) do gradient descent by ourselves---can't work out correctly
    alpha = 0.01
    iterations = 1500
    j = np.matrix(np.zeros((iterations, 1)))
    theta1 = np.matrix(np.zeros(theta.shape))
    for i in range(iterations):
        theta1 = Gradient_Descent(alpha, x, y, theta1)
    print('θ = ', theta1)

    # solutions
    information1 = np.matrix(np.array([1, 45, 85]))
    prediction1 = Sigmoid_Function(theta1 * information1.T)
    print('prediction to [45, 85] is ', prediction1)

    # plot the regression figure
    xx = np.linspace(30, 100, 100)
    yy = (theta1[0, 0] + theta1[0, 1] * xx) / (-theta1[0, 2])
    plt.figure()
    pos = plt.scatter(x=positive['Exam 1'], y=positive['Exam 2'], color='k', marker='+')
    neg = plt.scatter(x=negative['Exam 1'], y=negative['Exam 2'], color='y', marker='o')
    plt.legend([pos, neg], ['Admitted', 'Unadmitted'])
    plt.plot(xx, yy, color='b')
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')
    plt.show()

    # 1.2.3(2) do gradient descent with established function
    result = opt.fmin_tnc(func=Cost_Function, x0=theta, fprime=Gradient_Descent2, args=(x, y))
    theta2 = result[0]
    print('θ_fmin_tnc = ', theta2)

    # solutions
    information2 = np.matrix(np.array([1, 45, 85]))
    prediction2 = Sigmoid_Function(theta2 * information2.T)
    print('new prediction to [45, 85] is ', prediction2)

    # plot the regression figure
    xxx = np.linspace(30, 100, 100)
    yyy = (theta2[0] + theta2[1] * xxx) / (-theta2[2])
    plt.figure()
    pos = plt.scatter(x=positive['Exam 1'], y=positive['Exam 2'], color='k', marker='+')
    neg = plt.scatter(x=negative['Exam 1'], y=negative['Exam 2'], color='y', marker='o')
    plt.legend([pos, neg], ['Admitted', 'Unadmitted'])
    plt.plot(xxx, yyy, color='b')
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')
    plt.show()

    # check the accuracy
    predictions = Sigmoid_Function(theta2 * x.T)
    # compute the predictions
    predictions_classification = np.matrix(np.zeros(predictions.shape))
    for i in range(predictions.shape[1]):
        if predictions[0, i] >= 0.5:
            predictions_classification[0, i] = 1
        else:
            predictions_classification[0, i] = 0
    # judge which prediction is correct
    correct = np.matrix(np.zeros(predictions_classification.shape))
    j = 0
    print(predictions_classification.shape)
    print(y.shape)
    for i in range(predictions.shape[1]):
        if ((predictions_classification[0, i] == 1 and y[i, 0] == 1) or (predictions_classification[0, i] == 0 and y[i, 0] == 0)):
            correct[0, j] = 1
        else:
            correct[0, j] = 0
        j = j + 1
    # predictions_classification = [1 if x >= 0.5 else 0 for x in predictions[0, :]]
    # print(predictions_classification)
    # correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions_classification, y)]
    # accuracy = (sum(map(int, correct)) % len(correct))
    # print(correct.sum(axis=1))
    accuracy = (np.sum(correct, axis=1) / correct.shape[1]) * 100
    print('accuracy = {0}%'.format(accuracy[0, 0]))

'''
2.  Regularized logistic regression
'''
# 2.1 Visualizing the data
path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_2/ex2data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Microchip Test 1', 'Microchip Test 2', 'Decision'])
accepted = data2[data2['Decision'].isin([1])]
rejected = data2[data2['Decision'].isin([0])]
plt.figure()
accepted_plot = plt.scatter(x=accepted['Microchip Test 1'], y=accepted['Microchip Test 2'], color='k', marker='+')
rejected_plot = plt.scatter(x=rejected['Microchip Test 1'], y=rejected['Microchip Test 2'], color='y', marker='o')
plt.legend([accepted_plot, rejected_plot], ['Accepted', 'Rejected'])
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

# 2.2 Feature mapping
degree = 6
features = np.ones((data2['Microchip Test 1'].shape))
# print(np.power(data2['Microchip Test 1'], 2).shape)
# print(data2['Microchip Test 2'].shape)
# features = np.r_[np.matrix(np.ones((1, data2.shape[0]))), np.matrix(data2['Microchip Test 1']), np.matrix(data2['Microchip Test 2'])]
# print(features.shape)
for i in range(0, degree + 1):
    for j in range(0, degree + 1 - i):
        # 0:0-6, 1:0-5, 2:0-4, 3:0-3, 4:0-2, 5:0-1, 6:0
        features = np.c_[features, (np.power(data2['Microchip Test 1'], i) * np.power(data2['Microchip Test 2'], j))]
features = np.delete(features, 0, 1)
print(features.shape)

# 2.3 Cost function and gradient
def Cost_Function_regularized(theta, x, y, lamda):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = - y.T * np.log(Sigmoid_Function(theta * x.T)).T
    second = - (1 - y).T * np.log(1 - Sigmoid_Function(theta * x.T)).T
    J = np.sum(first + second) / len(x) + (lamda / (2 * len(x))) * np.sum(np.power(theta, 2))
    return J

def Gradient_Descent_regularized(theta, x, y, lamda):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    temp = np.matrix(np.zeros((theta.shape)))
    error = Sigmoid_Function(theta * x.T) - y.T
    for i in range(theta.shape[1]):
        if i == 0:
            temp[0, i] = (error * x[:, i]) / len(x)
        else:
            temp[0, i] = (error * x[:, i]) / len(x) + (lamda / len(x)) * theta[0, i]
        theta_partial = temp
    return theta_partial

def mapFeature(x1, x2, degree):
    z = np.matrix(np.ones(28))
    c = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            # print(j)
            # print('z = ', np.power(x1, i) * np.power(x2, j))
            z[0, c] = np.power(x1, i) * np.power(x2, j)
            c = c + 1
    return z

if __name__ == '__main__':
    # organize the data set
    data2.insert(0, 'Ones', 1)
    x2 = data2.loc[:, ['Microchip Test 1', 'Microchip Test 2']]
    x2 = np.matrix(x2.values)
    y2 = data2.loc[:, ['Decision']]
    y2 = np.matrix(y2.values)
    # initialize θ
    theta_2 = np.matrix(np.zeros(features.shape[1]))
    # initialize λ
    lamda = 1
    # compute the initial cost function
    J = Cost_Function_regularized(theta_2, features, y2, lamda)
    print('J2_initial is ', J)
    # use gradient descent to find θ
    result_2 = opt.fmin_tnc(func=Cost_Function_regularized, x0=theta_2, fprime=Gradient_Descent_regularized, args=(features, y2, lamda))
    theta_2_get = result_2[0]
    print('θ_regularized = ', theta_2_get)

    # Plotting the decision boundary
    feature_x1 = np.linspace(-1, 1.5, 50)
    feature_x2 = np.linspace(-1, 1.5, 50)
    feature_z = np.matrix(np.zeros((len(feature_x1), len(feature_x2))))
    # print('size = ', np.matrix(theta_2_get).shape)
    for i in range(len(feature_x1)):
        for j in range(len(feature_x2)):
            # print(mapFeature(feature_x1[i], feature_x2[j], degree).shape)
            feature_z[i, j] = np.matrix(theta_2_get) * mapFeature(feature_x1[i], feature_x2[j], degree).T
    plt.figure()
    accepted_plot = plt.scatter(x=accepted['Microchip Test 1'], y=accepted['Microchip Test 2'], color='k', marker='+')
    rejected_plot = plt.scatter(x=rejected['Microchip Test 1'], y=rejected['Microchip Test 2'], color='y', marker='o')
    plt.legend([accepted_plot, rejected_plot], ['Accepted', 'Rejected'])
    plt.contour(feature_x1, feature_x2, feature_z, [0])
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()

