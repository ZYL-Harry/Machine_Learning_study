import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
1. Create a 5*5 identity matrix
'''
A = np.eye(5)
print(A)

'''
2. Linear regression with one variable
'''
# 2.1 Plotting the Data
path = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_1/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data)
plt.figure()
data.plot(kind='scatter', x='Population', y='Profit', color='red', marker='x', figsize=(10,10), xlabel='Population of City in 10,000s', ylabel='Profit in $10,000s')
plt.show()
# 2.2 Gradient Descent
def Cost_Function(x, y, theta):
    inner = np.power(((theta * x.T) - y.T), 2)
    return np.sum(inner) / (2 * len(x))

def Update_theta(x, y, theta, alpha):
    error = (theta * x.T) - y.T
    # print(error)
    error_sum0 = np.sum(error) / len(x)
    # print(error_sum0)
    error_sum1 = np.sum(error * x[:, 1]) / len(x)
    # print(error_sum1)
    # need another variable to replace θ assignment
    temp = np.matrix(np.zeros(theta.shape))
    temp[0, 0] = theta[0, 0] - alpha * error_sum0
    temp[0, 1] = theta[0, 1] - alpha * error_sum1
    theta = temp
    return theta

if __name__ == '__main__':
    # add an additional first column to X and set it to all ones
    data.insert(0, 'Ones', 1)
    print(data)
    # take the value of x[ones,population] and y[profit]
    x = data.loc[:, ['Ones', 'Population']]
    x = np.matrix(x.values)
    print(x)
    y = data.loc[:, ['Profit']]
    y = np.matrix(y.values)
    print(y)
    # initialize theta
    theta = np.matrix(np.array([0, 0]))
    print(theta)
    # compute the cost function
    J = Cost_Function(x, y, theta)
    print(J)

    # gradient descent
    alpha = 0.01
    iterations = 1500
    j = np.zeros((iterations, 1))
    theta_all = theta
    for i in range(iterations):
        # compute the cost function
        j[i] = Cost_Function(x, y, theta)
        # update theta
        theta = Update_theta(x, y, theta, alpha)
        theta_all = np.row_stack((theta_all, theta))
    theta_iterations = np.delete(theta_all, 0, 0)
    print(theta)

    # plot the regression figure
    xx = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0, 0] + theta[0, 1] * xx
    plt.figure()
    data.plot(kind='scatter', x='Population', y='Profit', color='red', marker='x', figsize=(10, 10),xlabel='Population of City in 10,000s', ylabel='Profit in $10,000s')
    plt.plot(xx, f, color='blue')
    plt.show()

    # visualize the cost function
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    theta0, theta1 = np.meshgrid(theta_iterations[:, 0].ravel().T, theta_iterations[:, 1].ravel().T)
    j_plot = np.zeros((iterations,iterations))
    for i in range(iterations):
        for k in range(iterations):
            theta_temp = [theta0[i, k], theta1[i, k]]
            j_plot[i, k] = Cost_Function(x, y, theta_temp)
    print(theta0.shape)
    print(theta1.shape)
    print(j_plot.shape)
    ax.plot_surface(theta0, theta1, j_plot, cmap='rainbow')
    ax.set_xlabel('θ0')
    ax.set_ylabel('θ1')
    ax.set_zlabel('J')
    plt.show()

    # plot the contour figure
    plt.figure()
    plt.contour(theta0, theta1, j_plot)
    plt.plot(theta[0, 0], theta[0, 1], color='red', marker='x')
    plt.xlabel('θ0')
    plt.ylabel('θ1')
    plt.show()

    # solutions
    information1 = np.matrix(np.array([1, 3.5]))
    prediction1 = theta * information1.T
    print('the prediction on profits in areas of 35,000 people is ', prediction1)
    information2 = np.matrix(np.array([1, 7]))
    prediction2 = theta * information2.T
    print('the prediction on profits in areas of 70,000 is ', prediction2)

'''
3. Linear regression with multiple variables
'''
# 3.1 gradient descent
# 3.1.1 plot the data set
path2 = 'D:/新建文件夹/机器学习/Machine_Learning_exercise/exercise_1/ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2)
plt.figure()
ax2 = plt.axes(projection='3d')
ax2.scatter(data2.loc[:, 'Size'], data2.loc[:, 'Bedrooms'], data2.loc[:, 'Price'], color='red', marker='x')
ax2.set_xlabel('the size of the house (in square feet)')
ax2.set_ylabel(' the number of bedrooms')
ax2.set_zlabel('the price of the house')
plt.show()

# 3.1.2 Feature Normalization
data2_normalized = (data2 - data2.mean()) / data2.std()
print(data2_normalized)

# 3.1.3  Gradient Descent
def Cost_Function2(x, y, theta):
    inner = np.power(((theta * x.T) - y.T), 2)
    return np.sum(inner) / (2 * len(x))

def Update_theta2(x, y, theta, alpha):
    error = (theta * x.T) - y.T
    # print(error)
    error_sum0 = np.sum(error) / len(x)
    # print(error_sum0)
    error_sum1 = np.sum(error * x[:, 1]) / len(x)
    # print(error_sum1)
    error_sum2 = np.sum(error * x[:, 2]) / len(x)
    # need another variable to replace θ assignment
    temp = np.matrix(np.zeros(theta.shape))
    temp[0, 0] = theta[0, 0] - alpha * error_sum0
    temp[0, 1] = theta[0, 1] - alpha * error_sum1
    temp[0, 2] = theta[0, 2] - alpha * error_sum2
    theta = temp
    return theta

if __name__ == '__main__':
    # add an additional first column to X and set it to all ones
    data2_normalized.insert(0, 'Ones', 1)
    print(data2_normalized)
    # take the value of x[ones,population] and y[profit]
    x2 = data2_normalized.loc[:, ['Ones', 'Size', 'Bedrooms']]
    x2 = np.matrix(x2.values)
    print(x2)
    y2 = data2_normalized.loc[:, ['Price']]
    y2 = np.matrix(y2.values)
    print(y2)
    # initialize theta
    theta2 = np.matrix(np.array([0, 0, 0]))
    print(theta2)
    # compute the cost function
    J2 = Cost_Function2(x2, y2, theta2)
    print(J2)

    # gradient descent
    alpha2 = 0.01
    iterations2 = 1500
    j2 = np.zeros((iterations2, 1))
    theta_all2 = theta2
    for i in range(iterations2):
        # compute the cost function
        j2[i] = Cost_Function2(x2, y2, theta2)
        # update theta
        theta2 = Update_theta2(x2, y2, theta2, alpha2)
        theta_all2 = np.row_stack((theta_all2, theta2))
    theta_iterations = np.delete(theta_all2, 0, 0)
    print(theta2)

    # solutions
    information3 = np.matrix(np.array([1, (1650-data2.loc[:, 'Size'].mean())/data2.loc[:, 'Size'].std(), (3-data2.loc[:, 'Bedrooms'].mean())/data2.loc[:, 'Bedrooms'].std()]))
    prediction3_normalized = theta2 * information3.T
    prediction3 = prediction3_normalized * data2.loc[:, 'Price'].std() +data2.loc[:, 'Price'].mean()
    print('a price prediction for a 1650-square-foot house with 3 bedrooms is ', prediction3)

    # plot cost function with iterations
    plt.figure()
    ii = range(iterations2)
    plt.plot(ii, j2, color='green')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost Function')
    plt.show()

# 3.2 normal equation
def Normal_Equation(X, y):
    theta = np.linalg.inv(X.T * X) * X.T * y
    return theta

if __name__ == '__main__':
    data2.insert(0, 'Ones', 1)
    x3 = data2.loc[:, ['Ones', 'Size', 'Bedrooms']]
    x3 = np.matrix(x3.values)
    print(x3)
    y3 = data2.loc[:, ['Price']]
    y3 = np.matrix(y3.values)
    print(y3)
    theta3 = Normal_Equation(x3, y3)
    print(theta3)

    # solutions
    information4 = np.matrix(np.array([1, 1650, 3]))
    prediction4 = theta3.T * information4.T
    print('a price prediction for a 1650-square-foot house with 3 bedrooms is ', prediction4)