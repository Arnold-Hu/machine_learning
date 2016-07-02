# coding=utf-8

import gredient_descent_np as gd
import numpy as np
import matplotlib.pyplot as plt
# import copy

np.random.seed(10)
def h(theta, x):
    # tmp = np.dot(theta, np.transpose(x))
    # return 1 / (1 + np.exp(-tmp))
    tmp = 1+np.exp(-np.dot(theta, np.transpose(x)))
    # print "down ", tmp
    tmp = 1.0 / tmp
    # print "hx ", tmp
    return tmp

def batch_GD_step(theta, alpha, x, y):
    new_theta = np.ones([1, theta.size])
    # for index, ele in enumerate(theta):
    #     theta[index] = ele - alpha*partial_j(theta, x, y, index)
    for i in range(theta.size):
        s = 0
        for j in range(y.size):
            tmp = partial_j(theta, x[:, j], y[:, j], i)
            s += tmp
        s = s / x.size
        new_theta[0, i] = theta[0, i] + alpha[i] * s
    # print "new theta ", new_theta
    return new_theta

# def cost_func()

def partial_j(theta, x, y, index):
    # tmp = copy.copy(theta)

    # tmp = h(tmp, x) - y
    # result = tmp * x[index]
    # return result
    tmp = (y - h(theta, x))*x[index]
    # print "partial ", tmp
    return tmp

def produce_around_dots(dot, num):
    x_err = np.random.rand(1, num) + dot[0]
    y_err = np.random.rand(1, num) + dot[1]
    return x_err, y_err

def reg(x):
    return (x - np.mean(x)) / (np.max(x) - np.min(x))

def plot_2d(theta, x):
    # print "plot theta ", theta
    return (-theta[0, 0] - theta[0, 1] * x) / theta[0, 2]


if __name__ == "__main__":
    turns = 100
    center_dot1 = (0.5,0.5)
    center_dot2 = (0.3, 0.3)
    theta_batch = np.array([[1, 1, 1]])
    theta_batch_sqs = theta_batch
    alpha_batch = [100, 100, 100]
    length = 10
    dots1_x, dots1_x2 = produce_around_dots(center_dot1, length)
    dots2_x, dots2_x2 = produce_around_dots(center_dot2, length)
    x = np.hstack([dots1_x, dots2_x])
    x2 = np.hstack([dots1_x2, dots2_x2])
    x = (x - np.mean(x)) / (np.max(x) - np.min(x))
    x2 = (x2 - np.mean(x2)) / (np.max(x2) - np.min(x2))


    fig = plt.figure(figsize=(15,10), dpi=72, facecolor="white")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    ax1.scatter(x[0][:length], x2[0][:length], color='blue')
    ax1.scatter(x[0][length:], x2[0][length:], color='red')
    ax2.scatter(x[0][:length], x2[0][:length], color='blue')
    ax2.scatter(x[0][length:], x2[0][length:], color='red')
    x_line = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5)
    y_line = np.linspace(np.min(x2) - 0.5, np.max(x2) + 0.5)
    x_ex = np.vstack([np.ones([1, x.size]), x , x2])
    y = np.hstack((np.ones([1, length]), np.zeros([1, length])))
    print x_ex
    print y
    line, = ax1.plot(x_line, plot_2d(theta_batch, x_line), '--', linewidth=0.5, label="1", color="red")

    error = []

    error.append(np.var(h(theta_batch, np.transpose(x_ex)) - y))
    for i in range(turns):
        tmp = batch_GD_step(theta_batch, alpha_batch, x_ex, y)
        theta_batch_sqs = np.concatenate((theta_batch_sqs, tmp))
        line, = ax1.plot(x_line, plot_2d(tmp, x_line), '--', linewidth=0.5, label=str(i+2))
        theta_batch = tmp
        error.append(np.var(h(theta_batch, np.transpose(x_ex)) - y))
        print h(theta_batch, np.transpose(x_ex))
    print error
    ax3.plot(np.arange(len(error)), error)

    # ax1.legend(loc='upper left')

    plt.show()