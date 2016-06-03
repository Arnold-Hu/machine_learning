# coding=utf-8

import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
# 预定义 for each x, xi = 1


def j_theta(X, Y, x, y):
    print "y size ", y.size
    result = copy.deepcopy(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            s = 0
            for p in range(y.size):
                s += (h(np.array([X[i][j], Y[i][j]]), x[:, p]) - y[:, p])[0]**2
            result[i][j] = s / (2 * y.size)
    print "result", result
    return result


def partial_j(theta, x, y, index):
    print "theta", theta
    tmp = copy.copy(theta)

    tmp = h(tmp, x) - y
    result = tmp * x[index]
    print "partial j result ", result
    return result


def batch_GD_step(theta, alpha, x, y):
    print "into theta", theta
    new_theta = copy.deepcopy(theta)
    # for index, ele in enumerate(theta):
    #     theta[index] = ele - alpha*partial_j(theta, x, y, index)
    for i in range(theta.size):
        s = 0
        for j in range(y.size):
            s += partial_j(theta, x[:, j], y[:, j], i)[0]
            # tmp = (h(theta, x_y[0]) - x_y[1]) * x_y[index]
            # s += tmp
        s = s / x.size
        new_theta[0, i] = theta[0, i] - alpha[i] * s
        print " batch step length", s, "index", i, "new theta", new_theta
    return new_theta


def random_GD_step(theta, alpha, x, y, index):
    new_theta = copy.deepcopy(theta)
    for i in range(theta.size):
        tmp = partial_j(theta, x[:, index], y[:, index], i)[0]
        print "random step length", tmp, "index", i, "sample", index
        new_theta[0, i] = theta[0, i] - alpha[i] * tmp
    print "theta after batch", new_theta
    return new_theta


def h(theta, x):
    return np.dot(theta, np.transpose(x))


# def stop(theta, new_theta):
#     if sum(abs(i[0] - i[1]) for i in zip(theta, new_theta)):
#         return True
#     return False

# def produce_random_point(limit_size=100):
#     x = random.random()*10
#     # y = math.sin(x/10)*100 + random.random()*200
#     y = ture_h(x) + (random.random() - 0.5)*5
#     return [x,y]


def produce_random_x(length):
    return np.random.rand(1, length)


def pruduce_y_with_noise(x, noise, h):
    y = h(x) + noise
    return y


def ture_h(x):
    return 2 * x + 10


def h_plot_2d(theta, x):
    return theta[0, 0] + theta[0, 1] * x


def random_color(k):
    # generate a random color code like '#225c6f'
    color_rgb_element = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    color = []
    for i in range(k):
        temp = '#'
        for j in range(6):
            temp = temp + color_rgb_element[random.randint(0,15)]
        color.append(temp)
    return color

if __name__ == "__main__":


    x_line = np.linspace(-1, 1)
    theta_batch = np.zeros([1,2])
    theta_random = np.zeros([1,2])
    dots = []

    fig = plt.figure(figsize=(16,12), dpi=72, facecolor="white")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    # for i in range(20):
    #     dots.append(produce_random_point())

    length = 20
    x_dots = produce_random_x(length)

    noise = np.random.rand(1, length)
    y_dots = pruduce_y_with_noise(x_dots, noise, ture_h)

    b_x_dots = (x_dots - np.mean(x_dots)) / (np.max(x_dots) - np.min(x_dots))
    b_y_dots = (y_dots - np.mean(y_dots)) / (np.max(y_dots) - np.min(y_dots))

    x_dots_ex = np.concatenate((np.ones([1,length]),b_x_dots))

    ax1.scatter(b_x_dots, b_y_dots, marker='x')
    ax2.scatter(b_x_dots, b_y_dots, marker='x')

    x1 = np.arange(-0.3, 0.3, 0.03)
    y1 = np.arange(-0.3, 0.6, 0.03)
    X, Y = np.meshgrid(x1, y1)
    Z = j_theta(X, Y, x_dots_ex, b_y_dots)


    # fig2= plt.figure(figsize=(16,12), dpi=72, facecolor="white")
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
    CS = ax3.contour(X, Y, Z)
    ax3.clabel(CS, inline=1, fontsize=10)

    alpha_batch = [0.5, 0.5]
    alpha_random = [0.5, 0.5]
    # theta_batch_sqs = [[],[]]
    # theta_random_sqs = [[], []]
    for i in range(20):
        theta_batch = batch_GD_step(theta_batch, alpha_batch, x_dots_ex, b_y_dots)
        # theta_batch_sqs[0].append(theta_batch[0])
        # theta_batch_sqs[1].append(theta_batch[1])
        theta_random = random_GD_step(theta_random, alpha_random, x_dots_ex, b_y_dots, i%b_y_dots.size)
        # theta_random_sqs[0].append(theta_random[0])
        # theta_random_sqs[1].append(theta_random[1])
        print "batch theta", theta_batch, "random theta", theta_random
        line, = ax1.plot(x_line, h_plot_2d(theta_batch, x_line), '--', linewidth=2+0.1*i, label=str(i+1) + "th line")
        line, = ax2.plot(x_line, h_plot_2d(theta_random, x_line), '--', linewidth=2+0.1*i, label=str(i+1) + "th line")
    # ax3.plot(theta_batch_sqs[0], theta_batch_sqs[1])
    # for i,item in enumerate(zip(theta_batch_sqs[0], theta_batch_sqs[1])):
    #     ax3.scatter(item[0], item[1], marker='x', s=10+5*i)
    # ax3.plot(theta_random_sqs[0], theta_random_sqs[1])
    # for i,item in enumerate(zip(theta_random_sqs[0], theta_random_sqs[1])):
    #     ax3.scatter(item[0], item[1], marker='^', s=10+5*i)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.show()

