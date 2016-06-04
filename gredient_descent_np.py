# coding=utf-8

import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
# 预定义 for each x, xi = 1


def j_theta(X, Y, x, y):
    result = copy.deepcopy(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            s = 0
            for p in range(y.size):
                s += (h(np.array([X[i][j], Y[i][j]]), x[:, p]) - y[:, p])[0]**2
            result[i][j] = s / (2 * y.size)
    return result


def partial_j(theta, x, y, index):
    tmp = copy.copy(theta)

    tmp = h(tmp, x) - y
    result = tmp * x[index]
    return result


def batch_GD_step(theta, alpha, x, y):
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

    return new_theta


def random_GD_step(theta, alpha, x, y, index):
    new_theta = copy.deepcopy(theta)
    for i in range(theta.size):
        tmp = partial_j(theta, x[:, index], y[:, index], i)[0]
        new_theta[0, i] = theta[0, i] - alpha[i] * tmp
    return new_theta


def h(theta, x):
    return np.dot(theta, np.transpose(x))


def stop(x, y, theta):
    tmp = np.dot(theta, x) - y
    result = np.sum(tmp ** 2)
    return result





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
    min_error = 0.1
    max_turns = 200
    turns = 20
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

    length = 5
    x_dots = produce_random_x(length)

    noise = np.random.rand(1, length)
    y_dots = pruduce_y_with_noise(x_dots, noise, ture_h)

    b_x_dots = (x_dots - np.mean(x_dots)) / (np.max(x_dots) - np.min(x_dots))
    b_y_dots = (y_dots - np.mean(y_dots)) / (np.max(y_dots) - np.min(y_dots))

    x_dots_ex = np.concatenate((np.ones([1,length]),b_x_dots))

    ax1.scatter(b_x_dots, b_y_dots, marker='x')
    ax2.scatter(b_x_dots, b_y_dots, marker='x')

    x1 = np.arange(-0.5, 0.5, 0.05)
    y1 = np.arange(-0.5, 1.5, 0.05)
    X, Y = np.meshgrid(x1, y1)
    Z = j_theta(X, Y, x_dots_ex, b_y_dots)


    # fig2= plt.figure(figsize=(16,12), dpi=72, facecolor="white")
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
    CS = ax3.contour(X, Y, Z)
    ax3.clabel(CS, inline=1, fontsize=10)

    alpha_batch = [0.2, 0.2]
    alpha_random = [0.2, 0.2]

    theta_batch_sqs = theta_batch
    theta_random_sqs = theta_random
    count_batch = 0
    count_random = 0

#下面是两个不同的情况下的收敛，看情况用，注意要注释掉一个


#--------------------------这部分是完全靠终止条件-----------------------------
    # while True:
    #     tmp = batch_GD_step(theta_batch, alpha_batch, x_dots_ex, b_y_dots)
    #     theta_batch_sqs = np.concatenate((theta_batch_sqs, tmp))
    #     count_batch += 1
    #     line, = ax1.plot(x_line, h_plot_2d(tmp, x_line), '--', linewidth=2+0.1*count_batch, label=str(count_batch+1) + "th line")
    #     res = stop(x_dots_ex, b_y_dots, tmp)
    #     print "batch GD turns %s, theta is %s, and error is %s" % (count_batch, tmp, res)
    #     if res < min_error or count_batch>max_turns:
    #         break
    #     theta_batch = tmp

    # while True:
    #     tmp = random_GD_step(theta_random, alpha_random, x_dots_ex, b_y_dots, count_random%b_y_dots.size)
    #     theta_random_sqs = np.concatenate((theta_random_sqs, tmp))
    #     count_random += 1
    #     line, = ax2.plot(x_line, h_plot_2d(tmp, x_line), '--', linewidth=2+0.1*count_random, label=str(count_random+1) + "th line")
    #     res = stop(x_dots_ex, b_y_dots, tmp)
    #     print "random GD turns %s, theta is %s, and error is %s" % (count_random, tmp, res)
    #     if res < min_error or count_random>max_turns:
    #         break
    #     theta_random = tmp
# ----------------------------------------------------------------

# ----------------这部分是用来固定迭代次数的部分-------------------------
    for i in range(turns):
        tmp = batch_GD_step(theta_batch, alpha_batch, x_dots_ex, b_y_dots)
        theta_batch_sqs = np.concatenate((theta_batch_sqs, tmp))
        count_batch += 1
        line, = ax1.plot(x_line, h_plot_2d(tmp, x_line), '--', linewidth=2+0.1*count_batch, label=str(count_batch+1) + "th line")
        res = stop(x_dots_ex, b_y_dots, tmp)
        print "batch GD turns %s, theta is %s, and error is %s" % (count_batch, tmp, res)
        theta_batch = tmp

        tmp = random_GD_step(theta_random, alpha_random, x_dots_ex, b_y_dots, count_random%b_y_dots.size)
        theta_random_sqs = np.concatenate((theta_random_sqs, tmp))
        count_random += 1
        line, = ax2.plot(x_line, h_plot_2d(tmp, x_line), '--', linewidth=2+0.1*count_random, label=str(count_random+1) + "th line")
        res = stop(x_dots_ex, b_y_dots, tmp)
        print "random GD turns %s, theta is %s, and error is %s" % (count_random, tmp, res)
        theta_random = tmp
#--------------------------------------------------------------------

    ax3.plot(theta_batch_sqs[:, 0], theta_batch_sqs[:, 1])
    ax3.plot(theta_random_sqs[:, 0], theta_random_sqs[:, 1])

    for i in range(count_batch):
        ax3.scatter(theta_batch_sqs[i, 0], theta_batch_sqs[i, 1], marker='^', s=10+5*i)
    for i in range(count_random):
        ax3.scatter(theta_random_sqs[i, 0], theta_random_sqs[i, 1], marker='*', s=10+5*i)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.show()

