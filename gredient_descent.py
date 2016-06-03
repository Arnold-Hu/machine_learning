# coding=utf-8

import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
# 预定义 for each x, xi = 1

def j_theta(X, Y, training_set):
    result = copy.deepcopy(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            s = 0
            for x_y in training_set:
                s += (h((X[i][j], Y[i][j]), x_y[0]) - x_y[1])**2
            result[i][j] = s / (2*len(training_set))
    print "result " , result
    return result


def partial_j(theta, x, y, index):
    # print "theta %s" % theta
    # print "x, y, index %s %s %s"% (x, y, index)
    # print "h(theta, x) is %s" % h(theta, x)
    tmp = h(theta, x) - y
    result = tmp * x[index]
    # print result
    return result


def batch_GD_step(theta, alpha, training_set):
    new_theta = copy.deepcopy(theta)
    # for index, ele in enumerate(theta):
    #     theta[index] = ele - alpha*partial_j(theta, x, y, index)
    for index, ele in enumerate(theta):
        s = 0
        for x_y in training_set:
            s += partial_j(theta, x_y[0], x_y[1], index)
            # tmp = (h(theta, x_y[0]) - x_y[1]) * x_y[index]
            # s += tmp
        s = s / len(training_set)
        new_theta[index] = theta[index] - alpha[index]*s
        print " batch step length", s ,"index" , index
    return new_theta


def random_GD_step(theta, alpha, training_set, index):
    new_theta = copy.deepcopy(theta)
    x_y = training_set[index]
    for i, _ in enumerate(theta):
        tmp = partial_j(theta, x_y[0], x_y[1], i)
        print "random step length", tmp, "index", i, "sample", index
        new_theta[i] = theta[i] - alpha[i]*tmp
    return new_theta


def h(theta, x):
    return sum([i[0] * i[1] for i in zip(theta, x)])


def stop(theta, new_theta):
    if sum(abs(i[0] - i[1]) for i in zip(theta, new_theta)):
        return True
    return False

def produce_random_point(limit_size=100):
    x = random.random()
    # y = math.sin(x/10)*100 + random.random()*200
    y = ture_h(x) + random.random()
    return [x,y]

def h_plot_2d(theta, x):
    return theta[0] + theta[1]*x

def ture_h(x):
    return 2*x + 10
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


    x = np.linspace(-1, 1)
    theta_batch = [0, 0]
    theta_random = [0, 0]
    dots = []

    fig = plt.figure(figsize=(16,12), dpi=72, facecolor="white")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    for i in range(20):
        dots.append(produce_random_point())

    dots_x = [i[0] for i in dots]
    dots_y = [i[1] for i in dots]
    average_x = sum(dots_x) / len(dots_x)
    average_y = sum(dots_y) / len(dots_y)
    gap_x = max(dots_x) - min(dots_x)
    gap_y = max(dots_y) - min(dots_y)

    new_x = [(i- average_x) / gap_x for i in dots_x]
    new_y = [(i- average_y) / gap_y for i in dots_y]
    dots = [[i[0],i[1]] for i in zip(new_x, new_y)]

    for i in dots:
        i[0] = (1, i[0])

    ax1.scatter([i[0][1] for i in dots], [i[1] for i in dots], marker='x')
    ax2.scatter([i[0][1] for i in dots], [i[1] for i in dots], marker='x')

    x1 = np.arange(-0.5, 0.5, 0.05)
    y1 = np.arange(-1, 3, 0.05)
    X, Y = np.meshgrid(x1, y1)
    Z = j_theta(X, Y, dots)
    print average_y
    theta_random[0] = average_y/gap_y

    # fig2= plt.figure(figsize=(16,12), dpi=72, facecolor="white")
    # ax = mplot3d.Axes3D(fig2)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
    CS = ax3.contour(X, Y, Z)
    ax3.clabel(CS, inline=1, fontsize=10)


    # line, = ax1.plot(x, ture_h(x), '-', linewidth=5, label="ture")
    # line, = ax2.plot(x, ture_h(x), '-', linewidth=5, label="ture")
    alpha_batch = [1, 1]
    alpha_random = [1, 1]
    theta_batch_sqs = [[],[]]
    theta_random_sqs = [[], []]
    for i in range(20):
        theta_batch = batch_GD_step(theta_batch, alpha_batch, dots)
        theta_batch_sqs[0].append(theta_batch[0])
        theta_batch_sqs[1].append(theta_batch[1])
        theta_random = random_GD_step(theta_random, alpha_random, dots, i%len(dots_x))
        theta_random_sqs[0].append(theta_random[0])
        theta_random_sqs[1].append(theta_random[1])
        print "batch theta", theta_batch, "random theta", theta_random
        line, = ax1.plot(x, h_plot_2d(theta_batch, x), '--', linewidth=2+0.1*i, label=str(i+1) + "th line")
        line, = ax2.plot(x, h_plot_2d(theta_random, x), '--', linewidth=2+0.1*i, label=str(i+1) + "th line")
    ax3.plot(theta_batch_sqs[0], theta_batch_sqs[1], marker='x')
    ax3.plot(theta_random_sqs[0], theta_random_sqs[1], marker='^')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.show()

