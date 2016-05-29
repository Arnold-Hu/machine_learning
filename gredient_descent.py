# coding=utf-8

import random
import numpy as np
import matplotlib.pyplot as plt

def partial_j(theta, x, y, index):
    sum = 0
    for i in zip(x, y):
        result = h(theta, i[0]) - i[1]
        sum += result * x[index]
    return sum


def iteration_of_theta(theta, alpha, x, y):
    while not (stop_iter(theta)):
        for index, ele in enumerate(theta):
            theta[index] = ele - alpha*partial_j(theta, x, y, index)
    return theta


def h(theta, x)
