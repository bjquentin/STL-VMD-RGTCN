import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from pyDOE import lhs

def initial(pop, dim, ub, lb):
    X = []
    for i in range(dim):
        X1 = lb[i] + (ub[i] - lb[i]) * lhs(1, pop)
        X.append(X1)
    X = np.array(X).T
    X = X.reshape(pop, dim)
    return X, lb, ub


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def AOA(pop, M_iter, dim, lb, ub, fun, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9):
    # EPSILON = 1e-20
    time1 = datetime.datetime.now()
    EPSILON = 10e-10

    X, lb, ub = initial(pop, dim, ub, lb)

    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([M_iter, 1])

    for C_Iter in range(M_iter):
        time2 = datetime.datetime.now()
        moa = moa_min + (C_Iter + 1) * ((moa_max - moa_min) / M_iter)  # Eq 2
        mop = 1 - ((C_Iter + 1) ** (1.0 / alpha)) / (M_iter ** (1.0 / alpha))  # Eq 4

        for i in range(pop):
            for j in range(dim):
                r1, r2, r3 = np.random.rand(3)
                if r1 > moa:  # Exploration phase Eq 3
                    if r2 < 0.5:
                        X[i, j] = GbestPositon[0, j] / (mop + EPSILON) * ((ub[j] - lb[j]) * miu + lb[j])
                    else:
                        X[i, j] = GbestPositon[0, j] * mop * ((ub[j] - lb[j]) * miu + lb[j])

                else:  # Exploitation phase Eq 5
                    if r3 < 0.5:
                        X[i, j] = GbestPositon[0, j] - mop * ((ub[j] - lb[j]) * miu + lb[j])
                    else:
                        X[i, j] = GbestPositon[0, j] + mop * ((ub[j] - lb[j]) * miu + lb[j])

        X = BorderCheck(X, ub, lb, pop, dim)

        fitness = CaculateFitness(X, fun)
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        if (fitness[0] <= GbestScore):
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[C_Iter] = GbestScore
        time3 = datetime.datetime.now()
    return GbestScore, GbestPositon, Curve


if __name__ == '__main__':
    def f(nest):
        a, b, c = nest
        return a + b + c
    lb = [1, 1, 0.000001]
    ub = [80, 80, 0.01]
    m, n, t = AOA(100, 20, 3, lb, ub, f)
