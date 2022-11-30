import math

import numpy as np
import pandas as pd
import random


class FCM:

    def __init__(self, cluster_count, m, epsilon, data):

        self.cluster_count = cluster_count
        self.m = m
        self.epsilon = epsilon
        self.data = data

        w = []
        for i in range(self.data.shape[0]):
            x = np.random.sample(self.cluster_count)
            w.append(x / sum(x))
        self.w = pd.DataFrame(w)

        c = []
        for _ in range(self.cluster_count):
            x = random.randint(0, self.data.shape[0])
            c.append(self.data.iloc[x].apply(lambda x: x + random.uniform(-10e-3, 10e-3)))
        self.c = pd.DataFrame(c)

    def distance(self, x, y):
        s = 0
        for i in range(self.data.shape[1]):
            s += (x[i] - y[i])**2
        return math.sqrt(s)

    def step(self):
        w = self.w.copy()
        for i in range(self.cluster_count):
            for j in range(self.data.shape[0]):
                dist_x_c = self.distance(self.data.iloc[j], self.c.iloc[i])
                if np.greater(dist_x_c, 0.0):
                    s = 0
                    for k in range(self.cluster_count):
                        s += 1 / (self.distance(self.data.iloc[j], self.c.iloc[k])**(2 / (self.m - 1)))
                    w.iat[j, i] = 1 / (dist_x_c**(2 / (self.m - 1)) * s)
                elif i == j:
                    w.iat[j, i] = 1
                else:
                    w.iat[j, i] = 0
        w = pd.DataFrame(w)

        c = []
        for i in range(self.cluster_count):
            s, s_x = 0.0, 0.0
            for j in range(self.data.shape[0]):
                a = w.iat[j, i] ** self.m
                s += a
                s_x += a * self.data.iloc[j]
            c.append(s_x / s)
        c = pd.DataFrame(c)

        diff = []
        for i in range(self.cluster_count):
            diff.append(self.distance(c.iloc[i], self.c.iloc[i]))
        if np.greater(max(diff), self.epsilon):
            self.w = w
            self.c = c
            print("Next step\n")
            self.step()
        else:
            return self.w
