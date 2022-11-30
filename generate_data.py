import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random

data = pd.DataFrame(columns=['x', 'y'])

center = [5, 5]
data.loc[len(data)] = center
for _ in range(100):
    point = [center[0]+random.uniform(-1.8, 1.8), center[1]+random.uniform(-1.8, 1.8)]
    data.loc[len(data)] = point

center = [7, 3]
data.loc[len(data)] = center
for _ in range(100):
    point = [center[0]+random.uniform(-1.8, 1.8), center[1]+random.uniform(-1.8, 1.8)]
    data.loc[len(data)] = point

center = [7, 7]
data.loc[len(data)] = center
for _ in range(100):
    point = [center[0]+random.uniform(-1.8, 1.8), center[1]+random.uniform(-1.8, 1.8)]
    data.loc[len(data)] = point

plt.scatter(data['x'], data['y'])
plt.show()
# for i in range(3):
#     center = [random.uniform(5, 30), random.uniform(5, 30)]
#     data.append(pd.DataFrame(columns=["x", "y"]))
#     data[i].loc[len(data[i])] = center
#
#     for _ in range(199):
#         point = [center[0]+random.uniform(-3, 3), center[1]+random.uniform(-3, 3)]
#         data[i].loc[len(data[i])] = point
#
# for i in range(3):
#     plt.scatter(data[i]["x"], data[i]["y"])
#
# plt.show()
# result = pd.DataFrame(columns=["x", "y"])
# for x in data:
#     result = pd.concat([result, x])

data.to_csv("../circles.csv", index=False)
