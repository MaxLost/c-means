import numpy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from kneed import KneeLocator
import pandas as pd
from matplotlib import pyplot as plt

from DataLoader import DataLoader
from FCM import FCM

# colors = [["#1a2ec7"], ["#20d420"], ["#de0b0b"], ["#fcba03"], ["#fc03e8"]]
colors = [[26, 46, 199], [32, 212, 32], [222, 11, 11], [252, 186, 3], [252, 3, 232]]


def get_color(x):
    col = [0, 0, 0]
    for i in range(len(x)):
        for j in range(3):
            col[j] += colors[i][j]*x[i]
    return ("#{:02x}{:02x}{:02x}").format(int(col[0]), int(col[1]), int(col[2]))


def simple_dataset():
    N_CLUSTERS = 3

    data = DataLoader("../circles.csv").load_data()

    x = FCM(N_CLUSTERS, 2, 1.0e-6, data)
    x.step()

    for j in range(x.w.shape[0]):
        plt.scatter(x.data.iat[j, 0], x.data.iat[j, 1], c=get_color(x.w.iloc[j]))
    ax = plt.subplot()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    model = KMeans(N_CLUSTERS).fit_predict(data)
    for j in range(data.shape[0]):
        w = ([0]*N_CLUSTERS)
        w[model[j]] = 1
        plt.scatter(data.iat[j, 0], data.iat[j, 1], c=get_color(w))
    ax = plt.subplot()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    model = AgglomerativeClustering(N_CLUSTERS, linkage="complete").fit_predict(data)
    for j in range(data.shape[0]):
        w = ([0] * N_CLUSTERS)
        w[model[j]] = 1
        plt.scatter(data.iat[j, 0], data.iat[j, 1], c=get_color(w))
    ax = plt.subplot()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def real_dataset():
    N_CLUSTERS = 3

    data = DataLoader("../country_data.csv").load_data()
    # data = data.drop(["Unnamed: 0", "CUST_ID", "TENURE"], axis=1)
    processed_data = data.drop("country", axis=1)
    # score = []
    # for k in range(1, 12):
    #     kmeans = KMeans(k)
    #     kmeans.fit(processed_data)
    #     score.append(kmeans.inertia_)
    # plt.plot(range(1, 12), score)
    # plt.show()

    # colors = [["#1a2ec7"], ["#20d420"], ["#de0b0b"], ["#fcba03"], ["#fc03e8"]]
    pca = PCA(n_components=2)
    p = pd.DataFrame(pca.fit_transform(processed_data), columns=['x', 'y'])

    model = KMeans(N_CLUSTERS).fit_predict(processed_data)
    for i in range(processed_data.shape[0]):
        w = ([0] * N_CLUSTERS)
        w[model[i]] = 1
        # print(model[i], w)
        # print(data.iloc[i])
        # print()
        plt.scatter(p.iat[i, 0], p.iat[i, 1], c=get_color(w))
    ax = plt.subplot()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    # model = AgglomerativeClustering(N_CLUSTERS, linkage="complete").fit_predict(processed_data)
    # for i in range(processed_data.shape[0]):
    #     w = ([0] * N_CLUSTERS)
    #     w[model[i]] = 1
    #     plt.scatter(p.iat[i, 0], p.iat[i, 1], c=get_color(w))
    # ax = plt.subplot()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # plt.show()
    #
    # model = FCM(N_CLUSTERS, 2, 10e-4, processed_data)
    # model.step()
    #
    # for i in range(processed_data.shape[0]):
    #     plt.scatter(p.iat[i, 0], p.iat[i, 1], c=get_color(model.w.iloc[i]))
    # ax = plt.subplot()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # plt.show()


# simple_dataset()
real_dataset()
