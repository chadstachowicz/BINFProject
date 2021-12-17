import numpy as np
from numpy import linalg as LA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt

class MyLDA:
    def __init__(self, comps=None):
        self.comps = comps

    def run(self, X, Y):

        # get the shape of the data
        rows, columns = X.shape

        #how many clusters are in the data
        clusters_unique = np.unique(Y)
        cluster_length = len(clusters_unique)
        
        #create plot of data from covariance *  degrees of freedom
        data_plt = np.cov(X.T) * (rows - 1)
        
        scatter_w = 0
        
        #loop through the unique clusters to build within class scatter (intra-class)
        for i in range(cluster_length):
            class_items = np.flatnonzero(Y == clusters_unique[i])
            scatter_w = scatter_w + np.cov(X[class_items].T) * (len(class_items) - 1)
        print(scatter_w)

        #build between class scatter
        scatter_b = data_plt - scatter_w
        _, eig_vectors = np.linalg.eigh(np.linalg.pinv(scatter_w).dot(scatter_b))
        print(eig_vectors.shape)
        pc = X.dot(eig_vectors[:, ::-1][:, :self.comps])
        print(pc.shape)
        self.plot_data_2_class(pc,Y,X)
        self.plot_data_axis(pc, Y, X)
        return pc

    def plot_data(self,pc,Y,X):
        for color, label in zip(['y', 'r', 'g'], np.unique(Y)):
            graph_data = pc[np.flatnonzero(Y == label)]
            plt.scatter(graph_data[:, 0], graph_data[:, 1], c=color)
        plt.show()

    def plot_data_2_class(self,pc,Y,X):
        for color, label in zip(['y', 'r', 'g'], np.unique(Y)):
            graph_data = pc[np.flatnonzero(Y == label)]
            plt.scatter(graph_data[:, 0], graph_data[:, 1], c=color)
        plt.show()
    def plot_data_axis(self,pc,Y,X):
        lda = LinearDiscriminantAnalysis()
        lda_object = lda.fit(X, Y)
        for color, label in zip(['y', 'r', 'g'], np.unique(Y)):
            graph_data = pc[np.flatnonzero(Y == label)]
            plt.scatter(X[:, 0], X[:, 1], c=color)
        x1 = np.array([np.min(X[:, 0], axis=0), np.max(X[:, 0], axis=0)])
        b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]
        y1 = -(b + x1 * w1) / w2
        plt.plot(x1, y1, c="red")
        plt.show()

