import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

class MachineLearning:

    # constructor
    def __init__(self, algo):
        self.algorithm = algo
        self.result = {}
        self.cost_ = []

    # produce a plot of the regression vs. the data.
    def graph_line(self, x, y, coef):
        # make a scatter plot
        plt.scatter(x, y, color="#81D8D0", marker="v", s=50)
        # use our predicted y to make a line
        y_pred = coef[0] + coef[1] * x
        # plot the line
        plt.plot(x, y_pred, color="g")
        # label it
        plt.xlabel('x')
        plt.ylabel('y')
        # show the plot
        plt.show()
    #calculate the best fit line components
    def best_fit(self, x, y):
        #how many pieces of data are there
        n = np.size(x)
        # calculate the means
        mx = np.mean(x)
        my = np.mean(y)
        #calc ssxy and ssxx
        SS_xy = np.sum(y * x) - n * my * mx
        SS_xx = np.sum(x * x) - n * mx * mx
        # calculating regression coefficients
        coef2 = SS_xy / SS_xx
        coef1 = my - coef2 * mx
        return (coef1, coef2)

    def train_linear_regression(self,x,y,eta=0.001,loops=30):
        # make a blank array the size of one of our input to hold current predicted y
        self.w_ = np.zeros((x.shape[1], 1))
        # get number of rows of data
        m = x.shape[0]

        # loop through the amount of defined learning iterations for our regression
        for data in range(loops):
            # predict y
            predicted_y = self.predict_linear_regression(x)
            # calc y difference vector
            y_diff = predicted_y - y
            #calc gradient_vector
            gradient_vector = np.dot(x.T, y_diff)
            # calculate vector updates
            self.w_ -= (eta / m) * gradient_vector
            # calculate cost
            cost = np.sum((y_diff ** 2)) / (2 * m)
            # append cost to cost array
            self.cost_.append(cost)
            # put out best fit plot
            b = self.best_fit(x,predicted_y)
            self.graph_line(x,y,b)
        return self


    def predict_linear_regression(self, x):
        return np.dot(x, self.w_)

    def pca(self, data,corr="false"):
       # print(data)
        #mean of column 1
        data_mean = data.mean(axis=0)
        # mean of column 1
        data_std = data.std(axis=0)
        #make an array of the same length of the data, of the mean over and over
        array_of_means = np.tile(data_mean,reps=(data.shape[0], 1))
        #make an array of the same length of the data, of the std over and over
        array_of_std = np.tile(data_std,reps=(data.shape[0], 1))
        print(array_of_std)
        #find the centered mean
        mean_centered = data - array_of_means

        #covariance input
        cov_data = data / array_of_std

        #calc covariance - since no correction we use data directly
        if corr=="false":
            covariance = np.cov(data, rowvar=False)
        else:
            covariance = np.cov(cov_data, rowvar=False)
        print(covariance)
        #calculate eigen decompostion

        v1, v2 = LA.eig(covariance)

        #eigenvalues
        v1 = v1.real
        #print(v1)
        #eigenvectors
        v2 = v2.real

        #print(v2)
        #reverse the array
        arg = v1.argsort()[::-1]
        eigen_vals = v1[arg]
        eigen_vects = v2[:, arg]

        #print(eigen_vects)
        #calculate percent variance
        variance_percent = eigen_vals / sum(eigen_vals) * 100


        # perform matrix multiple from PCA data to eigen_vectors
        scores = np.matmul(cov_data, eigen_vects)

        #store PCS results
        self.result = {'raw_data': data, \
                        'pca_data': cov_data, \
                        'variance': eigen_vals, \
                        'variance_percent': variance_percent, \
                        'loadings': eigen_vects, \
                        'scores': scores}
        return self.result

    def plot_pca_scores(self):
        graph, g = plt.subplots()
        g.scatter(self.result['scores'][:, 0], self.result['scores'][:, 1], color='green')
        g.set_title('PCA Score Plot')
        g.set_ylabel('PC 2 (' + str(round(self.result['variance_percent'][1])) + '%)')
        g.set_xlabel('PC 1 (' + str(round(self.result['variance_percent'][0])) + '%)')
        graph.show()

    def plot_pca_raw_data(self,k=-20):
        # project dta onto PCA
        graph, g = plt.subplots()
        g.scatter(self.result['raw_data'][:, 0], self.result['raw_data'][:, 1], c=['blue'])
        #g.plot([0, k * self.result['loadings'][0, 0]], [0, k * self.result['loadings'][1, 0]],
        #        color='red', linewidth=3, label='PC 1')
        #g.plot([0, (-k) * self.result['loadings'][0, 1]], [0, (-k) * self.result['loadings'][1, 1]],
        #        color='green', linewidth=3, label='PC 2')
        g.set_title('Raw Data')
        g.set_xlabel('sample')
        g.set_ylabel('data')
        g.legend()
        graph.show()

    def plot_pca_projection(self,data):
        pca = PCA(n_components=1)
        pca.fit(data)
        X_pca = pca.transform(data)
        X_new = pca.inverse_transform(X_pca)
        #  plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
        plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
        # plt.axis('equal')
        plt.show()
        print("original shape:   ", data.shape)
        print("transformed shape:", X_pca.shape)

    def plot_pca_raw_data_pca(self,k=40):
        # project dta onto PCA
        graph, g = plt.subplots()
        g.scatter(self.result['raw_data'][:, 0], self.result['raw_data'][:, 1], c=['blue'])
        g.plot([0, k * self.result['loadings'][0, 0]], [0, k * self.result['loadings'][1, 0]],
                color='red', linewidth=3, label='PC 1')
        #g.plot([0, (-k) * self.result['loadings'][0, 1]], [0, (-k) * self.result['loadings'][1, 1]],
        #        color='green', linewidth=3, label='PC 2')
        g.set_title('Raw Data vs PC1')
        g.set_xlabel('v1')
        g.set_ylabel('v2')
        g.legend()
        graph.show()

    def plot_pca_scree(self):
      # scree (variance) plot
        graph, g = plt.subplots()
        g.scatter(range(len(self.result['variance_percent'])), \
                   self.result['variance_percent'],
                   color='green')
        g.set_title('scree (variance) plot')
        g.set_xlabel('PC index')
        g.set_ylabel('Percent Variance')
        g.set_ylim((-10.0, 110.0))
        graph.show()

    def plot_pca_loadings(self):
        graph, g = plt.subplots()
        g.scatter(self.result['loadings'][:, 0], self.result['loadings'][:, 1], color='blue')
        g.set_title('Loadings plot')
        g.set_xlabel('PC1')
        g.set_ylabel('PC2')
        for i in range(self.result['loadings'].shape[0]):
            g.text(self.result['loadings'][i, 0], self.result['loadings'][i, 1], 'x' + str(i + 1))
        graph.show()