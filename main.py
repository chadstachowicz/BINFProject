from machine_learning import MachineLearning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from MyLDA import MyLDA
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

example_algo = "lda_pca"

def read_data(name,rowskip=1,usecol=None,transpose="false"):
    file = name
    arr = np.loadtxt(file, delimiter=',', skiprows=rowskip, usecols=usecol)
    if transpose == "true":
        arr = arr.transpose()
    return arr


def main():
    if example_algo == "lda":
        # load the iris dataset
        name = "./dataset_1.csv"
        test_data = read_data(name, 1, None)
        x_array = []
        y_array = []
        label_array = []
        for d in test_data:
            x_array.append([d[0]])
            y_array.append([d[1]])
            label_array.append([d[2]])
        x = np.array(x_array)
        y = np.array(y_array)
        label_array = np.array(label_array)
        combined = np.column_stack((x, y))
        print(label_array.shape)
        X = combined
        Y = label_array
        X_training_data, X_testing_data, Y_training_data, Y_testing_data = train_test_split(X, Y, test_size=0.3)

        LDA_object = MyLDA(comps=2)
        trained = LDA_object.run(X_training_data, Y_training_data)

    if example_algo == "lda_sklearn_iris":
        test_data = load_iris()
        X = test_data.data
        Y = test_data.target
        X_training_data, X_testing_data, Y_training_data, Y_testing_data = train_test_split(X, Y, test_size=0.3)
        sklda = LDA(n_components=2)
        X_training_data = sklda.fit_transform(X_training_data, Y_training_data)
        print(sklda.explained_variance_ratio_)
        for color, label in zip(['y', 'r', 'g'], np.unique(Y_training_data)):
            graph_data = X_training_data[np.flatnonzero(Y_training_data == label)]
            plt.scatter(graph_data[:, 0], graph_data[:, 1], c=color)
        plt.show()
    if example_algo == "lda_iris":
        #load the iris dataset
        test_data = load_iris()
        print(test_data)

        X = test_data.data
        Y = test_data.target
        X_training_data, X_testing_data, Y_training_data, Y_testing_data = train_test_split(X, Y, test_size=0.3)

        LDA_object = MyLDA(comps=2)
        pc = LDA_object.run(X_training_data, Y_training_data)
        for color, label in zip(['y', 'r', 'g'], np.unique(Y_training_data)):
            graph_data = pc[np.flatnonzero(Y_training_data == label)]
            plt.scatter(graph_data[:, 0], graph_data[:, 1], c=color)
        plt.show()
    if example_algo == "lda_pca":
        k = 40
        name = "./dataset_1.csv"
        data = read_data(name,1,None)
        ml = MachineLearning("pca")
        ml.plot_pca_projection(data)
        lda = LDA()
        test_data = read_data(name, 1, None)
        x_array = []
        y_array = []
        label_array = []
        for d in test_data:
            x_array.append([d[0]])
            y_array.append([d[1]])
            label_array.append([d[2]])
        x = np.array(x_array)
        y = np.array(y_array)
        label_array = np.array(label_array)
        combined = np.column_stack((x, y))
        X = combined
        Y = label_array
        lda_object = lda.fit(X, Y)
        result = ml.pca(data)
        print("transformed shape:", result['pca_data'].shape)
        graph, g = plt.subplots()
        g.scatter(result['raw_data'][:, 0], result['raw_data'][:, 1], c=['blue'])
        g.plot([0, k * result['loadings'][0, 0]], [0, k * result['loadings'][1, 0]],
           color='red', linewidth=3, label='PC 1')
        x1 = np.array([np.min(X[:, 0], axis=0), np.max(X[:, 0], axis=0)])
        b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]
        y1 = -(b + x1 * w1) / w2
        plt.plot(x1, y1, c="green", label="LDA")
        # g.plot([0, (-k) * self.result['loadings'][0, 1]], [0, (-k) * self.result['loadings'][1, 1]],
        #        color='green', linewidth=3, label='PC 2')
        g.set_title('Raw Data vs PC1 vs LDA')
        g.set_xlabel('v1')
        g.set_ylabel('v2')
        g.legend()
        graph.show()

    if example_algo == "pca":
        name = "./dataset_1.csv"
        data = read_data(name,1,None)
        ml = MachineLearning("pca")
        ml.plot_pca_projection(data)
        result = ml.pca(data)
        print("transformed shape:", result['pca_data'].shape)
        ml.plot_pca_scree()
       # ml.plot_pca_scores()
       # ml.plot_pca_loadings()
        ml.plot_pca_raw_data()
        ml.plot_pca_raw_data_pca()
      #  ml.plot_pca_data()

      #  name = "./Homework_2_dataset_prob4.csv"
      #  data = read_data(name,1,(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),"true")
      #  ml2 = MachineLearning("pca")
      #  result = ml2.pca(data)

     #   ml2.plot_pca_scree()

        # scores plot
    #    graph, g = plt.subplots()
        #  print(result['scores'])
     #   g.set_title('scores plot')
     #   g.scatter(result['scores'][:, 0], result['scores'][:, 1], color='blue')
     #   g.scatter(result['scores'][0:20, 0], result['scores'][0:20, 1], color='red')
     #   g.set_xlabel('PC1')
     #   g.set_ylabel('PC2')
     #   graph.show()

   #     ml2.plot_pca_loadings()
    elif example_algo == "lin_reg":
        name = "./lin_reg_data.csv"
        x_array = []
        y_array = []
        data = read_data(name, 1, None)
        ml = MachineLearning("lin_reg")
        for d in data:
            x_array.append([d[0]])
            y_array.append([d[1]])
        x = np.array(x_array)
        y = np.array(y_array)
        print(x.shape)
        ml.train_linear_regression(x,y)
        print(ml.predict_linear_regression(12))



if __name__ == '__main__':
    main()
