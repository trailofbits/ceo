from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def plot_data(progs, X, y, verbose=0):
    print X[0].shape
    # time to plot!
    reducer = make_pipeline(StandardScaler(with_std=False), PCA(n_components=2))
    X_reduced = reducer.fit_transform(X)
    print reducer.components_

    cmap = ["green", "blue", "yellow", "red"]
    plt.figure(figsize=(10,10))

    for i in range(len(X)):

        x0 = X_reduced[i, 0]
        x1 = X_reduced[i, 1]
        #print y[i]
        plt.scatter(x0, x1, c = cmap[y[i]])
        plt.text(x0-0.01, x1+0.01, progs[i])
        plt.savefig("plot.png", dpi=300)
    #print X_reduced
    # len(x_train), len(y_train)
