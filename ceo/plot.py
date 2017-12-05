from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ceo.selector import make_features_pipeline
from ceo.vectorizer import init_vectorizers
import numpy as np

def plot_data(progs, option, X, y, verbose=0):
    # time to plot!
    np.set_printoptions(threshold='nan')
    np.set_printoptions(suppress=True)

    selected_features = X.keys()
    vectorizers = init_vectorizers(selected_features)
    
    if option != "none":
        vectorizers["params"] = vectorizers[option]

    print "[+] Reducing the dimensionality of the data to plot"
    reducer = Pipeline([
                ('scaler', (StandardScaler(with_mean=True,with_std=False))),
                ('reducer', PCA(n_components=None)),
                ])
    
    transformer = make_features_pipeline(selected_features, vectorizers, reducer)
    X_reduced = transformer.fit_transform(X)
    print map(lambda x: round(x,2), transformer.named_steps["classifier"].named_steps["reducer"].explained_variance_ratio_)
    
    
    for name, vectorizer in  transformer.named_steps["preprocesor"].transformer_list: 
        print name
        vectorizer_name = filter(lambda x: "vect" in x, vectorizer.named_steps.keys())[0]
        names = vectorizer.named_steps[vectorizer_name].vocabulary_
        for name, idx in sorted(names.items(), key=lambda x: x[1]):
            print name,
        print " "
 
    print "[+] Plotting"
 
    cmap = ["green", "blue", "yellow", "red"]
    plt.figure(figsize=(10,10))
    for i in range(len(X)):

        x0 = X_reduced[i, 0]
        x1 = X_reduced[i, 1]
        #print y[i]
        color = cmap[-1]
        if option != "none": 
            color = cmap[y[i]]
        
        plt.scatter(x0, x1, c = color)
        plt.text(x0-0.01, x1+0.01, progs[i])

    plt.savefig("plot."+option+".png", dpi=300)

    #print X_reduced
    # len(x_train), len(y_train)
