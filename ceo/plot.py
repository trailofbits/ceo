from ceo.vectorizer import init_vectorizers

def plot_data(progs, x_train, y_train, verbose=0):
   
    data = dict()
    features_names = x_train[0]["exec_features"].keys()

    for feature in features_names:
        print "reading", feature
        data[feature] = []
        for x in x_train:
            data[feature].append(x["exec_features"][feature])

    
    exec_vectorizers, param_vectorizers = init_vectorizers(data)
    print exec_vectorizers
    print param_vectorizers

    for feature in features_names:
        print "vectorizing", feature 
        vectorizer = exec_vectorizers[feature]
        #print vectorizer.transform(data[feature])
 


 
