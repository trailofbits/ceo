from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler, SMOTE

from ceo.sampling import stratified_shuffle, split_shuffle
from ceo.vectorizer import init_vectorizers
from ceo.selector import make_features_pipeline

def train_predictor(progs, option, x_train, y_train, cpus, verbose=0):

    #count = dict()
    #for i in range(4):
    #    count[i] = y_train.count(i)

    #mprogs = []
    #mx_train = []
    #my_train = []

    #for prog, x, label in zip(progs, x_train, y_train):
    #    if count[label] <= 10:
    #        continue
    #
    #    mx_train.append(x)
    #    my_train.append("r"+str(label))
    #    mprogs.append(prog)

    #progs = mprogs
    #x_train = mx_train
    #y_train = my_train

    selected_features = x_train.keys()
    vectorizers = init_vectorizers(selected_features)
    #vectorizers = exec_vectorizers
    vectorizers["params"] = vectorizers[option]
 
    #print vectorizers
    #preprocessing = make_feature_pipeline(selected_features, vectorizers)
    #print pipeline
    #print pipeline.fit_transform(X, y_train)[5]
    #assert(0)
  
    gkf = []
    clfs = []
    params = []

    #X = SDict(X)

    for train_index, test_index in GroupKFold( n_splits=2).split(x_train,y_train,progs):
        #y = []
        #for index in train_index:
        #    y.append(y_train[index])

        gkf.append((train_index, test_index))


    #clf = Pipeline([("pre",preprocessing), ("classifier", SVC())])
    """clf = make_pipeline(selected_features, vectorizers, SVC())

    print clf.get_params().keys()
    #assert(0)
    parameters = [
    #{'C': [0.1, 1, 10, 100, 1000, 10000], 'kernel': ['linear'], 'class_weight':['balanced']},
    {'classifier__C': [0.1, 1, 10, 100, 1000, 10000], 'classifier__gamma': [0.1, 0.01, 0.001], 'classifier__kernel': ['rbf'], 'classifier__class_weight':['balanced']},
    ]
    clfs.append(clf)
    params.append(parameters) 
    """
    #clf = RandomForestClassifier()
    clf = make_features_pipeline(selected_features, vectorizers, RandomForestClassifier())

    parameters = [{'classifier__n_estimators' : [3,5,10]}]

    clfs.append(clf)
    params.append(parameters) 

    #clf = KNeighborsClassifier()
    clf = make_features_pipeline(selected_features, vectorizers, KNeighborsClassifier())
    parameters = [{'classifier__n_neighbors':[1,2,3,4,5,6,7]}]

    clfs.append(clf)
    params.append(parameters) 
    
    best_score = 0
    best_estimator = None

    for clf, parameters in zip(clfs, params):

        grid_search = GridSearchCV(clf, parameters, scoring="recall_macro", cv=gkf, n_jobs=cpus)
        result = grid_search.fit(x_train, y_train)

        if verbose:
            print result.best_estimator_
            print result.best_score_
            #print result.best_estimator_.named_steps["pre"].transformer_list[0][1].named_steps["transmited-vectorizer"].vocabulary_ #.transformed_list #["transmited-vectorizer"]

        if result.best_score_ > best_score:
            best_estimator = result.best_estimator_
            best_score = result.best_score_

    return best_estimator

"""
def eval_rf(progs, X, labels): 
  
    unique_progs = list(set(progs))
    #print set(labels)
    scores = []
    for i in range(40):

        progs_train, progs_test = split_shuffle(unique_progs)
        x_train, x_test = [], []
        y_train, y_test = [], []

        for prog, x, y in zip(progs, X, labels):
            #print prog
            if prog in progs_train:
                x_train.append(x)
                y_train.append(y)
            elif prog in progs_test:
                x_test.append(x)
                y_test.append(y)
            else:
                assert(0)

        #for i in range(4):
        #    print i, y_train.count("r"+str(i)), "-",

        #print set(y_test), 

        ros = RandomOverSampler()
        x_train, y_train = ros.fit_sample(x_train, y_train)
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        result = recall_score(y_test, clf.predict(x_test), average=None)
        scores.extend(result)
        #if i == 0:
        #    conf = classification_report(y_test, clf.predict(x_test))
    
    return mean(scores) #, conf)

def train_rf(x_train, y_train):
    ros = RandomOverSampler()
    x_train, y_train = ros.fit_sample(x_train, y_train)
 
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    return clf



def eval_svm(progs, X, labels):
 
    unique_progs = list(set(progs))
    scores = []
    for i in range(40):

        progs_train, progs_test = split_shuffle(unique_progs)
        

        x_train, x_test = [], []
        y_train, y_test = [], []

        for prog, x, y in zip(progs, X, labels):
            #print prog
            if prog in progs_train:
                x_train.append(x)
                y_train.append(y)
            elif prog in progs_test:
                x_test.append(x)
                y_test.append(y)
            else:
                assert(0)

        #print set(y_test), 
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_sample(x_train, y_train) 

        clf = SVC()
        clf.fit(x_train, y_train)
        result = recall_score(y_test, clf.predict(x_test), average=None)
        scores.extend(result)
        #if i == 0:
        #    conf = classification_report(y_test, clf.predict(x_test))

    return mean(scores)#, conf)

def train_svm(x_train, y_train):
    ros = RandomOverSampler()
    x_train, y_train = ros.fit_sample(x_train, y_train)
 
    clf = SVC()
    clf.fit(x_train, y_train)

    return clf


def train_knn(x_train, y_train):
    ros = RandomOverSampler()
    x_train, y_train = ros.fit_sample(x_train, y_train)
 
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    return clf


def eval_knn(progs, X, labels):
    unique_progs = list(set(progs))
    scores = []
    for i in range(40):

        progs_train, progs_test = split_shuffle(unique_progs)

        x_train, x_test = [], []
        y_train, y_test = [], []

        for prog, x, y in zip(progs, X, labels):
            #print prog
            if prog in progs_train:
                x_train.append(x)
                y_train.append(y)
            elif prog in progs_test:
                x_test.append(x)
                y_test.append(y)
            else:
                assert(0)

        #print set(y_test), 
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_sample(x_train, y_train)
 
        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)
        result = recall_score(y_test, clf.predict(x_test), average=None)
        scores.extend(result)
        #if i == 0:
        #    conf = classification_report(y_test, clf.predict(x_test))

    return mean(scores)#, conf)
"""
