from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE

from ceo.sampling import stratified_shuffle, split_shuffle

def train_predictor(progs, x_train, y_train):

    count = dict()
    for i in range(4):
        count[i] = y_train.count(i)

    mprogs = []
    mx_train = []
    my_train = []

    for prog, x, label in zip(progs, x_train, y_train):
        if count[label] <= 10:
            continue

        mx_train.append(x)
        my_train.append("r"+str(label))
        mprogs.append(prog)

    progs = mprogs
    x_train = mx_train
    y_train = my_train
  
    gkf = []
    for train_index, test_index in GroupKFold( n_splits=10).split(x_train,y_train,progs):
        y = []
        for index in train_index:
            y.append(y_train[index])

        #for i in range(4):
        #    print i, y.count("r"+str(i)) 
        gkf.append((train_index, test_index))


    clf = SVC()
    parameters = [
    {'C': [0.1, 1, 10, 100, 1000, 10000], 'kernel': ['linear'], 'class_weight':['balanced']},
    {'C': [0.1, 1, 10, 100, 1000, 10000], 'gamma': [10,1,0.1, 0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf'], 'class_weight':['balanced']},
    ]

    grid_search = GridSearchCV(clf, parameters, scoring="recall_macro", cv=gkf, n_jobs=3)
    result = grid_search.fit(x_train, y_train)
    print result.best_estimator_
    print result.best_score_

    clf = RandomForestClassifier()
    parameters = [{'n_estimators' : [3,5,10]}]
    grid_search = GridSearchCV(clf, parameters, scoring="recall_macro", cv=gkf, n_jobs=3)
    result = grid_search.fit(x_train, y_train)
    print result.best_estimator_
    print result.best_score_

    clf = KNeighborsClassifier()
    parameters = [{'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}]
    grid_search = GridSearchCV(clf, parameters, scoring="recall_macro", cv=gkf, n_jobs=3)
    result = grid_search.fit(x_train, y_train)
    print result.best_estimator_
    print result.best_score_

    """assert(0)
    best_score = 0
    best_pred = None

    names = ["rf","svm", "knn"]
    evals = [eval_rf, eval_svm, eval_knn]
    preds = [train_rf, train_svm, train_knn]

    for name_pred, eval_pred, train_pred in zip(names, evals, preds):
        res = eval_pred(progs, x_train, y_train)
        #print ""
        print name_pred, res
        if res > best_score:
            best_score = res
            best_pred = train_pred(x_train, y_train)

    return best_pred
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

