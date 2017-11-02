from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE
from ceo.sampling import stratified_shuffle, split_shuffle

# TODO: train and test with different programs

def eval_rf(progs, X, labels): 
  
    unique_progs = list(set(progs))
    print set(labels)
    #print len(x_test)
    scores = []
    for i in range(100):

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

        ros = RandomOverSampler()
        x_train, y_train = ros.fit_sample(x_train, y_train)
        #x_train, y_train, x_test, y_test = stratified_shuffle(X,labels)  
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        result = recall_score(y_test, clf.predict(x_test), average=None)
        scores.extend(result)
        if i == 0:
            conf = classification_report(y_test, clf.predict(x_test))
    return (mean(scores), conf)

def eval_svc(progs, X, labels):
 
    unique_progs = list(set(progs))
    #print len(x_train)
    #print len(x_test)
    scores = []
    for i in range(100):

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

        ros = RandomOverSampler()
        x_train, y_train = ros.fit_sample(x_train, y_train) 

        clf = SVC()
        clf.fit(x_train, y_train)
        result = recall_score(y_test, clf.predict(x_test), average=None)
        scores.extend(result)
        if i == 0:
            conf = classification_report(y_test, clf.predict(x_test))

    return (mean(scores), conf)


def train_knn(x_train, y_train):
    ros = RandomOverSampler()
    x_train, y_train = ros.fit_sample(x_train, y_train)
 
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    return clf


def eval_knn(progs, X, labels):
    unique_progs = list(set(progs))
    #print len(x_train)
    #print len(x_test)
    scores = []
    for i in range(100):

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

        ros = RandomOverSampler()
        x_train, y_train = ros.fit_sample(x_train, y_train)
 
        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)
        result = recall_score(y_test, clf.predict(x_test), average=None)
        scores.extend(result)
        if i == 0:
            conf = classification_report(y_test, clf.predict(x_test))

    return (mean(scores), conf)

