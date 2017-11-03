from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
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

    best_score = 0
    best_pred = None

    evals = [eval_rf, eval_svm, eval_knn]
    preds = [train_rf, train_svm, train_knn]

    for eval_pred, train_pred in zip(evals, preds):
        res = eval_pred(progs, x_train, y_train)
        if res > best_score:
            best_score = res
            best_pred = train_pred(x_train, y_train)

    return best_pred


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
        #if i == 0:
        #    conf = classification_report(y_test, clf.predict(x_test))

    return mean(scores)#, conf)

