from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.utils import shuffle as skshuffle

def shuffle(x, random_state=None):
    return skshuffle(x, random_state=random_state)

def split_shuffle(X,y=None, random_state=None):
    sss = ShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    train_index, test_index  = sss.split(X, y).next()

    for index in train_index:
        X_train.append(X[index])
        if y is not None:
            y_train.append(y[index])


    for index in test_index:
        X_test.append(X[index])
        if y is not None:
            y_test.append(y[index])

    if y is not None:
        return X_train, y_train, X_test, y_test
    else:
        return X_train, X_test

def stratified_shuffle(X,y, random_state=None):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    X_train = []
    X_test = []
    y_train = []
    y_test = []


    train_index, test_index  = sss.split(X, y).next()

    for index in train_index:
        X_train.append(X[index])
        y_train.append(y[index])


    for index in test_index:
        X_test.append(X[index])
        y_test.append(y[index])


    return X_train, y_train, X_test, y_test
