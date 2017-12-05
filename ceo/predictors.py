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

    selected_features = x_train.keys()
    vectorizers = init_vectorizers(selected_features)
    vectorizers["params"] = vectorizers[option]
  
    gkf = []
    clfs = []
    params = []
  

    for _ in range(5):
        for train_index, test_index in GroupKFold( n_splits=10).split(x_train,y_train,progs):
            gkf.append((train_index, test_index))


    #clf = make_features_pipeline(selected_features, vectorizers, SVC())

    #parameters = [
    #{'C': [0.1, 1, 10, 100, 1000, 10000], 'kernel': ['linear'], 'class_weight':['balanced']},
    #{'classifier__C': [0.1, 1, 10, 100, 1000, 10000], 'classifier__gamma': [0.1, 0.01, 0.001], 'classifier__kernel': ['rbf'], 'classifier__class_weight':['balanced']},
    #]
    #clfs.append(clf)
    #params.append(parameters) 
    
    clf = make_features_pipeline(selected_features, vectorizers, RandomForestClassifier())

    parameters = [{'classifier__n_estimators' : [3,5,10]}]

    clfs.append(clf)
    params.append(parameters) 

    clf = make_features_pipeline(selected_features, vectorizers, KNeighborsClassifier())
    parameters = [{'classifier__n_neighbors':[1,2,3,4,5,6,7]}]

    clfs.append(clf)
    params.append(parameters) 
    
    best_score = 0
    best_estimator = None
    scores = []

    for clf, parameters in zip(clfs, params):

        grid_search = GridSearchCV(clf, parameters, scoring="recall_macro", cv=gkf, n_jobs=cpus, verbose=verbose)
        result = grid_search.fit(x_train, y_train)
        scores.append(result.best_score_)

        if verbose:
            print result.best_estimator_
            print result.best_score_
            #print result.best_estimator_.named_steps["pre"].transformer_list[0][1].named_steps["transmited-vectorizer"].vocabulary_ #.transformed_list #["transmited-vectorizer"]

        if result.best_score_ > best_score:
            best_estimator = result.best_estimator_
            best_score = result.best_score_
            if hasattr(best_estimator.named_steps["classifier"], "feature_importances_"):
                print "Feature importance:"
                print best_estimator.named_steps["classifier"].feature_importances_

    print option, map(lambda x: round(x,2), scores)
    return best_score, best_estimator
