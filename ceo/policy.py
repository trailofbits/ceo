import os
import random
import sys
import numpy as np
import pickle as pickle
from scipy import stats
from sklearn.externals import joblib
from sklearn.semi_supervised import label_propagation

from ceo.vectorizer import TFIDFVectorizer, CountVectorizer, Sent2VecVectorizer
from ceo.predictors import train_predictor
from ceo.labels import lbls

class Policy(object):
    ''' Base class for prioritization '''
    pass

class Storage(object):
    def __init__(self, path, features_list, options):
        self.storage = str(path)
        self.options = list(options)
        self.features_list = list(features_list)
        self.testcases = dict() # tid -> testcase
        self.labels = dict()    # str -> (tid, pid) -> int
 
        for x in options:
            self.labels[x] = dict()

    def __contains__(self, x):
        if self.testcases is None:
            return False
        return hash(x) in self.testcases   
   
    def add(self, tc, exec_features, param_features, labels):
        
        # testcases
        tid = hash(tc) 
        if tid not in self.testcases:
            self.testcases[tid] = tc

        # save exec features
        exec_features.save(self.storage+"/exec_features/"+str(tid)+".csv.gz")
        
        # add label
        for option in self.options:
            pid = hash(param_features[option])
            self.labels[option][(tid, pid)] = labels[option] 

        # save params features
        for option in self.options:
            pid = hash(param_features[option])
            param_features[option].save(self.storage+"/param_features/"+option+"."+str(tid)+"."+str(pid)+".csv.gz")

        # save testcases and labels dicts
        joblib.dump(self.testcases, self.storage+'/testcases.pkl')
        joblib.dump(self.labels, self.storage+'/labels.pkl')

class MultiPolicy(Policy):
    def __init__(self, options, storages):
       
        self.features_list = None 
        self.storages = list(storages)
        self.options = list(options)
        self.param_generators = None
        self.exec_features = dict()   # tid -> narray
        self.param_features = dict()  # str -> (tid,pid) -> narray
        self.labels = dict()          # str -> (tid, pid) -> int
        self.testcases = dict()       # tid -> testcase

        for x in options:
            self.param_features[x] = dict()
            self.labels[x] = dict()

        self.exec_vectorizers = None
        self.param_vectorizers = None
        
        self.predictors = dict()
        for x in options:
            self.predictors[x] = label_propagation.LabelSpreading(kernel='knn', alpha=0.2)

        self._load()

    def _load(self):
        self.storage = self.storages[0]
        self.exec_vectorizers = joblib.load(open(self.storage+'/exec_vectorizers.pkl', 'rb'))
        self.features_list = self.exec_vectorizers.keys()
        self.param_vectorizers = joblib.load(open(self.storage+'/param_vectorizers.pkl', 'rb'))
        self.param_generators = joblib.load(open(self.storage+'/param_generators.pkl', 'rb'))

        for storage in self.storages:
            #self.testcases =
            testcases = joblib.load(open(storage+'/testcases.pkl', 'rb'))
            for x,y in testcases.items():
                assert(not (x in self.testcases))
                self.testcases[x] = y
            #self.labels = 
            labels = joblib.load(open(storage+'/labels.pkl', 'rb'))
            for option in self.options:
                for x,y in labels[option].items():
                    assert(not (x in self.labels))
                    self.labels[option][x] = y
 
            for x, y, files in os.walk(storage+"/param_features"):
                for f in files:
                    feature_filepath = x + "/".join(y) + "/" + f
                    option = str(f.split(".")[0])
                    if not (option in self.options):
                        continue
                    tid = int(f.split(".")[1])
                    pid = int(f.split(".")[2])
                    #print feature_filepath, x,y,f
                    self.param_features[option][(tid,pid)] = np.load(feature_filepath)['arr_0']

            for x, y, files in os.walk(storage+"/exec_features"):
                for f in files:
                    feature_filepath = x + "/".join(y) + "/" + f
                    tid = int(f.split(".")[0])
                    #print feature_filepath, x,y,f
                    self.exec_features[tid] = np.load(feature_filepath)['arr_0']

    def __contains__(self, x):
        if self.exec_features is None:
            return False
        return hash(x) in self.exec_features
 
    def join_features(self, option, (tid, pid)):
        v = []
        v.append(self.param_features[option][tid,pid])
        v.append(self.exec_features[tid])
        return np.concatenate(v)

    def get_data(self):
        ret = dict()

        for option in self.options:

            X = []
            labels = []
            progs = []

            for ((tid,pid),label) in self.labels[option].items():
                v = self.join_features(option, (tid,pid))
                X.append(v)
                labels.append(label)
                progs.append(self.testcases[tid].target_filename)

            ret[option] = progs,X,labels

        return ret

    def choice(self, raw_exec_features, cpus):
        ret = dict()
        pred = dict()
        data = self.get_data()

        for option in self.options:

            ret[option] = []
            progs, X_train, y_train = data[option]
            X_test = []
            params_test = []

            for raw_param_features in self.param_generators[option].enumerate():

                features = [self.param_vectorizers[option].transform([raw_param_features])]

                for name in self.features_list:
                    exec_vectorizer = self.exec_vectorizers[name]
                    raw_exec_feature = raw_exec_features[name]
                    row = exec_vectorizer.transform([raw_exec_feature])
                    features.append(row)

                #for i,raw_feature in enumerate(raw_exec_features):
                #    row = self.exec_vectorizers[i].transform([raw_feature])
                #    features.append(row)
                features = np.concatenate(features, axis=1).flatten()
                X_test.append(features)
                params_test.append(raw_param_features)
           
            
            clf = train_predictor(progs, X_train, y_train, cpus)
            results = clf.predict(X_test)
            pred[option] = clf
             
            for i,label in enumerate(results):
                ret[option].append((params_test[i], label))

        return (ret, pred)


