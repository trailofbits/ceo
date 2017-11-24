import os
import random
import sys
import numpy as np
import pickle as pickle
from scipy import stats
from sklearn.externals import joblib
from sklearn.semi_supervised import label_propagation

from ceo.features  import ExecFeatures,ParamFeatures
from ceo.vectorizer import TFIDFVectorizer, CountVectorizer, Sent2VecVectorizer
from ceo.predictors import train_predictor
from ceo.labels import lbls

class Policy(object):
    ''' Base class for prioritization '''
    pass

class PredictivePolicy(Policy):
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
        #for x in options:
        #    self.predictors[x] = label_propagation.LabelSpreading(kernel='knn', alpha=0.2)

        self._load()

    def _load(self):
        self.storage = self.storages[0]
        
        #self.exec_vectorizers = joblib.load(open(self.storage+'/exec_vectorizers.pkl', 'rb'))
        #self.features_list = self.exec_vectorizers.keys()
        #self.param_vectorizers = joblib.load(open(self.storage+'/param_vectorizers.pkl', 'rb'))
        #self.param_generators = joblib.load(open(self.storage+'/param_generators.pkl', 'rb'))

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

                    features = ParamFeatures()
                    features.load(feature_filepath)
                    self.param_features[option][(tid,pid)] = features.get() 
                    #print self.param_features[option][(tid,pid)] 

            for x, y, files in os.walk(storage+"/exec_features"):
                for f in files:
                    feature_filepath = x + "/".join(y) + "/" + f
                    tid = int(f.split(".")[0])

                    features = ExecFeatures()
                    features.load(feature_filepath)
                    self.exec_features[tid] = features.get() 

    def __contains__(self, x):
        if self.exec_features is None:
            return False
        return hash(x) in self.exec_features
 
    def join_features(self, option, (tid, pid)):
        x = dict()
        x["param_features"] = self.param_features[option][tid,pid]
        x["exec_features"] = self.exec_features[tid]
        return x

    def get_data(self):
        ret = dict()

        for option in self.options:

            X = []
            labels = []
            progs = []

            for ((tid,pid),label) in self.labels[option].items():
                x = self.join_features(option, (tid,pid))
                X.append(x)
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


