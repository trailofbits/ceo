import os
import random
import sys
import numpy as np
import pickle as pickle
from scipy import stats
from sklearn.externals import joblib
from sklearn.semi_supervised import label_propagation

from ceo.vectorizer import TFIDFVectorizer, CountVectorizer, Sent2VecVectorizer
from ceo.predictors import train_knn
from ceo.labels import lbls

class Policy(object):
    ''' Base class for prioritization '''
    pass

class TestcasePolicy(Policy):
    def __init__(self, init_vectorizers, init_generators, options, storage="ceo"):
        self.storage = str(storage)
        self.options = list(options)
        self.param_generators = None
        self.exec_features = dict()  # tid -> narray
        self.param_features = dict()  # str -> (tid,pid) -> narray
        self.labels = dict()    # str -> (tid, pid) -> int
        self.testcases = dict() # tid -> testcase

        for x in options:
            self.param_features[x] = dict()
            self.labels[x] = dict()

        self.exec_vectorizers = None
        self.param_vectorizers = None
        
        self.predictors = dict()
        for x in options:
            self.predictors[x] = label_propagation.LabelSpreading(kernel='knn', alpha=0.2)

        self._load()
        if (self.exec_vectorizers, self.param_vectorizers) == (None, None):
            print "[+] Re-creating vectorizers.."
            self.exec_vectorizers, self.param_vectorizers = init_vectorizers()

        if (self.param_generators == None):
            print "[+] Re-creating generators.."
            self.param_generators = init_generators() 

    def add_exec_features(self, tc, raw_exec_features):
        features = []
        for i,raw_feature in enumerate(raw_exec_features):
            # TODO: check if it is not array first
            #print i, raw_feature[0:5]
            row = self.exec_vectorizers[i].transform([raw_feature])
            features.append(row)
        #print map(lambda x: x.shape, features)
        features = np.concatenate(features, axis=1).flatten()
        self.exec_features[hash(tc)] = features
   
    def add_param_features(self, tc, raw_param_features):

        for option, param_features in raw_param_features:
            features = [self.param_vectorizers[option].transform([param_features])]
            features = np.concatenate(features, axis=1).flatten()
            param_features = tuple(param_features)
            self.param_features[option][(hash(tc),hash(param_features))] = features

    def add_labels(self, tc, raw_param_features, labels):

        labels = dict(labels)
        for option, param_features in raw_param_features:
            param_features = tuple(param_features)
            self.labels[option][(hash(tc), hash(param_features))] = labels[option]
    
    def __contains__(self, x):
        if self.testcases is None:
            return False
        return hash(x) in self.exec_features
    
    def get_exec_features(self, x):
        return self.exec_features.get(hash(x), None)

    #def get_label(self, option, x):
    #    return self.labels[option].get(hash(x), None)
    
   
    def add(self, tc, exec_features, param_features, labels):
        
        # testcases
        if hash(tc) not in self.testcases:
            self.testcases[hash(tc)] = tc
            self.add_exec_features(tc, exec_features)

        self.add_param_features(tc, param_features)
        self.add_labels(tc, param_features, labels)
        #print self.exec_features
        #print self.param_features
        self._save()

    def join_features(self, option, (tid, pid)):
        v = []
        v.append(self.param_features[option][tid,pid])
        v.append(self.exec_features[tid])
        return np.concatenate(v)
 
    def choice(self, raw_exec_features):
        ret = dict()

        for option in self.options:

            ret[option] = []
            X = []
            labels = []
            params = []

            for ((tid,pid),label) in self.labels[option].items():
                v = self.join_features(option, (tid,pid))
                X.append(v)
                #print v.shape
                labels.append(label)
                params.append(None)

            for raw_param_features in self.param_generators[option].enumerate():

                features = [self.param_vectorizers[option].transform([raw_param_features])]
                for i,raw_feature in enumerate(raw_exec_features):
                    row = self.exec_vectorizers[i].transform([raw_feature])
                    features.append(row)
                #print features
                features = np.concatenate(features, axis=1).flatten()
                #print features.shape 
                X.append(features)
                labels.append(lbls['?'])
                params.append(raw_param_features)
 
            #print labels
            self.predictors[option].fit(X,labels)
            pred_entropies = stats.distributions.entropy(
                                self.predictors[option].label_distributions_.T)
            results = zip(self.predictors[option].transduction_, pred_entropies)
             
            for i,(label,entropy) in enumerate(results):
                if params[i] is None:
                    continue
                ret[option].append((params[i], label, entropy))

        return ret

    def _save(self):

        for option in self.options:
            for ((tid,pid),x) in self.param_features[option].items():
                np.savez_compressed(self.storage+'/param_features/'+option+'.'+str(tid)+'.'+str(pid)+'.npz',x)

        for (tid,x) in self.exec_features.items():
                np.savez_compressed(self.storage+'/exec_features/'+str(tid)+'.npz',x)


        #np.savez_compressed("data/features.npz",X)
        joblib.dump(self.testcases, self.storage+'/testcases.pkl')
        joblib.dump(self.labels, self.storage+'/labels.pkl')
        joblib.dump(self.exec_vectorizers, self.storage+'/exec_vectorizers.pkl')
        joblib.dump(self.param_vectorizers, self.storage+'/param_vectorizers.pkl') 
        joblib.dump(self.param_generators, self.storage+'/param_generators.pkl')
    
    def _load(self):
        try:
            self.exec_vectorizers = joblib.load(open(self.storage+'/exec_vectorizers.pkl', 'rb'))
            self.param_vectorizers = joblib.load(open(self.storage+'/param_vectorizers.pkl', 'rb'))
            self.param_generators = joblib.load(open(self.storage+'/param_generators.pkl', 'rb'))

            self.testcases = joblib.load(open(self.storage+'/testcases.pkl', 'rb'))
            self.labels = joblib.load(open(self.storage+'/labels.pkl', 'rb'))

            for x, y, files in os.walk(self.storage+"/param_features"):
                for f in files:
                    feature_filepath = x + "/".join(y) + "/" + f
                    option = str(f.split(".")[0])
                    tid = int(f.split(".")[1])
                    pid = int(f.split(".")[2])
                    #print feature_filepath, x,y,f
                    self.param_features[option][(tid,pid)] = np.load(feature_filepath)['arr_0']

            for x, y, files in os.walk(self.storage+"/exec_features"):
                for f in files:
                    feature_filepath = x + "/".join(y) + "/" + f
                    tid = int(f.split(".")[0])
                    #print feature_filepath, x,y,f
                    self.exec_features[tid] = np.load(feature_filepath)['arr_0'] 

        except:
            return False 


class MultiPolicy(Policy):
    def __init__(self, options, storages):
        self.storages = list(storages)
        self.options = list(options)
        self.param_generators = None
        self.exec_features = dict()  # tid -> narray
        self.param_features = dict()  # str -> (tid,pid) -> narray
        self.labels = dict()    # str -> (tid, pid) -> int
        self.testcases = dict() # tid -> testcase

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

    def choice(self, raw_exec_features):
        ret = dict()

        for option in self.options:

            ret[option] = []
            X_train = []
            y_train = []
            X_test = []
            params_test = []

            for ((tid,pid),label) in self.labels[option].items():
                v = self.join_features(option, (tid,pid))
                X_train.append(v)
                y_train.append(label)
                #params.append(None)
            
            for raw_param_features in self.param_generators[option].enumerate():

                features = [self.param_vectorizers[option].transform([raw_param_features])]
                for i,raw_feature in enumerate(raw_exec_features):
                    row = self.exec_vectorizers[i].transform([raw_feature])
                    features.append(row)
                #print features
                features = np.concatenate(features, axis=1).flatten()
                #print features.shape 
                X_test.append(features)
                #.append(lbls['?'])
                params_test.append(raw_param_features)
           
            
            clf = train_knn(X_train, y_train)
            results = clf.predict(X_test)
             
            for i,label in enumerate(results):
                ret[option].append((params_test[i], label))

        return ret


