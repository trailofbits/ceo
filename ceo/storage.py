from os import makedirs

from sklearn.externals import joblib

class Storage(object):
    def __init__(self, path, features_list, options):
        self.storage = str(path) # TODO: rename to path?
        self.options = list(options)
        self.features_list = list(features_list)
        self.testcases = dict() # tid -> testcase
        self.labels = dict()    # str -> (tid, pid) -> int
 
        for x in options:
            self.labels[x] = dict()

        self._init_storage()

    def __contains__(self, x):
        if self.testcases is None:
            return False
        return hash(x) in self.testcases   

    def _init_storage(self):
        try:
            makedirs(self.storage+"/exec_features")
            makedirs(self.storage+"/param_features")
        except OSError:
            pass

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
