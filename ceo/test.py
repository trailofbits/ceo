import os

from random import choice

from ceo.policy import PredictivePolicy
from ceo.sampling import shuffle, stratified_shuffle
from ceo.testcase import Testcase, load_targets, load_testcases
from ceo.extraction  import get_features
from ceo.reduce  import reduce_inputs, reduce_testcase
from ceo.predictors import train_predictor #eval_rf, eval_svm, eval_knn
from ceo.plot import plot_data
from ceo.features import select_features
from ceo.vectorizer import init_vectorizers
from ceo.labels import lbls

def stats(options, features, target_filename, cpus, storage="ceo", verbose=0):

    # auto CPU selection
    if cpus is None:
        cpus = 1
 
    print "[+] Reading data"

    storages = []
    targetss = []
    prefixes = []

    walk = list(os.walk("."))
    for x, y, files in walk:
        if "fold" in x and x.endswith("/"+storage):
            storages.append(x)
        if "fold" in x and target_filename in files:
            prefixes.append(x)
            targetss.append(load_targets(x+"/"+target_filename, None))

    #features = ["syscalls","visited"]
    policy = PredictivePolicy(options, storages)
    data = policy.get_data(options,features)    

    print "[+] Fraction of labels collected:"
    for option, (progs, X, labels) in data.items():
        count = dict()
        print option
        for label,n in lbls.items():
            if n < 0:
                continue
            count[n] = labels.count(n)
            print label, count[n], count[n] / float(len(labels))

    #features = ["visited"]
    for option, (progs, X, labels) in data.items():
        plot_data(progs, option, X, labels, verbose=verbose) 
    for option, (progs, X, labels) in data.items():
        train_predictor(progs, option, X, labels, cpus, verbose=verbose)
    
def test(options, target_filename, cpus, extraction_timeout, storage="ceo", verbose=1):

    # auto CPU selection
    if cpus is None:
        cpus = 1
  
    storages = []
    targetss = [load_targets(target_filename)]
    prefixes = ["."]

    walk = list(os.walk("."))
    for x, y, files in walk:
        if "fold" in x and x.endswith("/"+storage):
            storages.append(x)


    policy = MultiPolicy(options, storages)
    for prefix,targets in zip(prefixes, targetss):
        names, paths = targets
        for name, path in zip(names, paths):
            testcases = load_testcases([prefix+"/"+name], [path])
            tc = choice(testcases)
            try:
                os.makedirs(name+"/inputs")
            except OSError:
                pass

            if "afl" in options:
                tc_min, label = reduce_testcase(tc, "min", verbose=verbose)
            else:
                print "[+] AFL disabled, skipping input minimization" 
                tc_min = tc

            if tc_min in policy:
                print "[+] Test case already analyzed!"
                print tc_min
                continue

            print "[+] Predicting best action for test case:"
            print tc_min

            print "[+] Extracting features"
            exec_features = get_features(tc_min, extraction_timeout, verbose=verbose)
            #print exec_features
            print "[+] Finding best predictor"
            res, preds = policy.choice(exec_features, cpus)
            for option in options:
                print "[+] For",option,", the best predictor is:"
                print preds[option]
                print "[+] Possible outcomes are:"
                for param, label in res[option]:
                    print param, u'\u2192' , label
