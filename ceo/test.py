import os

from random import choice

from ceo.policy import MultiPolicy
from ceo.sampling import shuffle, stratified_shuffle
from ceo.testcase import Testcase, load_targets, load_testcases
from ceo.extraction  import get_features
from ceo.reduce  import reduce_inputs, reduce_testcase
from ceo.predictors import eval_rf, eval_svm, eval_knn

def stats(options, target_filename, storage_dir):
 
    storages = []
    targetss = []
    prefixes = []

    walk = list(os.walk("."))
    for x, y, files in walk:
        if "fold" in x and x.endswith("/"+storage_dir):
            storages.append(x)
        if "fold" in x and target_filename in files:
            prefixes.append(x)
            targetss.append(load_targets(x+"/"+target_filename))

    #print targets
    #options = ["afl", "mcore", "grr"] 
    policy = MultiPolicy(options, storages)
    data = policy.get_data()

    for option, (progs, X, labels) in data.items():
        count = dict()
        print option
        for i in range(4):
            count[i] = labels.count(i)
            print i, count[i], count[i] / float(len(labels))

        mX = []
        mlabels = []
        mprogs = []


        for prog, x, label in zip(progs, X, labels):
            if count[label] <= 10:
                 continue

            mX.append(x)
            mlabels.append("c"+str(label))
            mprogs.append(prog)

        score, report =  eval_rf(mprogs, mX, mlabels)
        print "eval rf:", score
        #print report

        score, report =  eval_svc(mprogs, mX, mlabels)    
        print "eval svc:", score
        #print report

        score, report = eval_knn(mprogs, mX, mlabels)
        print "eval knn:", score
        #print report

def test(options, target_filename, storage_dir, verbose=1):
 
    storages = []
    targetss = [load_targets(target_filename)]
    prefixes = ["."]

    walk = list(os.walk("."))
    for x, y, files in walk:
        if "fold" in x and x.endswith("/"+storage_dir):
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
            exec_features = get_features(tc_min, verbose=verbose)
            #print exec_features
            print "[+] Finding best predictor"
            res, preds = policy.choice(exec_features)
            for option in options:
                print "[+] For",option,", the best predictor is:"
                print preds[option]
                print "[+] Possible outcomes are:"
                for param, label in res[option]:
                    print param, u'\u2192' , label
