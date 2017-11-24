from os import makedirs
from multiprocessing import cpu_count
from numpy import array_split
from shutil import move, copy
from random import choice

from ceo.testcase import Testcase, load_targets, init_testcases, load_testcases
from ceo.reduce  import reduce_inputs, reduce_testcase
from ceo.extraction  import get_features
#from ceo.storage  import init_storage

def init(options, target_filename, nsamples, cpus, timeout, storage="ceo", verbose=1):

    #init_storage(storage)
    names, paths = load_targets(target_filename)
    print "[+] Initilization testcases"
    init_testcases(names, paths)
    print "[+] Collecting data to boot"
    print "[+] Available", len(paths), "to execute"

    for _ in range(nsamples):
        name, path =  choice(zip(names, paths))
        if "afl" in options:
            print "[+] Performing corpus minimization in", name
            reduce_inputs(name+"/inputs", path, verbose=verbose)
        testcases = load_testcases([name], [path])
        tc = choice(testcases)
        if "afl" in options:
            tc_min, label = reduce_testcase(tc, verbose=verbose)
        else:
            tc_min = tc

        if len(tc_min) == 0:
            continue
        print tc_min
        exec_features = get_features(tc_min, timeout, boot=True, verbose=verbose)

    if cpus is None: 
        cpus = max(1,cpu_count()-2)

    for i,batch in enumerate(array_split(zip(names, paths), cpus)):
         dirname = "fold-"+str(i)
         #init_storage(dirname + "/" + storage)
         target_file = dirname+"/" + target_filename

         try:
             makedirs(dirname)
         except OSError:
             pass

         try:
             f = open(target_file,"a+")
         except IOError:
             f = open(target_file,"w+")
         
         for name, path in batch:
            #print name, path, i
            f.write(path+"\n")
            move(name,dirname)
            copy("boot.csv.gz", dirname)
 
    print "[+] Done!"


