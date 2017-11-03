from os import makedirs
from multiprocessing import cpu_count
from numpy import array_split
from shutil import move, copy
from random import choice

from ceo.testcase import Testcase, load_targets, init_testcases, load_testcases
from ceo.reduce  import reduce_inputs, reduce_testcase
from ceo.extraction  import get_features
from ceo.storage  import init_storage

def init(options, target_filename, storage_dir):

    init_storage(storage_dir)
    names, paths = load_targets(target_filename)
    print "[+] Initilization testcases"
    init_testcases(names, paths)
    print "[+] Collecting data to boot"
    print "[+] Available", len(paths), "to execute"
    for name, path in zip(names, paths):
        print "[+] Performing corpus minimization in", name
        if "afl" in options:
            reduce_inputs(name+"/inputs", path)
        testcases = load_testcases([name], [path])
        tc = choice(testcases)
        if "afl" in options:
            tc_min, label = reduce_testcase(tc)
        else:
            tc_min = tc

        if len(tc_min) == 0:
            continue

        exec_features = get_features(tc_min, boot=True)
    
    cpus = max(2,cpu_count()-2)
    for i,batch in enumerate(array_split(zip(names, paths), cpus)):
         dirname = "fold-"+str(i)
         init_storage(dirname + "/" + storage_dir)
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

