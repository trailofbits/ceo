#import sys
import os
import hashlib

from ceo.sampling import shuffle, stratified_shuffle
from ceo.naughtystrings import naughty_strings

def load_targets(filename):
    names = []
    paths = []

    for path in open(filename,"r").read().split("\n"):
        if path != '':
            paths.append(path)
            names.append(path.split("/")[-1])

    # shuffle
    x = list(shuffle(zip(names, paths)))
    return zip(*x)

def init_testcases(names, targets):
    for (name,target_filepath) in zip(names, targets):

        input_dir = name+"/inputs"
        walk = list(os.walk(input_dir))
        if walk == []:
            #print name
            os.makedirs(input_dir)
            for i,data in enumerate(naughty_strings()):
                input_filepath = input_dir+"/n" + hashlib.md5(data).hexdigest()
                open(input_filepath, "w+").write(data)  

def load_testcases(names, targets):
    testcases = []
    for (name,target_filepath) in zip(names, targets):

        try:
            os.makedirs(name+"/crashes")
        except OSError:
            pass
        input_dir = name+"/inputs"
        walk = list(os.walk(input_dir))
        for x, y, files in walk:
            for f in files:
                input_filepath = x + "/".join(y) + "/" + f
                testcases.append(Testcase(target_filepath, input_filepath))

    return testcases


class Testcase(object):
    '''
        ????
    ''' 

    def __init__(self, target_filepath, input_filepath):
        self.target_filepath = str(target_filepath)
        self.target_filename = self.target_filepath.split("/")[-1] 
        self.input_filepath = str(input_filepath)
        self.input_filename = self.input_filepath.split("/")[-1]
        self.input_data = open(self.input_filepath,"rb").read()
        self._find_platform()

    def _find_platform(self):
        magic = file(self.target_filepath).read(4)
        if magic == '\x7fELF':
            # Linux
            self.target_platform = "linux"
        elif magic == '\x7fCGC':
            # Decree
            self.target_platform = "decree"
        elif magic[:1] == '`':
            # EVM
            self.target_platform = "evm"
        else:
            raise NotImplementedError("Binary {} not supported.".format(self.target_filepath))

    def __len__(self):
        return len(self.input_data)

    def __hash__(self):
        return hash((self.target_filename, self.input_data))
 
    def __str__(self):
        return str((self.target_platform, self.target_filename, self.input_filename, self.input_data))

    def id(self):
        return hash(self)

