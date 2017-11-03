import os
import os.path
import random
import shutil
import subprocess
import hashlib

from manticore import Manticore
#from manticore.features.features import ExecFeatures
from ceo.tools import manticore_policies, manticore_dist_symb_bytes
from ceo.tools import grrshot_path, grrplay_path, grr_mutators 
from ceo.tools import afltmin_path, aflcmin_path, aflfuzz_path
from ceo.labels import lbls
from ceo.aflcmin import cmin
from ceo.features import ExecFeatures

class Action(object):
    '''
        Abstract class to represent an action to execute
    ''' 
    pass

class Parameters(object):
    '''
        Abstract class to represent parameters for an action to execute
    ''' 
    pass


class FeaturesMCore(Action):
    '''
        Use Manticore to extract execution features from a particular testcase
    ''' 

    def __init__(self, target_path, input_path, workspace):
        self.target_path = str(target_path)
        self.input_path = str(input_path)
        self.workspace = str(workspace)

    def run(self, procs=1, timeout=60, verbose=0, write=None):

        shutil.rmtree(self.workspace, True) 
        m = Manticore(self.target_path, workspace_url=self.workspace,  policy="random",  
                  concrete_data=self.input_path, extract_features=False)

        Manticore.verbosity(verbose)
        features = ExecFeatures()

        @m.hook(None)
        def explore(state): 
            if random.random() <= 0.05:
                features.add_insns(state)
            features.add_syscalls_seq(state)
        m.run(procs=procs, timeout=timeout)

        if write != None:
            features.write(write)
  
        return features.get()[1:]


class ParametersExploreAFL(object):
    '''
        Abstract class to represent an action to execute
    ''' 
    def __init__(self):
        pass

    def sample(self):
        pars = dict()
        return pars

    def enumerate(self):
        return [[]]

class ParametersExploreGrr(object):
    '''
        Abstract class to represent an action to execute
    ''' 
    def __init__(self):
        self.mutators = grr_mutators
    def sample(self):
        pars = dict()
        pars["mutator"] = random.choice(self.mutators)
        return pars

    def enumerate(self):
        pars = []
        for x in self.mutators:
            pars.append([x])
        return pars




class ParametersExploreMCore(object):
    '''
        Abstract class to represent an action to execute
    ''' 
    def __init__(self):
        self.policies = manticore_policies
        self.dist_symb_bytes = manticore_dist_symb_bytes

    def sample(self):
        pars = dict()
        pars["policy"] = random.choice(self.policies)
        pars["dist_symb_bytes"] = random.choice(self.dist_symb_bytes)
        pars["rand_symb_bytes"] = round(random.random(),1)
        return pars

    def enumerate(self):

        pars = []
        for x in self.policies:
            for y in self.dist_symb_bytes:
                for z in range(1,10):
                    pars.append([x, y, z/10.0])

        return pars


def init_generators():

    param_generators = dict()
    param_generators["afl"] = ParametersExploreAFL()
    param_generators["mcore"] = ParametersExploreMCore()
    param_generators["grr"] = ParametersExploreGrr()
 
    return param_generators


def copy_and_rename(base, files, dirname):
    #print base, files, dirname
    for f in files:
        src_filename = base + '/' + f
        name = hashlib.md5(open(src_filename,"rb").read()).hexdigest()
        dst_filename = dirname + '/' + name
        shutil.copy(src_filename, dst_filename)
    

class ExploreMCore(Action):
    '''
        Use Manticore to explore a particular testcase
    ''' 

    def __init__(self, target_path, input_path, extra_args, workspace):
        self.target_path = str(target_path)
        self.input_path = str(input_path)
        self.workspace = str(workspace)
        self.param_generator = ParametersExploreMCore()
        if extra_args is None:
            self.extra_args = self.param_generator.sample() 

        self.inputs = []
        self.crashes = []


    def get_features(self):
        features = []
        features.append(self.extra_args["policy"])
        features.append(self.extra_args["dist_symb_bytes"])
        features.append(self.extra_args["rand_symb_bytes"])
        return features

    def _check_output(self, m):
        last_state_id = m._executor._workspace._last_id.value
        print "last_state_id", last_state_id
        files = m._executor._workspace._store.ls("*.messages")
        uri = m._executor._workspace._store.uri
         
        for f in files:
            filename = uri + '/' + f.replace("messages","stdin")
            if not os.path.isfile(filename):
                continue
            if "Invalid memory access" in open(uri + "/" + f,"rb").read():
                self.crashes.append(filename)
            else: 
                self.inputs.append(filename)

        if len(self.crashes) > 0:
            return lbls['found']

        if  len(self.inputs) > 1:
            return lbls['new']

        if  len(self.inputs) == 1:
            return lbls['nothing']

        return lbls['fail']

    def save_results(self, input_dir, crash_dir):
        print self.inputs
        print self.crashes
        copy_and_rename(".", self.inputs, input_dir)
        copy_and_rename(".", self.crashes, crash_dir)

    def run(self, procs=1, timeout=600, verbose=0):

        shutil.rmtree(self.workspace, True) 
        m = Manticore(self.target_path, workspace_url=self.workspace,  #policy=self.policy,  
                      #rand_symb_bytes=self.rand_symb_bytes, dist_symb_bytes=self.dist_symb_bytes,
                      concrete_data=self.input_path, 
                      extract_features=False, **self.extra_args)
        
        Manticore.verbosity(verbose)
        m.run(procs=procs, timeout=timeout)
        return self._check_output(m)


class MinimizeInputsAFL(Action):
    '''
        Use AFL to minimize a directory of inputs
    '''

    def __init__(self, target_path, inputs_path):
        self.target_path = str(target_path)
        self.inputs_path = str(inputs_path)
        self.outputs_path = ".tmp"
        shutil.rmtree(self.outputs_path, True) 

    def run(self, procs=1, timeout=600, verbose=0):

        ret = cmin(self.target_path, self.inputs_path, self.outputs_path)
        if ret:
            shutil.rmtree(self.inputs_path, True) 
            shutil.move(self.outputs_path, self.inputs_path) 
        return None#self._parse_output(output)
        #return (ret == 0) 

class MinimizeTestcaseAFL(Action):
    '''
        Use AFL to minimize a testcase
    '''

    def __init__(self, target_path, input_path, output_path):
        self.target_path = str(target_path)
        self.input_path = str(input_path)
        self.output_path = str(output_path)

    def _parse_output(self, output):
        #print output
        for x in output.split("\x1b"):
            #print repr(x)
            if "No instrumentation detected" in x:
                return None
            if "Target binary times out" in x:
                return None
            if "Program exits with a signal" in x:
                return lbls['found']
            if "crash" in x:
                #print repr(x)
                for y in x.split(" "):
                    if "crash" in y: 
                        if  y == "crash=0":
                            return lbls['?']
                        else:
                            print y
                            return lbls['found']
        
        return lbls['?']

    def run(self, procs=8, timeout=600, verbose=0):
        exe = afltmin_path
        args = ["-i", self.input_path, "-o", self.output_path, "-Q",
                "-m", "none", "-t", "5000", "--", self.target_path ]
        if verbose > 0:
            print "Executing", exe, args
        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args,
                                    stdout=devnull,
                                    stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode
        #print output
        return self._parse_output(output)
        #return (ret == 0) 

class AnalyzeTestcaseAFL(Action):
    '''
        Use AFL to analyze a testcase
    ''' 
    pass

class ExploreAFL(Action):
    '''
        Use AFL to look for more testcases
    ''' 
    def __init__(self, target_path, input_path, workspace):
 
        self.target_path = str(target_path)
        self.input_path = str(input_path)

        self.workspace = str(workspace)# + "/afl"
        self.input_dir = self.workspace + "/in" #+ str(input_path)
        self.output_dir = self.workspace + "/out" 
        shutil.rmtree(self.workspace, True) 
 
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        shutil.copy(self.input_path, self.input_dir)
        self.inputs = []
        self.crashes = []

    def get_features(self):
        features = []
        return features

    def save_results(self, input_dir, crash_dir):
        print self.inputs
        print self.crashes
        copy_and_rename(".", self.inputs, input_dir)
        copy_and_rename(".", self.crashes, crash_dir)

    def _check_output(self, output):
        if "No instrumentation detected" in output:
            return lbls['fail']

        crashes = list(os.walk(self.output_dir+"/crashes"))[0][2]
        
        for f in crashes:
            if "README" not in f:
                self.crashes.append(self.output_dir+"/crashes/"+f) 

        inputs = list(os.walk(self.output_dir+"/queue"))[0][2]
        for f in inputs:
            self.inputs.append(self.output_dir+"/queue/"+f) 
 
        if len(self.crashes) > 0:
            return lbls['found']

        if  len(self.inputs) > 1:
            return lbls['new']

        if  len(self.inputs) == 1:
            return lbls['nothing']

        return lbls['fail']

    def run(self, procs=8, timeout=600, verbose=0):
        exe = "timeout"
        args = [str(timeout), aflfuzz_path, "-i", self.input_dir, "-o", self.output_dir, "-Q",
                "-m", "none", "-t", "5000", "--", self.target_path ]
        print "Executing", exe, args
        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args,
                                    stdout=devnull,
                                    stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode
        #print output
        return self._check_output(output)
        #return lbls['?'] #(ret == 0) 



class ExploreGrr(Action):
    '''
        Use Grr to look for more crashes
    ''' 
    def __init__(self, target_path, input_path, extra_args, workspace):
 
        self.target_path = str(target_path)
        self.input_path = str(input_path)

        self.workspace = str(workspace)# + "/afl"
        #self.input_dir = self.workspace + "/in" #+ str(input_path)
        self.output_dir = self.workspace + "/out" 
        self.inputs = []
        self.crashes = []

        self.param_generator = ParametersExploreGrr()
        if extra_args is None:
            self.extra_args = self.param_generator.sample() 
 
        shutil.rmtree(self.workspace, True) 
 
        #os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        #os.makedirs(self.persist_dir)

        shutil.copy(self.target_path, self.workspace+"/0001")

    def get_features(self):
        features = []
        features.append(self.extra_args["mutator"])
        return features

    def save_results(self, input_dir, crash_dir):
        print self.inputs
        print self.crashes
        copy_and_rename(".", self.inputs, input_dir)
        copy_and_rename(".", self.crashes, crash_dir)

    def _check_output(self):

        outdir = list(os.walk(self.output_dir))[0][2]
 
        for f in outdir:
            if "crash" in f:
                self.crashes.append(self.output_dir+'/'+f)
            else:
                self.inputs.append(self.output_dir+'/'+f) 

        if len(self.crashes) > 0:
            return lbls['found']

        if  len(self.inputs) > 1:
            return lbls['new']

        if  len(self.inputs) == 1:
            return lbls['nothing']

        return lbls['fail']

    def run(self, procs=1, timeout=600, verbose=0):

        exe = grrshot_path
        args = ["--num_exe=1", "--exe_dir="+self.workspace, "--exe_prefix=",
                "--snapshot_dir="+self.workspace]

        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args)
                                    #stdout=devnull,
                                    #stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode

            if verbose > 0:
                print "ret", ret

        exe = grrplay_path #"grrplay"
        args = ["--num_exe=1", "--snapshot_dir="+self.workspace, "--nopersist",
                "--output_snapshot_dir="+self.workspace]

        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args)
                                    #stdout=devnull,
                                    #stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode
            if verbose > 0:
                print "ret", ret

        exe = "timeout"
        args = ["-k", "1", str(timeout), grrplay_path , "--num_exe=1", "--persist", "--persist_dir="+self.workspace,
                "--snapshot_dir="+self.workspace,  "--input="+self.input_path, "-print_num_mutations", "--output_dir="+self.output_dir,
                "--path_coverage", "--remutate", "--input_mutator="+self.extra_args["mutator"]]
        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args)
                                    #stdout=devnull,
                                    #stderr=devnull)
            output = proc.communicate()
            ret = proc.returncode

            if verbose > 0:
                print "ret", ret

        return self._check_output() 
