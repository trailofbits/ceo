import os
import os.path
import random
import shutil
import subprocess
import hashlib

from manticore import Manticore
from ceo.tools import grrshot_path, grrplay_path, grr_mutators 
from ceo.tools import afltmin_path, aflcmin_path, aflfuzz_path
from ceo.labels import lbls
from ceo.aflcmin import cmin
from ceo.features import ExecFeatures, ParamFeatures
from ceo.plugins import FeatureCollector, FeatureExtractor, StateCollector, Abandonware
from ceo.concrete_execution import make_initial_concrete_state 
from ceo.symbolic_execution import make_initial_symbolic_state, reconstruct_concrete_file 

#from ceo.evm import ManticoreEVM

from ceo.parameters import ParametersExploreAFL, ParametersExploreMCore, ParametersExploreGrr

class Action(object):
    '''
        Abstract class to represent an action to execute
    ''' 
    pass

class FeaturesMCore(Action):
    '''
        Use Manticore to extract execution features from a particular test case
    ''' 

    def __init__(self, target_path, input_path, workspace):
        self.target_path = str(target_path)
        self.input_path = str(input_path)
        self.workspace = str(workspace)

    def run(self, procs=1, timeout=60, verbose=1, write=None):

        shutil.rmtree(self.workspace, True)

        concrete_data = list(open(self.input_path,"r").read())
        initial_concrete_state = make_initial_concrete_state(self.target_path,
                                                             concrete_data)
                                                           
        m = Manticore(initial_concrete_state, workspace_url=self.workspace, policy="random")
        m.verbosity(verbose) 
        m.plugins = set()
        features = ExecFeatures()
        m.register_plugin(FeatureCollector(features, 0.05))
        m.register_plugin(FeatureExtractor())
        m.register_plugin(Abandonware())
        
        try:
            m.run(procs=procs, timeout=timeout)
        except AssertionError:
            pass

        #if write != None:
        #    features.write(write)
        return features#.get()

class FeaturesMCoreEVM(Action):
    '''
        Use Manticore to extract execution features from a particular smart contract
    ''' 

    def __init__(self, target_path, input_path, workspace):
        self.target_path = str(target_path)
        self.input_path = str(input_path)
        self.workspace = str(workspace)

    def run(self, procs=1, timeout=60, verbose=1, write=None):

        shutil.rmtree(self.workspace, True)

        bytecode = list(open(self.target_path,"r").read())
        concrete_data = list(open(self.input_path,"r").read())

        m = ManticoreEVM(workspace_url=self.workspace, policy="random")
        m.verbosity(verbose) 
        user_account = m.create_account(balance=1024)
        contract_account = m.create_contract(owner=user_account,
                                             balance=0,
                                             init=bytecode)


        def explore(state,pc, instruction):
            print pc, instruction

        m.register_plugin(FeatureCollector())
        m.transaction(  caller=user_account,
                           address=contract_account,
                           value=None,
                           data=concrete_data,
                         )



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
        features = ParamFeatures()
        features.add_params(self.extra_args)
        return features

    #def _parse_txt(self, f):
    #    return open(f, "r").read().split("_1:")[1].replace("'","").decode('string_escape')

    def _check_output(self, m):
        last_state_id = m._executor._workspace._last_id.value
        #print "last_state_id", last_state_id
        files = m._executor._workspace._store.ls("*.messages")
        uri = m._executor._workspace._store.uri
         
        for f in files:
            txt_filename = uri + '/' + f.replace("messages","txt") 
            stdin_filename = uri + '/' + f.replace("messages","stdin")

            if not os.path.isfile(stdin_filename):
                #if i
                data = reconstruct_concrete_file(txt_filename, self.input_path)
                open(stdin_filename,"w+").write(data)

            if "Invalid memory access" in open(uri + "/" + f,"rb").read():
                self.crashes.append(stdin_filename)
            else: 
                self.inputs.append(stdin_filename)

        if len(self.crashes) > 0:
            return lbls['found']

        if len(self.inputs) > 1:
            return lbls['new']

        #if  len(self.inputs) == 1:
        #    return lbls['nothing']

        return lbls['fail']

    def save_results(self, input_dir, crash_dir):
        #print self.inputs
        #print self.crashes
        copy_and_rename(".", self.inputs, input_dir)
        copy_and_rename(".", self.crashes, crash_dir)

    def run(self, procs=1, timeout=600, verbose=0):

        shutil.rmtree(self.workspace, True)
        concrete_data = list(open(self.input_path,"r").read()) 
        initial_symbolic_state = make_initial_symbolic_state(self.target_path,
                                                             concrete_data,
                                                             self.extra_args["dist_symb_bytes"],
                                                             self.extra_args["rand_symb_bytes"])
 
        m = Manticore(initial_symbolic_state, workspace_url=self.workspace, policy=self.extra_args["policy"])  
        #m.plugins = set() 
        Manticore.verbosity(verbose)
        m.register_plugin(StateCollector())
        m.register_plugin(Abandonware()) 

        try:
            m.run(procs=procs, timeout=timeout)
        except AssertionError:
            return lbls['fail']
        
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

        ret = cmin(self.target_path, self.inputs_path, self.outputs_path, verbose=verbose)
        if ret:
            shutil.rmtree(self.inputs_path, True) 
            shutil.move(self.outputs_path, self.inputs_path) 
        return None

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
        if verbose > 0:
            print "---"
            print output
            print "---"
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
    def __init__(self, target_path, input_path, extra_args, workspace):
 
        self.target_path = str(target_path)
        self.input_path = str(input_path)

        self.workspace = str(workspace)# + "/afl"
        self.input_dir = self.workspace + "/in" #+ str(input_path)
        self.output_dir = self.workspace + "/out" 
        shutil.rmtree(self.workspace, True) 

        self.param_generator = ParametersExploreAFL()
        if extra_args is None:
            self.extra_args = self.param_generator.sample() 
 

        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        shutil.copy(self.input_path, self.input_dir)
        self.inputs = []
        self.crashes = []

    def get_features(self):
        features = ParamFeatures()
        features.add_params(self.extra_args)
        return features

    def save_results(self, input_dir, crash_dir):
        #print self.inputs
        #print self.crashes
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

        #if  len(self.inputs) == 1:
        #    return lbls['nothing']

        return lbls['fail']

    def run(self, procs=8, timeout=600, verbose=0):
        exe = "timeout"
        args = [str(timeout), aflfuzz_path, "-i", self.input_dir, "-o", self.output_dir, "-Q",
                "-m", "none", "-t", "5000", "--", self.target_path ]
        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args,
                                    stdout=devnull,
                                    stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode

        if verbose > 0:
            print "---"
            print output
            print "---"
 
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
        features = ParamFeatures()
        features.add_params(self.extra_args)
        return features

    def save_results(self, input_dir, crash_dir):
        #print self.inputs
        #print self.crashes
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

        #if  len(self.inputs) == 1:
        #    return lbls['nothing']

        return lbls['fail']

    def run(self, procs=1, timeout=600, verbose=0):

        exe = grrshot_path
        args = ["--num_exe=1", "--exe_dir="+self.workspace, "--exe_prefix=",
                "--snapshot_dir="+self.workspace]

        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args,
                                    stdout=devnull,
                                    stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode

            if verbose > 0:
                print "ret", ret

        if verbose > 0:
            print "---"
            print output
            print "---"
 
        exe = grrplay_path
        args = ["--num_exe=1", "--snapshot_dir="+self.workspace, "--nopersist",
                "--output_snapshot_dir="+self.workspace]

        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args,
                                    stdout=devnull,
                                    stderr=subprocess.PIPE)
            output = proc.communicate()[1]
            ret = proc.returncode
            if verbose > 0:
                print "ret", ret

        if verbose > 0:
            print "---"
            print output
            print "---"
 
        exe = "timeout"
        args = ["-k", "1", str(timeout), grrplay_path , "--num_exe=1", "--persist", "--persist_dir="+self.workspace,
                "--snapshot_dir="+self.workspace,  "--input="+self.input_path, "-print_num_mutations", "--output_dir="+self.output_dir,
                "--path_coverage", "--remutate", "--input_mutator="+self.extra_args["mutator"]]

        if verbose > 0:
            print "Executing", exe, args

        with open(os.devnull, 'wb') as devnull:
            proc = subprocess.Popen([exe]+args,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            output = proc.communicate()
            ret = proc.returncode

            if verbose > 0:
                print "ret", ret

        if verbose > 0:
            print "---"
            print output
            print "---"

        return self._check_output() 
