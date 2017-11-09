import random

from ceo.tools import manticore_policies, manticore_dist_symb_bytes
from ceo.tools import grr_mutators 

class Parameters(object):
    '''
        Abstract class to represent parameters for an action to execute
    ''' 
    pass

class ParametersExploreAFL(object):
    '''
        AFL parameter: none
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
        Grr parameters:
          * mutator
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
        Manticore parameters:
          * policy
          * dist_symb_bytes
          * rand_symb_bytes
    ''' 
    def __init__(self):
        self.policies = manticore_policies
        self.dist_symb_bytes = manticore_dist_symb_bytes

    def sample(self):
        pars = dict()
        pars["policy"] = random.choice(self.policies)
        pars["dist_symb_bytes"] = random.choice(self.dist_symb_bytes)
        pars["rand_symb_bytes"] = max(0.01, round(random.random(),1))
        return pars

    def enumerate(self):

        pars = []
        for x in self.policies:
            for y in self.dist_symb_bytes:
                for z in range(1,10):
                    pars.append([x, y, z/10.0])

        return pars

def init_parameter_generators():

    param_generators = dict()
    param_generators["afl"] = ParametersExploreAFL()
    param_generators["mcore"] = ParametersExploreMCore()
    param_generators["grr"] = ParametersExploreGrr()
 
    return param_generators


