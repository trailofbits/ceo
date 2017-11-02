from os import makedirs

def init_storage(dirname):
    try:
        makedirs(dirname+"/exec_features")
        makedirs(dirname+"/param_features")
    except OSError:
        pass


