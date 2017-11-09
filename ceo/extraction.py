from ceo.actions  import FeaturesMCore, FeaturesMCoreEVM

def get_features(tc, timeout, boot=False, verbose=1):

    if tc.target_platform == "evm":
        get_features = FeaturesMCoreEVM( tc.target_filepath, tc.input_filepath, tc.target_filename+"/evm")
    else:
        get_features = FeaturesMCore( tc.target_filepath, tc.input_filepath, tc.target_filename+"/mcore")
    if boot:
       filename = "boot.csv.gz" 
    else:
       filename = None

    return get_features.run(verbose=verbose, timeout=timeout, write=filename) 


