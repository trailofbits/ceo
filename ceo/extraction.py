from ceo.actions  import FeaturesMCore

def get_features(tc, boot=False, verbose=1):

    get_features = FeaturesMCore( tc.target_filepath, tc.input_filepath, tc.target_filename+"/mcore")
    if boot:
       filename = "boot.csv.gz" 
    else:
       filename = None

    return get_features.run(verbose=verbose, timeout=300, write=filename) 


