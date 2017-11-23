from ceo.actions  import ExploreGrr, ExploreAFL, ExploreMCore

def explore(options, tc, timeout, verbose=1):
    print "[+] Exploring testcase ... "
    labels = dict()
    param_features = dict()
    input_dir = tc.target_filename+"/inputs"
    crash_dir = tc.target_filename+"/crashes"


    if "mcore" in options:
        explore = ExploreMCore( tc.target_filepath, tc.input_filepath, None, tc.target_filename+"/mcore")
        param_features["mcore"] = explore.get_features()
        label = explore.run(verbose=verbose, timeout=timeout)
        labels["mcore"] = label
        explore.save_results(input_dir, crash_dir)
 
    if "grr" in options:
        explore = ExploreGrr( tc.target_filepath, tc.input_filepath, None, tc.target_filename+"/grr")
        param_features["grr"] = explore.get_features()
        label = explore.run(verbose=verbose, timeout=timeout)
        explore.save_results(input_dir, crash_dir)
        labels["grr"] = label

    if "afl" in options:
        explore = ExploreAFL( tc.target_filepath, tc.input_filepath, None, tc.target_filename+"/afl")
        param_features["afl"] = explore.get_features()
        label = explore.run(verbose=verbose, timeout=timeout)
        labels["afl"] = label
        explore.save_results(input_dir, crash_dir)
    
  
    return labels, param_features 


