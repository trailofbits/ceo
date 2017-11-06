from ceo.actions  import ExploreGrr, ExploreAFL, ExploreMCore

def explore(options, tc, verbose=1):
    print "[+] Exploring testcase ... "
    labels = []
    param_features = []
    input_dir = tc.target_filename+"/inputs"
    crash_dir = tc.target_filename+"/crashes"

    if "grr" in options:
        explore = ExploreGrr( tc.target_filepath, tc.input_filepath, None, tc.target_filename+"/grr")
        param_features.append(("grr", explore.get_features()))
        label = explore.run(verbose=verbose, timeout=600)
        explore.save_results(input_dir, crash_dir)
        labels.append(("grr", label))

    if "afl" in options:
        explore = ExploreAFL( tc.target_filepath, tc.input_filepath, None, tc.target_filename+"/afl")
        param_features.append(("afl", explore.get_features()))
        label = explore.run(verbose=verbose, timeout=600)
        labels.append(("afl", label))
        explore.save_results(input_dir, crash_dir)

    if "mcore" in options:
        explore = ExploreMCore( tc.target_filepath, tc.input_filepath, None, tc.target_filename+"/mcore")
        param_features.append(("mcore", explore.get_features()))
        label = explore.run(verbose=verbose, timeout=600)
        labels.append(("mcore", label))
        explore.save_results(input_dir, crash_dir)
   
    return labels, param_features 


