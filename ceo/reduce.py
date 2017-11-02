from ceo.actions  import MinimizeTestcaseAFL, MinimizeInputsAFL
from ceo.labels   import lbls
from ceo.testcase import Testcase

def reduce_inputs(inputs, target_filepath):
    minimize = MinimizeInputsAFL(target_filepath, inputs) 
    ret = minimize.run()

def reduce_testcase(tc, output=None):
    print "[+] Performing input minimization"
    label = lbls['?']
    if output is None:
        output = tc.input_filepath+".min"

    if ".min" in tc.input_filename:
        tc_min = tc
    else:
        minimize = MinimizeTestcaseAFL( tc.target_filepath, tc.input_filepath, output) 
        ret = minimize.run()
                 
        if ret == None:
            tc_min = tc
        else:
            label = ret
            tc_min = Testcase( tc.target_filepath, output)

            #if len(tc_min) == 0:
            #    tc_min = tc

    return tc_min, label


