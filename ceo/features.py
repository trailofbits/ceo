import csv
import gzip

features_list = ["syscalls", "insns", "reads", "writes", "allocs", "deallocs"]

def analyze_insn(insn):
    if insn is None:
        return "none"
    r = insn.mnemonic
    if len(insn.operands) > 0:
        r = r + "("
        for op in insn.operands:
            r = r + op.type[:3] + ","
        r = r[:-1] + ")"
    return r

def mkfeatures_insns(features, state, max_size=128):
    insns = state.context["last_insn"] #cpu._last_insns

    for insn in insns:
        features.append(analyze_insn(insn))

    return features[-max_size:]

def analyze_syscall(syscall):
    if syscall is None:
        return "none"
    return syscall[0]

def mkfeatures_syscalls(state, max_size=128):
    features = map(analyze_syscall, list(state.platform.syscall_trace))
    return features[-max_size:]

def analyze_allocations(syscall):
    name = syscall[0]
    if name == "_allocate":
        return syscall[2]
    return 0

def analyze_deallocations(syscall):
    name = syscall[0]
    if name == "_deallocate":
        return syscall[2]
    return 0

def mkfeatures_alloc_dealloc(state):
    alloc_size = sum(map(analyze_allocations, list(state.platform.syscall_trace)))
    dealloc_size = sum(map(analyze_deallocations, list(state.platform.syscall_trace)))
   
    if alloc_size == 0:
        ratio = float(0)
    else:
        ratio = round(float(dealloc_size) / float(alloc_size),2)

    return (alloc_size, dealloc_size, ratio)


def analyze_reads(syscall):
    name = syscall[0]
    if name == "_receive":
        return len(syscall[2])
    return 0

def analyze_writes(syscall):
    name = syscall[0]
    if name == "_transmit":
        return len(syscall[2])
    return 0

def mkfeatures_reads_writes(state):
    read_size = sum(map(analyze_reads, list(state.platform.syscall_trace)))
    write_size = sum(map(analyze_writes, list(state.platform.syscall_trace)))
   
    if read_size == 0:
        ratio = float(0)
    else:
        ratio = round(float(write_size) / float(read_size),2)

    return (read_size, write_size, ratio)

def mkfeatures_sols(solutions):
    return len(solutions)

def mkfeatures_sols_time(time_taken):
    return round(time_taken,1)

def mkfeatures_id(program, state_id):
    return hash((program,state_id))

def mkfeatures_depth(depth):
    return depth + 1

def find_program(state):
    return state.platform._path

class Features(object):
    '''
        Abstract class for feature extraction
    ''' 

class ExecFeatures(Features):
    '''
        Feature extraction from executions
    ''' 

    def __init__(self):
        self._program = None
        self._features = {}

    def add_insns(self, state):
        if self._program is None:
            self._program = find_program(state)

        features = self._features
        insns = features.get("insns",[])
        features["insns"] = insns + ["."] + mkfeatures_insns([], state)

    def add_syscalls_seq(self, state):

        features = self._features
        current_syscalls = mkfeatures_syscalls(state)
        previous_syscalls = features.get("syscalls",[])
        if len(previous_syscalls) <= len(current_syscalls):
            features["syscalls"] = current_syscalls
        else:
            features["syscalls"] = previous_syscalls + current_syscalls
 
    def add_syscalls_stats(self, state):

        features = self._features 
        rw = features.get("read-write",[(0,0,0)]) 
        x = mkfeatures_reads_writes(state)
        if x != rw[-1]:
            features["read-write"] = rw + [x]

        ad = features.get("alloc-dealloc",[(0,0,0)])
        x = mkfeatures_alloc_dealloc(state)
        if x != ad[-1]:
            features["alloc-dealloc"] = ad + [x]

    def get(self):
        ret = dict()

        ret["program"] = [self._program]
        features = self._features
        ret["insns"] = " ".join(features.get("insns",[])) 
        ret["syscalls"] = " ".join(features.get("syscalls",[]))

        x = features.get("read-write",[(0,0,0)])
        reads, writes, ratio = zip(*x)

        ret["reads"] = reads
        ret["writes"] = writes

        x = features.get("alloc-dealloc",[(0,0,0)])
        allocs, deallocs, ratio = zip(*x)

        ret["allocs"] = allocs
        ret["deallocs"] = deallocs

        return ret
 
    def write(self, filename):
        row = [self._program]
        features = self._features
        row.append( " ".join(features.get("insns",[])) )
        row.append( " ".join(features.get("syscalls",[])) )

        x = features.get("read-write",[(0,0,0)])
        reads, writes, ratio = zip(*x)

        row.append( ",".join(map(str, reads)))
        row.append( ",".join(map(str, writes)))

        x = features.get("alloc-dealloc",[(0,0,0)])
        allocs, deallocs, ratio = zip(*x)

        row.append( ",".join(map(str, allocs)))
        row.append( ",".join(map(str, deallocs)))

        with gzip.open(filename, 'a+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(row)