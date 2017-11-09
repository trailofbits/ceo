from random import randint, sample

from manticore.core.smtlib import ConstraintSet
from manticore.core.state import State
from manticore.platforms import linux, decree, evm

def make_initial_symbolic_state(binary_path, concrete_data, dist_symb_bytes, rand_symb_bytes):
    magic = file(binary_path).read(4)
    if magic == '\x7fELF':
        # Linux
        state = make_symbolic_linux(binary_path, concrete_data, dist_symb_bytes, rand_symb_bytes)
    elif magic == '\x7fCGC':
        # Decree
        state = make_symbolic_decree(binary_path, concrete_data, dist_symb_bytes, rand_symb_bytes)
    elif magic == '#EVM':
        assert(0)
        #state = make_concrete_evm(binary_path, **kwargs)
    else:
        raise NotImplementedError("Binary {} not supported.".format(binary_path))
    return state


def make_symbolic_decree(program, concrete_data, dist_symb_bytes, rand_symb_bytes):
    assert( rand_symb_bytes > 0.0) 
    assert( concrete_data != '') 

    constraints = ConstraintSet()
    platform = decree.SDecree(constraints, program)
    initial_state = State(constraints, platform)
    concrete_len = len(concrete_data)
    symb_size = max(1, int(16*rand_symb_bytes))
    #print dist_symb_bytes, rand_symb_bytes

    if concrete_len < symb_size:
        random_symbolic = range(symb_size)
        concrete_data = [None]*symb_size 
    elif dist_symb_bytes == "dense":
        offset_symbolic = randint(0, concrete_len-symb_size)
        random_symbolic = range(offset_symbolic, offset_symbolic + symb_size)
    elif dist_symb_bytes == "sparse":
        random_symbolic = sample(range(concrete_len), symb_size)
    else:
        assert(0)

    for i in random_symbolic:
        concrete_data[i] = '+'

    concrete_data = "".join(concrete_data)
    print('Starting with concrete input: {}'.format(repr(concrete_data)))
 
    platform.input.transmit(initial_state.symbolicate_buffer(concrete_data, label='RECEIVE'))
    return initial_state
