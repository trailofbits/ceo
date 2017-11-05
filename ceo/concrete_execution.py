from manticore.core.smtlib import ConstraintSet
from manticore.core.state import State
from manticore.platforms import linux, decree, windows, evm

def make_initial_concrete_state(binary_path, concrete_data):
    magic = file(binary_path).read(4)
    if magic == '\x7fELF':
        # Linux
        state = make_concrete_linux(binary_path, concrete_data)
    elif magic == '\x7fCGC':
        # Decree
        state = make_concrete_decree(binary_path, concrete_data)
    elif magic == '#EVM':
        assert(0)
        #state = make_concrete_evm(binary_path, **kwargs)
    else:
        raise NotImplementedError("Binary {} not supported.".format(binary_path))
    return state


def make_concrete_decree(program, concrete_data, **kwargs):
    constraints = ConstraintSet()
    platform = decree.SDecree(constraints, program)
    initial_state = State(constraints, platform)
    #logger.info('Loading program %s', program)

    #if concrete_data != '':
    #    logger.info('Starting with concrete input: {}'.format(concrete_data))
    platform.input.transmit(concrete_data)
    #platform.input.transmit(initial_state.symbolicate_buffer('+'*14, label='RECEIVE'))
    return initial_state
