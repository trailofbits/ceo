import random

from collections import deque
from manticore.core.plugin import Plugin

class FeatureExtractor(Plugin):
    def will_start_run_callback(self, state):
        state.context['last_insn'] = deque([],128)

    def did_execute_instruction_callback(self, state, pc, target_pc, instruction):
        #print state.context['last_insn']
        state.context['last_insn'].append(instruction)

class FeatureCollector(Plugin):
    def __init__(self, features, rate):
        self.features = features
        self.last_state = None
        self.rate = rate
        super(FeatureCollector, self).__init__()

    def did_execute_instruction_callback(self, state, prev_pc, next_pc, instruction):
        if random.random() <= self.rate:
            self.features.add_insns(state)

        self.features.add_visited((prev_pc, next_pc)) 
        self.last_state = state
        
    def did_finish_run_callback(self):
        self.features.add_syscalls_seq(self.last_state)
        self.features.add_syscalls_stats(self.last_state)


        #print self.last_state.platform.syscall_trace

class StateCollector(Plugin):
    def __init__(self):
        super(StateCollector, self).__init__()

    def will_fork_state_callback(self, state, expression, solutions, policy):
        self.manticore._executor.generate_testcase(state, "New forked state!")

class Abandonware(Plugin):

    def will_execute_instruction_callback(self, state, pc, insn):
        name = state.cpu.canonicalize_instruction_name(insn)
        if not hasattr(state.cpu, name):
            state.abandon()
