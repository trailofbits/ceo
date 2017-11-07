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
        self.rate = rate
        super(FeatureCollector, self).__init__()

    def did_execute_instruction_callback(self, state, pc, target_pc, instruction):
        if random.random() <= self.rate:
            self.features.add_insns(state)
        self.features.add_syscalls_seq(state)
        self.features.add_syscalls_stats(state)


