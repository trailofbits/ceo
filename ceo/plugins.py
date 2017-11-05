from collections import deque

from manticore.core.plugin import Plugin

class FeatureCollector(Plugin):
    def will_start_run_callback(self, state):
        state.context['last_insn'] = deque([],128)

    def did_execute_instruction_callback(self, state, pc, target_pc, instruction):
        state.context['last_insn'].append(instruction)
