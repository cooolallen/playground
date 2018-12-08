'''The Reward class is used to calculate the reward value for each action
with three different state (Explore, Attack, Evade)
'''
from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from . import SimpleAgent
from ..helpers.mcts import MCTree
from ..helpers.reward import Reward
from .. import constants
import timeout_decorator

class HeuristicAgent(SimpleAgent):
    """Heuristic agent"""
    def __init__(self, *args, **kwargs):
        super(HeuristicAgent, self).__init__(*args, **kwargs)
        self.bestAction = None

    def act(self, obs, action_space):
        try:
            # try to return the action by method if not time out
            return self._act(obs, action_space)
        except:
            # if it is timeout return the current best action
            print('time out, best action:', self.bestAction)
            if self.bestAction is None:
                return action_space.sample()
            else:
                if isinstance(self.bestAction, list):
                    return random.sample(self.bestAction)
                else:
                    return self.bestAction

                # reset the best action for the next run
                self.bestAction = None

    @timeout_decorator.timeout(0.1)       # the function will timeout after 100ms
    def _act(self, obs, action_space):
        # modify the obs
        mode = Reward().decideMode(obs, action_space)
        # check mode and return the acts
        if mode in {constants.Mode.Evade, constants.Mode.Attack}:
            mcts = MCTree(obs, parent=self)
            action = mcts.bestAction()
            print(action)
            return action
        else :
            return super().act(obs, action_space)
