'''The Reward class is used to calculate the reward value for each action
with three different state (Explore, Attack, Evade)
'''
from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from . import SimpleAgent
from ..helpers.mcts import MCTree, SimTree
from ..helpers.reward import Reward
from .. import constants
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError

class HeuristicAgent(SimpleAgent):
    """Heuristic agent"""
    def __init__(self, minmax=False):
        super(HeuristicAgent, self).__init__()
        self.best_action = None
        self.minmax = minmax

    def act(self, obs, action_space):
        try:
            # try to return the action by method if not time out
            return self._act(obs, action_space)
        except TimeoutError:
            # if it is timeout return the current best action
            # print('time out, best action:', self.best_action)
            if self.best_action is None:
                return action_space.sample()
            else:
                if isinstance(self.best_action, list):
                    return random.sample(self.best_action)
                else:
                    return self.best_action

                # reset the best action for the next run
                self.best_action = None

    @timeout_decorator.timeout(0.1)       # the function will timeout after 100ms
    def _act(self, obs, action_space):
        # modify the obs
        mode = Reward().decideMode(obs, action_space)
        # check mode and return the acts
        if mode in {constants.Mode.Evade, constants.Mode.Attack}:
            #mcts = MCTree(obs, agent=self)
            #action = mcts.bestAction()
            sim_tree = SimTree(obs, agent=self)
            action = sim_tree.bestAction(minimax=self.minmax)
            # print("best_action", action)
            return action
        else :
            return super().act(obs, action_space)
