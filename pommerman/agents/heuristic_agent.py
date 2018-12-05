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
from .. import utility

class HeuristicAgent(SimpleAgent):
    """Heuristic agent"""
    def __init__(self, *args, **kwargs):
        super(HeuristicAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        # modify the obs
        mode = Reward().decideMode(obs, action_space)
        # check mode and return the acts
        if mode in {constants.Mode.Evade, constants.Mode.Attack}:
            mcts = MCTree(obs)
            return mcts.bestAction()
        else :
            return super().act(obs, action_space)
