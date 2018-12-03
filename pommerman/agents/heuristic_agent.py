'''The Reward class is used to calculate the reward value for each action
with three different state (Explore, Attack, Evade)
'''
from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from . import SimpleAgent
from .reward import Reward
from .. import constants
from .. import utility

class HeuristicAgent(SimpleAgent):
    def __init__(self, *args, **kwargs):
        super(HeuristicAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        # modify the obs
        obs['board'][obs['board'] == obs['teammate'].value] = constants.Item.Flames.value
        mode = Reward().decideMode(obs, action_space)
        # check mode and return the acts
        if mode == 0 or 1:
            return mcts(obs)
        else :
            return super(IgnoreAgent, self).act(obs, action_space)
