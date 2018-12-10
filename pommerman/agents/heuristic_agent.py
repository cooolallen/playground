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
            return self.bfs(obs, mode)
        else :
            return super(SimpleAgent, self).act(obs, action_space)

    def bfs(self, obs, mode):
        moves = [
            constants.Action.Left, constants.Action.Right,
            constants.Action.Up, constants.Action.Down, constants.Action.Bomb
        ]
        pos = obs['position']
        move = constants.Action.Stop
        reward = Reward().reward(obs, mode)
        for newMove in moves:
            if newMove == 1:
                # out of board boundary
                if pos[1] - 1 < 0: continue
                nextPos = obs['board'][pos[0]][pos[1]-1]
                # the new position is blocked
                if nextPos == 1 or nextPos == 2 or nextPos == 3: continue
                # update environment
                obs['board'][pos[0]][pos[1]-1] = 10 # this need to configure
                obs['board'][pos[0]][pos[1]] = 0
                newReward = Reward().reward(obs, mode)
                if newReward > reward:
                    reward = newReward
                    move = newMove
                # backtrack environment
                obs['board'][pos[0]][pos[1]-1] = 0
                obs['board'][pos[0]][pos[1]] = 10 # this need to configure
            elif newMove == 2:
                # out of board boundary
                if pos[1] + 1 > 10: continue
                nextPos = obs['board'][pos[0]][pos[1]+1]
                # the new position is blocked
                if nextPos == 1 or nextPos == 2 or nextPos == 3: continue
                # update environment
                obs['board'][pos[0]][pos[1]+1] = 10 # this need to configure
                obs['board'][pos[0]][pos[1]] = 0
                newReward = Reward().reward(obs, mode)
                if newReward > reward:
                    reward = newReward
                    move = newMove
                # backtrack environment
                obs['board'][pos[0]][pos[1]+1] = 0
                obs['board'][pos[0]][pos[1]] = 10 # this need to configure
            elif newMove == 3:
                # out of board boundary
                if pos[0] - 1 < 0: continue
                nextPos = obs['board'][pos[0]-1][pos[1]]
                # the new position is blocked
                if nextPos == 1 or nextPos == 2 or nextPos == 3: continue
                # update environment
                obs['board'][pos[0]-1][pos[1]] = 10 # this need to configure
                obs['board'][pos[0]][pos[1]] = 0
                newReward = Reward().reward(obs, mode)
                if newReward > reward:
                    reward = newReward
                    move = newMove
                # backtrack environment
                obs['board'][pos[0]-1][pos[1]] = 0
                obs['board'][pos[0]][pos[1]] = 10 # this need to configure
            elif newMove == 4:
                # out of board boundary
                if pos[0] + 1 > 10: continue
                nextPos = obs['board'][pos[0]+1][pos[1]]
                # the new position is blocked
                if nextPos == 1 or nextPos == 2 or nextPos == 3: continue
                # update environment
                obs['board'][pos[0]+1][pos[1]] = 10 # this need to configure
                obs['board'][pos[0]][pos[1]] = 0
                newReward = Reward().reward(obs, mode)
                if newReward > reward:
                    reward = newReward
                    move = newMove
                # backtrack environment
                obs['board'][pos[0]+1][pos[1]] = 0
                obs['board'][pos[0]][pos[1]] = 10 # this need to configure
            elif newMove == 5:
                if obs['bomb_blast_strength'][pos[0]][pos[1]] != 0: continue
                # place bomb
                oriStrength = obs['bomb_blast_strength'][pos[0]][pos[1]]
                oriLife = obs['bomb_life'][pos[0]][pos[1]]
                obs['bomb_blast_strength'][pos[0]][pos[1]] = obs['blast_strength']
                obs['bomb_life'][pos[0]][pos[1]] = 10
                newReward = Reward().reward(obs, mode)
                if newReward > reward:
                    reward = newReward
                    move = newMove
                # backtrack bomb
                obs['bomb_blast_strength'][pos[0]][pos[1]] = oriStrength
                obs['bomb_life'] = oriLife

        return move
