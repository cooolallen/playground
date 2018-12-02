#!/usr/bin/env python
import random
import math
import operator
import hashlib
import argparse
import collections
from .simulator import Simulator  
from .reward import Reward

'''Globals'''
ACTIONS = [
    constants.Action.Stop, constants.Action.Up, constants.Action.Down, 
    constants.Action.Left, constants.Action.Right, constants.Action.Bomb
]

class SimTree:
    def __init__(self, obs, level=2):
        pass


class MCTree:
    '''Monte-Carlo Tree'''
    def __init__(self, obs={}, level=2):
        self.root = Node(obs, root_flag=True)
        self.level = level
        self.best_action = constants.Action.Stop

    def bestAction(self):
        """
        Backpropagate and return the best action
        """
        leaves = self._buildTree()
        self._backPropagate(leaves)
        root = leaves[0]
        ave_rewards = {}

        for action in ACTIONS:
            rewards = [x.aggregating_reward for x in root.children[action]]
            ave_rewards[action] = sum(rewards)/len(rewards)
        best_action = max(ave_rewards.items(), key=operator.itemgetter(1))[0]        
        return best_action

    def _buildTree(self):
        queue = [self.root]
        temp = []
        for _ in range(self.level):
            while queue:
                curr = queue.pop(0)
                temp.extend(curr.expandAll())
            queue = temp

        for node in queue:
            node.setAggregatingReward(node.getReward())    
        
        return queue

    def _backPropagate(self, leaves)
        temp = []
        while not leaves[0].isRoot() :
            parent = leaves[0].parent
            num_of_children = parent.num_of_children
            aggregating_rewards = []
            
            for _ in num_of_children:
                leaf = leaves.pop(0)
                aggregating_reward.append(leaf.getAggregatingReward())
            
            parent.setAggregatingReward(sum(aggregating_rewards)/len(aggregating_rewards))
            parent.setMaxReward(max(aggregating_rewards))
            temp.append(parent)
            
            if not leaves:
                leaves = temp

        
class Node:
    '''Tree Node'''
	def __init__(self, obs, parent=None, reward=0.0, 
                action_space={}, bomb_tracker={}, root_flag=False):
		self.visits = 1
        self.obs = obs
        self.root_flag = root_flag 
        self.mode = Reward().decideMode(obs, action_space)
		self.children = collections.defaultdict(list)
        self.parent = parent
        self.reward = reward
        self.max_reward = -float('inf')
        self.aggregating_reward = 0
        self.simulator = Simulator(obs, bomb_tracker)
        self.num_of_children = 0
        self.num_of_next_obs = self.simulator.getNumOfNextObs(obs)
        self.counter = self.num_of_next_obs

    def isRoot(self):
        return root_flag
    
    def getReward(self):
        return self.reward
    
    def getAggregatingReward(self):
        return self.aggregating_reward
    
    def setAggregatingReward(self, reward):
        self.aggregating_reward = reward
    
    def getMaxReward(self):
        return self.max_reward
    
    def setMaxReward(self, reward):
        self.max_reward = reward
        
    def _expand(self, action):
		"""
        Add children with specific action
        """
        next_observations = self.simulator.update(action)
        for next_obs in next_observations:
            next_reward = Reward().reward(next_obs, self.mode)
            self.counter -= 1
            child = Node(next_obs, self, next_reward)
            self.num_of_children += 1
		    self.children[action].append(child)
	
    def expandAll(self):
        """
        expand all child node
        """
        for action in ACTIONS:
            self._expand(action)
        return [x for l in self.children.values() for x in l]
    
	def fullyExpanded(self):
        return self.counter == 0

if __name__=="__main__":
    pass