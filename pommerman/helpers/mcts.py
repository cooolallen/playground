#!/usr/bin/env python
import random
import operator
import collections
import uuid
import heapq
from .simulator import Simulator  
from .reward import Reward
from .. import constants

'''Globals'''
STOP = constants.Action.Stop
UP = constants.Action.Up
DOWN = constants.Action.Down
LEFT = constants.Action.Left
RIGHT = constants.Action.Right
BOMB = constants.Action.Bomb
 
ACTIONS = [
    STOP, UP, DOWN, LEFT, RIGHT, BOMB
]

class Generator:
    '''Generator'''
    def __init__(self, generator, remains):
        self.generator = generator
        self.remains = remains
    
class Node:
    '''Tree Node'''
    def __init__(self, obs, parent=None, reward=0.0,
                action_space={}, bomb_tracker={}, root_flag=False):
        self.visits = 0
        self.uid = uuid.uuid4()
        self.isVisited = False
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
        self.num_of_next_obs = {
            STOP: self.simulator.getNumOfNextObs(STOP),
            UP: self.simulator.getNumOfNextObs(UP),
            DOWN: self.simulator.getNumOfNextObs(DOWN),
            LEFT: self.simulator.getNumOfNextObs(LEFT),
            RIGHT: self.simulator.getNumOfNextObs(RIGHT),
            BOMB: self.simulator.getNumOfNextObs(BOMB)
        }
        #print(self.num_of_next_obs)
        self.expected_num_of_children = sum(self.num_of_next_obs.values()) 
        self.counter = self.expected_num_of_children
        self.obs_generators = {
            STOP: Generator(self.simulator.update(STOP), self.num_of_next_obs[STOP]),
            UP: Generator(self.simulator.update(UP), self.num_of_next_obs[UP]),
            DOWN: Generator(self.simulator.update(DOWN), self.num_of_next_obs[DOWN]),
            LEFT: Generator(self.simulator.update(LEFT), self.num_of_next_obs[LEFT]),
            RIGHT: Generator(self.simulator.update(RIGHT), self.num_of_next_obs[RIGHT]),
            BOMB: Generator(self.simulator.update(BOMB), self.num_of_next_obs[BOMB])
        }
        
    def isRoot(self):        
        return self.root_flag
    
    def getReward(self):
        return self.reward
    
    def getAggregatingReward(self):
        return self.aggregating_reward
    
    def setAggregatingReward(self, reward):
        self.aggregating_reward = reward
    
    def getMaxReward(self):
        return self.max_reward

    def getNumOfNextObs4SingleAct(self, action):
        return self.num_of_next_obs[action]
    
    def setMaxReward(self, reward):
        self.max_reward = reward
    
    def getNext(self, next_obs):
        next_reward = Reward().reward(next_obs, self.mode) or 0.0
        return Node(next_obs, parent=self, reward=next_reward)

    def updateStatus(self):
        children = [x for l in self.children.values() for x in l]
        visits = [x.isVisited for x in children]
        rewards = [x.aggregating_reward for x in children]
        #print(rewards)
        self.isVisited = sum(visits) == self.expected_num_of_children
        self.aggregating_reward = sum(rewards)/len(rewards) + self.reward #should add reward or not?
        self.max_reward = max(rewards) + self.reward
        
    def _expand(self, action, computer_reward=False):
        """
        Add children with specific action
        """
        if self.mode == constants.Mode.Explore: return

        next_observations = self.simulator.update(action)
        if computer_reward:
            for next_obs in next_observations:
                child = self.getNext()
                self.counter -= 1
                self.num_of_children += 1
                self.children[action].append(child)
            return
        for next_obs in next_observations:
            self.counter -= 1
            child = Node(next_obs, parent=self)
            self.num_of_children += 1
            self.children[action].append(child)
        
    def expandAll(self, computer_reward=False):
        """
        expand all child node
        """
        for action in ACTIONS:
            self._expand(action, computer_reward=computer_reward)
        return [x for l in self.children.values() for x in l]

    def fullyExpanded(self):
        return self.counter == 0

class Act:
    '''self-defined object for act-reward pair'''
    def __init__(self, action):
        self.action = action
        self.reward = 0.0
    
    def __lt__(self, other):
        return self.reward > other.reward

    def setReward(self, reward):
        self.reward = reward
    
    def getReward(self):
        return self.reward
    
    def getAction(self):
        return self.action

class SimTree:
    '''Just do some random play-out'''
    def __init__(self, obs, level=2, agent=None):
        self.root = Node(obs, root_flag=True)
        self.level = level
        self.best_action = random.choice(ACTIONS)
        self.agent = agent
        self.priority = []
        self.rewards = {
            STOP: Act(STOP), 
            UP: Act(UP), 
            DOWN: Act(DOWN), 
            LEFT: Act(LEFT), 
            RIGHT: Act(RIGHT), 
            BOMB: Act(BOMB)
        }

    def _initHeap(self):
        for action in self.rewards:
            self.priority.append(self.rewards[action])
    
    def _updateBestAction(self):
        '''The function to propagate the current best action back to the parent'''
        self.best_action = self.priority[0].getAction()
        # print("best so far", self.best_action)
        self.agent.best_action = self.best_action.value
        
    def _randomSelect(self, curr, is_leaf=False):
        '''Randomly return a next node'''
        action = random.choice(list(curr.obs_generators.keys()))
        num_of_next_obs = curr.getNumOfNextObs4SingleAct(action)
        while num_of_next_obs == 0:
            del curr.obs_generators[action]
            if not curr.obs_generators: break
            action = random.choice(list(curr.obs_generators.keys()))
            num_of_next_obs = curr.getNumOfNextObs4SingleAct(action)
        
        if not curr.obs_generators:
            return None, None
        
        if is_leaf:
            while curr.obs_generators[action].remains == 0:
                action = random.choice(list(curr.obs_generators.keys()))
                num_of_next_obs = curr.getNumOfNextObs4SingleAct(action)

        rdn = random.randint(0, num_of_next_obs-1)
        if is_leaf or rdn >= len(curr.children[action]):
            
            obs_generator = curr.obs_generators[action]
            #print('rdn', rdn)
            #print('num_of_children', len(curr.children[action]))
            #print('is_leaf', is_leaf)
            nxt_node = curr.getNext(next(obs_generator.generator))
            obs_generator.remains -= 1
            curr.num_of_children += 1
            curr.children[action].append(nxt_node)
            return action, nxt_node 
        else:
            nxt_node = curr.children[action][rdn]
            if nxt_node.isVisited:
                return self._randomSelect(curr)
            return action, nxt_node

        return None, None

    def bestAction(self):
        '''Return best action'''
        # print("Start finding best action")
        self._initHeap()
        curr = self.root
        first_step = None
        while not self.root.isVisited:
            #print("root isVisited", self.root.isVisited)
            '''Traverse to leaf'''
            curr_level = self.level - 1 
            while curr_level: 
                #print("curr_level", curr_level)
                prev = curr
                step, curr = self._randomSelect(curr)
                if not curr: break
                if curr_level == self.level - 1:
                    first_step = step
                curr_level -= 1
            
            if curr:  
                step, leaf = self._randomSelect(curr, is_leaf=True)
                if leaf:
                    curr = leaf
            else:
                curr = prev
            curr.isVisited = True
            curr.setAggregatingReward(curr.getReward())    
            '''Back propagate'''
            curr = curr.parent
            while curr != self.root:
                curr.updateStatus()
                curr = curr.parent
            #print("root's children", self.root.children)
            #print("first_step", first_step)
            self.root.isVisited = sum([n.isVisited for l in self.root.children.values() for n in l]) == self.root.expected_num_of_children
            '''Update best action'''
            rewards = [n.aggregating_reward for n in self.root.children[first_step]]
            updating_reward = sum(rewards)/len(rewards)
            self.rewards[first_step].setReward(updating_reward)
            heapq.heapify(self.priority)
            self._updateBestAction()
        
        return self.best_action.value

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
        root = self._backPropagate(leaves)
        ave_rewards = {}

        for action in ACTIONS:
            rewards = [x.aggregating_reward for x in root.children[action]]
            ave_rewards[action] = sum(rewards)/len(rewards)

        max_agg_reward = max(ave_rewards.values())
        best_actions = [k for k in ave_rewards if ave_rewards[k] == max_agg_reward]
        #print(ave_rewards.values())
        return random.choice(best_actions)

    def _buildTree(self):
        queue = [self.root]
        for _ in range(self.level):
            temp = []
            while queue:
                curr = queue.pop(0)
                temp.extend(curr.expandAll(computer_reward=True))
            queue = temp
            # print('reward:', [n.reward for n in queue])

        for node in queue:
            node.setAggregatingReward(node.getReward())    
        
        return queue

    def _backPropagate(self, leaves):
        temp = []
        while not leaves[0].isRoot():
            parent = leaves[0].parent
            num_of_children = parent.num_of_children
            aggregating_rewards = []
            
            for _ in range(num_of_children):
                leaf = leaves.pop(0)
                aggregating_rewards.append(leaf.getAggregatingReward())
            
            parent.setAggregatingReward(sum(aggregating_rewards)/len(aggregating_rewards)+parent.reward)
            parent.setMaxReward(max(aggregating_rewards)+parent.reward)
            #parent.setAggregatingReward(random.random())
            #parent.setMaxReward(random.random())
            temp.append(parent)
            
            if not leaves:
                leaves = temp
                temp = []
        return leaves[0]        


if __name__=="__main__":
    pass