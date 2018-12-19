#!/usr/bin/env python
import random
import operator
import collections
import uuid
import math
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

SCALAR=1/math.sqrt(2.0)


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

    def incrementVisit(self):
        self.visits += 1

    def updateStatus(self, step=None, minimax=False):
        children = [x for l in self.children.values() for x in l]
        visits = [x.isVisited for x in children]
        self.isVisited = sum(visits) == self.expected_num_of_children
        if step:
            rewards = [x.getAggregatingReward() for x in self.children[step]]
            if minimax:
                return min(rewards), max(rewards)
            return sum(rewards)/len(rewards), max(rewards)
        
        if minimax:
            rewards = [min([n.getAggregatingReward()]) for action in self.children for n in self.children[action]]
            _max = max(rewards)
            return _max, _max
        
        rewards = [x.getAggregatingReward() for x in children]
        return sum(rewards)/len(rewards) + self.reward, max(rewards) + self.reward

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


class MCTree:
    '''Monte-Carlo Tree'''
    def __init__(self, obs, level=2, agent=None, turn=1000):
        self.root = Node(obs, root_flag=True)
        self.level = level
        self.best_action = random.choice(ACTIONS)
        self.agent = agent
        self.priority = []
        self.turn = turn
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
        best_act = self.priority[0]
        self.best_action = random.choice([n.getAction() for n in self.priority if n.getReward()==best_act.getReward()])
        self.agent.best_action = self.best_action.value
        
    def _expand(self, curr, is_leaf=False):
        '''return the next to-be-visited node'''
        #print("is_leaf:",is_leaf)
        if not curr.obs_generators:
            return None, None
        action = random.choice(list(curr.obs_generators.keys()))
        num_of_next_obs = curr.getNumOfNextObs4SingleAct(action)
        while num_of_next_obs == 0:
            del curr.obs_generators[action]
            if not curr.obs_generators: break
            action = random.choice(list(curr.obs_generators.keys()))
            num_of_next_obs = curr.getNumOfNextObs4SingleAct(action)
        
        if not curr.obs_generators:
            return None, None
        
        #print(curr.children[action])
        if not curr.children[action]:
            obs_generator = curr.obs_generators[action]
            nxt_node = curr.getNext(next(obs_generator.generator))
            obs_generator.remains -= 1
            curr.num_of_children += 1
            curr.children[action].append(nxt_node)
            return action, nxt_node 
        
        elif random.uniform(0,1) < .5:
            return action, self._bestChild(action, curr, SCALAR)

        elif curr.obs_generators[action].remains != 0:
            obs_generator = curr.obs_generators[action]
            nxt_node = curr.getNext(next(obs_generator.generator))
            obs_generator.remains -= 1
            curr.num_of_children += 1
            curr.children[action].append(nxt_node)
            return action, nxt_node
        
        return action, self._bestChild(action, curr, SCALAR)
    
    def _bestChild(self, action, node, scalar):
        best_score=-float('inf')
        best_children = []
        for c in node.children[action]:
            exploit=c.reward
            explore=math.sqrt(2.0*math.log(node.visits)/c.visits)
            score = exploit + scalar * explore
            if score == best_score:
                best_children.append(c)
            if score>best_score:
                best_children=[c]
                best_score=score
        
        return random.choice(best_children)
         
    def bestAction(self, minimax=False):
        '''Return best action'''
        self._initHeap()
        curr = self.root
        first_step = None
        curr_turn = self.turn
        while curr_turn:
            '''Traverse to leaf'''
            curr = self.root
            curr_level = self.level - 1 
            while curr_level: 
                prev = curr
                step, curr = self._expand(curr)
                if not curr: break
                if curr_level == self.level - 1:
                    first_step = step
                curr_level -= 1
            
            if curr:  
                step, leaf = self._expand(curr, is_leaf=True)
                if leaf:
                    curr = leaf
            else:
                curr = prev
            
            curr.isVisited = True
            curr.incrementVisit()
            curr.setAggregatingReward(curr.getReward())    
            curr = curr.parent
            
            '''Back propagate'''
            while curr and curr != self.root:
                _agg, _max = curr.updateStatus(minimax=minimax)
                curr.incrementVisit()
                curr.setAggregatingReward(_agg)
                curr.setMaxReward(_max)
                curr = curr.parent
            
            '''Update best action'''
            if self.root.num_of_children:
                _agg, _ = self.root.updateStatus(step=first_step, minimax=minimax)
                self.root.incrementVisit()
                self.rewards[first_step].setReward(_agg)
                heapq.heapify(self.priority)
                self._updateBestAction()
            
            curr_turn -= 1

        return self.best_action.value


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

    def bestAction(self, minimax=False):
        '''Return best action'''
        self._initHeap()
        curr = self.root
        first_step = None
        while not self.root.isVisited:
            '''Traverse to leaf'''
            curr = self.root
            curr_level = self.level - 1 
            while curr_level: 
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
            curr = curr.parent
            
            '''Back propagate'''
            while curr and curr != self.root:
                _agg, _max = curr.updateStatus(minimax=minimax)
                curr.setAggregatingReward(_agg)
                curr.setMaxReward(_max)
                curr = curr.parent
            
            '''Update best action'''
            if self.root.num_of_children:
                _agg, _ = self.root.updateStatus(step=first_step, minimax=minimax)
                self.rewards[first_step].setReward(_agg)
                heapq.heapify(self.priority)
                self._updateBestAction()
            
        return self.best_action.value


class BFSTree:
    '''BFS Tree'''
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
            temp.append(parent)
            
            if not leaves:
                leaves = temp
                temp = []
        return leaves[0]        


if __name__=="__main__":
    pass