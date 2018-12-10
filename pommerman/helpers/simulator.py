"""Helpers"""
from .. import constants
from .. import utility
from ..characters import Bomber, Bomb, Flame
from ..forward_model import ForwardModel
from .. import constants
import numpy as np
import itertools
import random

class Simulator:
    """Simulator for Monte-Carlo Tree"""
    def __init__(self, obs, blast_tracker):
        self._obs = obs
        self._myself_idx = self._get_myself_idx()
        self._blast_tracker = blast_tracker  # a tracker to track all the blast
        self._args = self._get_args()
        self._observed_alive_agents = self._get_observed_alive_agents()

    def getNumOfNextObs(self):
        return 6 ** len(self._observed_alive_agents)
    
    def update(self, action):
        for actions in self._action_combination_generator(action):
            yield self._simulate(actions)
    
    def _action_combination_generator(self, own_action):
        # remove myself from alive set because we know what myself going to do
        # create a pointer list for others
        others_action_pointer = [agent_id - 10 for agent_id in self._observed_alive_agents if agent_id != self._myself_idx]
        actions_template = [0] * 4
        actions_template[self._myself_idx - 10] = own_action.value

        # create the random order action arguments
        random_action_list = list(itertools.product(range(6), repeat=len(others_action_pointer)))
        random.shuffle(random_action_list)

        for actions in random_action_list:
            curr_actions = actions_template.copy()
            for idx, action in zip(others_action_pointer, actions):
                curr_actions[idx] = action
            yield curr_actions
    
    def _simulate(self, actions):
        simulate_args = self._args.copy()
        simulate_args['actions'] = actions
        board, agents, bombs, items, flames = ForwardModel.step(**simulate_args)
        simulate_obs = ForwardModel.get_observations(None, board, agents, bombs, True, 4, constants.GameType(self._obs['game_type']), self._obs['game_env'])
        own_obs = self._get_own_obs(simulate_obs)
        return own_obs
    
    def _get_args(self):
        args = {}
        args['curr_board'] = self._obs['board'].copy()
        args['curr_agents'] = self._get_agents()
        board = self._obs['board']
        curr_bombs, curr_items, curr_flames = [], {}, []
        rows, cols = len(board), len(board[0])
        dummy_bomber = Bomber(10)
        for i in range(rows):
            for j in range(cols):
                pos = (i, j)
                if board[pos] == constants.Item.Bomb.value:
                    bomb = Bomb(dummy_bomber, pos, self._obs['bomb_life'][pos], int(self._obs['bomb_blast_strength'][pos]))
                    curr_bombs.append(bomb)
                elif utility.position_is_powerup(board, pos):
                    curr_items[pos] = board[pos]
                elif utility.position_is_flames(board, pos):
                    flame = Flame(pos)
                    curr_flames.append(flame)
        args['curr_bombs'] = curr_bombs
        args['curr_items'] = curr_items
        args['curr_flames'] = curr_flames
        return args
    
    def _get_observed_alive_agents(self):
        alive_agents = []
        for agent in self._args['curr_agents']:
            if agent.is_alive:
                alive_agents.append(agent.agent_id + 10)
        return alive_agents
    
    def _get_agents(self):
        agents = []
        board = self._obs['board']
        default_pos = [(1, 1), (9, 1), (9, 9), (1, 9)]
        for agent_id in [10, 11, 12, 13]:
            agent = Bomber(agent_id - 10, self._obs['game_type'])        # the agent id is start from 0
            pos = np.argwhere(board == agent_id)
            if pos.shape[0] == 0:
                # we cannot find the agent
                agent.is_alive = False
                # give them the default position
                agent.position = default_pos[agent_id - 10]
            else:
                agent.position = tuple(pos[0])
                # update the agent information
                if self._is_myself(agent_id):
                    agent.blast_strength = self._obs['blast_strength']
                    agent.can_kick = self._obs['can_kick']
                    agent.ammo = self._obs['ammo']
                else:
                    # if not myself, try to image a strong enemies
                    agent.blast_strength = 7    # todo modify it to tracker
                    agent.ammo = 10             # it can drop bomb every time
            agents.append(agent)
        return agents
    
    def _is_myself(self, agent_id):
        return agent_id == self._myself_idx
    
    def _get_own_obs(self, simulate_obs):
        for agent_id in [10, 11, 12, 13]:
            if self._is_myself(agent_id):
                own_obs = simulate_obs[agent_id - 10]
                break
        # update the alive
        if own_obs['alive'] == self._observed_alive_agents:
            own_obs['alive'] = self._obs['alive']
        else:
            # some agents die due to the simulation
            dead_agents = [agent_id for agent_id in self._observed_alive_agents if agent_id not in own_obs['alive']]
            own_obs['alive'] = [agent_id for agent_id in self._obs['alive'] if agent_id not in dead_agents]
        return own_obs

    def _get_myself_idx(self):
        for agent_id in [10, 11, 12, 13]:
            if constants.Item(agent_id) != self._obs['teammate'] and \
                            constants.Item(agent_id) not in self._obs['enemies']:
                return agent_id

        print('agent_id not found')
        print('teammate', self._obs['teammate'], 'enemies', self._obs['enemies'])
