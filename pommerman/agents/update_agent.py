'''An agent playground to test the update function'''
from . import BaseAgent
from .. import constants
from .. import utility
import numpy as np
from collections import defaultdict


class UpdateAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        print(self.children_count(obs, action_space))
        return action_space.sample()

    @staticmethod
    def children_count(obs, action_space):
        filter = np.vectorize(lambda x : x >= 9)        # a function check is it an agent elementwise
        player_cnt = np.sum(filter(obs['board']))

        return player_cnt * action_space.n

    @staticmethod
    def update(action, obs):
        pass


class Simulator(object):
    def __init__(self, action, obs, blast_tracker):
        self.obs = obs
        self.action = action
        self.blast_tracker = blast_tracker  # a tracker to track all the blast

        # get the dictionary of observed agents id: position
        self.agents = self._get_agents()

    def children_count(self):
        filter = np.vectorize(lambda x : x >= 9)        # a function check is it an agent elementwise
        player_cnt = np.sum(filter(self.obs['board']))
        return player_cnt * self.action_space.n



    # given a combination of action, i can return a new observation
    def simulate(self, actions):
        # make a copy of the current observation
        obs = self.obs.copy()

        # all the flames on the board will remain the same

        # step the agents
        # update position according to the actions
        agents_pos = self.agents.copy()
        for agent_id, pos in self.agents:
            # remove the agent from the board
            obs['board'] = constants.Item.Passage.value

            # skip if agent not found or it is stop
            if agent_id not in actions or actions[agent_id] == constants.Action.Stop.value:
                continue

            action = actions[agent_id]

            if action == constants.Action.Bomb.value:
                if self._is_myself(agent_id):
                    if obs['ammo'] > 0 and obs['bomb_blast_strength'][pos] == 0:
                        # decrease the ammo and lay the bomb
                        obs['ammo'] -= 1
                        obs['bomb_life'] = 10
                        obs['bomb_blast_strength'] = obs['blast_strength']
                else:
                    if obs['bomb_blast_strength'][pos] == 0:
                        obs['bomb_life'] = 10
                        obs['bomb_blast_strength'] = self.blast_tracker[agent_id]

            else:   # the action is the direction
                if utility.is_valid_direction(obs['board'], pos, action):
                    next_pos = self._get_next_pos(pos, action)
                    agents_pos[agent_id] = next_pos

        # Gather desired next positions for moving bombs. Handle kicks later.
        #TODO will not distinguish the moving bomb
        #TODO don't know can kick the bomb or not
        # NOTE the bomb cannot moving through powerup object

        # Position switches:
        # Agent <-> Agent => revert both to previous position.
        # Bomb <-> Bomb => revert both to previous position.
        # Agent <-> Bomb => revert Bomb to previous position.
        crossings = {}

        def crossing(current, desired):
            '''Checks to see if an agent is crossing paths'''
            current_x, current_y = current
            desired_x, desired_y = desired
            if current_x != desired_x:
                assert current_y == desired_y
                return ('X', min(current_x, desired_x), current_y)
            assert current_x == desired_x
            return ('Y', current_x, min(current_y, desired_y))

        for agent_id in agents_pos.keys():
            # check is the agent cross
            if agents_pos[agent_id] != self.agents[agent_id]:
                desired_pos = agents_pos[agent_id]
                border = crossing(self.agents[agent_id], desired_pos)
                if border in crossings:
                    # Crossed another agent - revert both to previous pos
                    agents_pos[agent_id] = self.agents[agent_id]
                    agent2, _ = crossings[border]
                    agents_pos[agent2] = self.agents[agent2]
                else:
                    crossings[border] = (agent_id, True)

        # assume no bomb is moving so no bomb <-> bomb switch

        # deal with multiple agents collisions
        agent_occupancy = defaultdict(int)
        for pos in agents_pos.values():
            agent_occupancy[pos] += 1

        change = True
        while change:
            change = False
            for agent_id in agents_pos.keys():
                desired_pos = agents_pos[agent_id]
                curr_pos = self.agents[agent_id]
                # Another agent is going to this position
                if desired_pos != curr_pos and agent_occupancy[desired_pos] > 1:
                    desired_pos[agent_id] = curr_pos
                    agent_occupancy[curr_pos] += 1
                    change = True

            # assume bomb is not moving

        # Handle kicks.
        for agent_id, pos in agents_pos.items():
            if agent_occupancy[pos] == 0:
                # There was never an agent around to kick or collide.
                continue

            if not utility.position_is_bomb(obs['board'], pos):
                # The collision is not with Bomb
                continue

            # assume other agents cannot kick
            if not self._is_myself() or not obs['can_kick']:
                # bounce back
                agents_pos[agent_id] = self.agents[agent_id]
                continue

            # if it can kick
            direction = actions[agent_id]
            target_pos = utility.get_next_position(pos, direction)

            if utility.position_on_board(obs['board'], target_pos) and \
                        agent_occupancy[target_pos] == 0 and \
                        not utility.position_is_bomb(obs['board'], target_pos) and \
                        not utility.position_is_powerup(obs['board'], target_pos) and \
                        not utility.position_is_wall(obs['board'], target_pos):
                # ok we can kick
                # skip a lot of check, just kick
                self._move_bomb(obs, pos, target_pos)

        # Handle explode


        # Handle pickup.



    def _handle_move

    # get the dictionary of observed agents id: position
    def _get_agents(self):
        pass

    def _is_myself(self, agent_id):
        pass

    @staticmethod
    def _get_next_pos(pos, action):
        pass

    @staticmethod
    def _move_bomb(obs, curr_pos, target_pos):
        pass

if __name__ == '__main__':
    blast_tracker = {'global_max': 9, '9': 2, '10': 2, '11': 2, '12': 2, '13': 2}