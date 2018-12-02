"""Reward"""
from collections import defaultdict
import queue
import random
from .. import constants

import numpy as np

class Reward:
    '''Reward functions'''
    def convert_bombs(self, bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret

    def decideMode(self, obs, action_space):
        ## switch condition and get action
        bombs = self.convert_bombs(np.array(obs['bomb_blast_strength']))
        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        ammo = int(obs['ammo'])
        if self.evadeCondition(my_position, bombs, obs['bomb_life']):
            # return evadeScore(my_position, obs['bomb_blast_strength'], obs['bomb_life'])
            return constants.Mode.Evade
        elif self.attackCondition(ammo, my_position, board):
            ''' 0 stand for empty safe position;
                1 stand for blocked by agent, wall, or bomb;
                2 stand for reachable by bomb'''
            # return attackScore(pos, board['board'], obs['bomb_blast_strength'])
            return constants.Mode.Attack
        else:
            return constants.Mode.Explore

    def reward(self, obs, mode):
        my_position = tuple(obs['position'])
        if mode == 0:
            return self.evadeScore(my_position, obs['bomb_blast_strength'], obs['bomb_life'])
        elif mode == 1:
            return self.attackScore(my_position, obs)

    def evadeCondition(self, pos, bombs, bomb_life):
        bomb_cnt = 0
        tick_cnt = 0
        for bomb in bombs:
            if self.calDistance(bomb['position'][0], bomb['position'][1], pos[0], pos[1]) <= bomb['blast_strength']:
                bomb_cnt += 1
                tick_cnt += bomb_life[bomb['position'][0]][bomb['position'][1]]
        return tick_cnt < 5 + 2*bomb_cnt

    def evadeScore(self, pos, bomb_strength, bomb_life):
        score = 100
        for i in range(len(bomb_strength)):
            for j in range(len(bomb_strength[0])):
                if bomb_strength[i][j] > 0 and bomb_strength[i][j] >= (abs(pos[0]-i) + abs(pos[1]-j)):
                    score = score - (25 * (11 - bomb_life[i][j])/10)
        return score

    def attackCondition(self, ammo, my_pos, board):
        if ammo == 0: return False
        # this can be improve using bfs
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in enemyList and self.calDistance(i, j, my_pos[0], my_pos[1]):
                    return True
        return False

    def attackScore(self, pos, obs):
        fill_area = self.calFillArea(pos)
        empty_safe_area = self.calEmptySafeArea(pos, obs)
        return 100 * (1 - float(empty_safe_area)/fill_area)

    def calFillArea(self, pos):
        fill_area = 13
        rToBound = min(2, min(pos[0] - 0, 10 - pos[0]))
        cToBound = min(2, min(pos[1] - 0, 10 - pos[1]))
        if rToBound < 2 or cToBound < 2:
            if rToBound == 0 and cToBound == 0:
                fill_area = 6
            elif rToBound == 1 and cToBound == 1:
                fill_area = 11
            elif rToBound < 2 and cToBound < 2:
                fill_area = 8
            else:
                if rToBound == 0 or cToBound == 0:
                    fill_area = 9
                else:
                    fill_area = 12
        return fill_area

    def calEmptySafeArea(self, pos, obs):
        board = np.zeros((11,11), dtype=int)
        agent_list = []
        for enemy in obs['enemies']:
            agent_list.append(enemy.value)
        for i in range(11):
            for j in range(11):
                if obs['board'][i][j] == 1 or obs['board'][i][j] == 2 or obs['board'][i][j] in agent_list:
                    board[i][j] = 1
        locations = np.where(obs['bomb_blast_strength'] > 0)
        for r, c in zip(locations[0], locations[1]):
            strength = obs['bomb_blast_strength'][r][c]
            # down
            for i in range(int(strength)):
                if r+i < 11 and board[r+i][c] == 0:
                    board[r+i][c] = 2
                elif r+i < 11 and (board[r+i][c] == 1 or board[r+i][c] == 2):
                    break
            # up
            for i in range(int(strength)):
                if r-i >=0 and board[r-i][c] == 0:
                    board[r-i][c] = 2
                elif r-i >=0 and board[r-i][c] == 1:
                    break
            # right
            for i in range(int(strength)):
                if c+i < 11 and board[r][c+i] == 0:
                    board[r][c+i] = 2
                elif c+i < 11 and board[r][c+i] == 1:
                    break
            # left
            for i in range(int(strength)):
                if c-i >=0 and board[r][c-i] == 0:
                    board[r][c-i] = 2
                elif c-i >=0 and board[r][c-i] == 1:
                    break
        # count empty safe area
        empty_safe_area = 0
        for i in range(5):
            for j in range(5):
                newR = pos[0]+i-2
                newC = pos[1]+j-2
                if newR >= 0 and newR < 11 and newC >= 0 and newC < 11 and self.calDistance(newR, newC, pos[0], pos[1]) <= 2:
                    if board[newR][newC] == 0:
                        empty_safe_area += 1
        return empty_safe_area

    def calDistance(self, pos1R, pos1C, pos2R, pos2C):
        return abs(pos1R - pos2R) + abs(pos1C - pos2C)
