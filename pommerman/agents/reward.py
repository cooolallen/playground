from collections import defaultdict
import queue
import random

import numpy as np

class Reward:

    def decideMode(obs, action_space):
        ## switch condition and get action
        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        ammo = int(obs['ammo'])
        if evadeCondition(my_position, bombs, obs['bomb_life']):
            # return evadeScore(my_position, obs['bomb_blast_strength'], obs['bomb_life'])
            return 0
        elif attackCondition(ammo, my_position, board):
            ''' 0 stand for empty safe position;
                1 stand for blocked by agent, wall, or bomb;
                2 stand for reachable by bomb'''
            # return attackScore(pos, board['board'], obs['bomb_blast_strength'])
            return 1
        else:
            return 2

    def reward(obs, mode):
        my_position = tuple(obs['position'])
        bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        if mode == 0:
            return evadeScore(my_position, bombs, obs['bomb_life'])
        elif mode == 1:
            return attackScore(my_position, obs)

    def evadeCondition(pos, bombs, bombLife):
        bombCnt = 0
        tickCnt = 0
        for bomb in bombs:
            if calDistance(bomb['position'][0], bomb['position'][1], my_position[0], my_position[1]) <= bomb['blast_strength']:
                bombCnt += 1
                tickCnt += bombLife[bomb['position'][0]][bomb['position'][1]]
        return tickCnt < 5 + 2*bombCnt

    def evadeScore(pos, bombStrength, bombLife):
        score = 100
        for i in range(len(bombStrength)):
            for j in range(len(bombStrength[0])):
                if bombStrength[i][j] > 0 and [i][j] >= (abs(pos[0]-i) + abs(pos[1]-j)):
                    score = score - (25 * (11 - bombLife[i][j])/10)
        return score

    def attackCondition(ammo, myPos, board):
        if ammo == 0: return False
        # this can be improve using bfs
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in enemyList and calDistance(i, j, myPos[0], myPos[1]):
                    return True
        return False

    def attackScore(pos, obs):
        fillArea = calFillArea(pos)
        emptySafeArea = calEmptySafeArea(pos, attackMap)
        return 100 * (1 - float(fillArea)/emptySafeArea)

    def calFillArea(pos):
        fillArea = 13
        rToBound = min(2, min(pos[0] - 0, 10 - pos[0]))
        cToBound = min(2, min(pos[1] - 0, 10 - pos[1]))
        if rToBound < 2 or cToBound < 2:
            if rToBound == 0 and cToBound == 0:
                fillArea = 6
            elif rToBound == 1 and cToBound == 1:
                fillArea = 11
            elif rToBound < 2 and cToBound < 2:
                fillArea = 8
            else:
                if rToBound == 0 or cToBound == 0:
                    fillArea = 9
                else:
                    fillArea = 12
        return fillArea

    def calEmptySafeArea(pos, obs):
        board = np.zeros((11,11), dtype=int)
        for i in range(len(11)):
            for j in range(len(11)):
                if obs['board'][i][j] == 1 or obs['board'][i][j] == 2 or obs['board'][i][j] in agentList:
                    board[i][j] = 1
        locations = np.where(obs['bomb_blast_strength'] > 0)
        for r, c in zip(locations[0], locations[1]):
            strength = obs['bomb_blast_strength'][r][c]
            # down
            for i in range(strength):
                if r+i < 11 and board[r+i][c] == 0:
                    board[r+i][c] = 2
                elif board[r+i][c] == 1 or board[r+i][c] == 2:
                    break
            # up
            for i in range(strength):
                if r-i >=0 and board[r-i][c] == 0:
                    board[r-i][c] = 2
                elif board[r-i][c] == 1:
                    break
            # right
            for i in range(strength):
                if c+i < 11 and board[r][c+i] == 0:
                    board[r][c+i] = 2
                elif board[r][c+i] == 1:
                    break
            # left
            for i in range(strength):
                if c-i >=0 and board[r][c-i] == 0:
                    board[r][c-i] = 2
                elif board[r][c-i] == 1:
                    break
        # count emptySafeArea
        emptySafeArea = 0
        for i in range(len(5)):
            for j in range(len(5)):
                newR = pos[0]+i-2
                newC = pos[1]+j-2
                if newR 0 and newR < 11 and newC >= 0 and newC < 11 and calDistance(newR, newC, pos[0], pos[1]) <= 2:
                    if attackMap[newR][newC] == 0:
                        emptySafeArea += 1
        return emptySafeArea

    def calDistance(pos1R, pos1C, pos2R, pos2C):
        return abs(pos1R - pos2R) + abs(pos1C - pos2C)
