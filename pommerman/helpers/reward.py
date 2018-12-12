from collections import defaultdict
import queue
import random
from .. import constants
import numpy as np

class Reward:
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
        enemyList = [x.value for x in obs.get('enemies')]
        if self.evadeCondition(obs) == True:
            # return evadeScore(my_position, obs['bomb_blast_strength'], obs['bomb_life'])
            return constants.Mode.Evade
        elif self.attackCondition(obs) == True:
            ''' 0 stand for empty safe position;
                1 stand for blocked by agent, wall, or bomb;
                2 stand for reachable by bomb'''
            # return attackScore(pos, board['board'], obs['bomb_blast_strength'])
            return constants.Mode.Attack
        else:
            return constants.Mode.Explore

    def reward(self, obs, mode):
        myPos = tuple(obs['position'])
        board = np.array(obs['board'])
        if mode == constants.Mode.Evade:
            return self.evadeScore(myPos, obs['bomb_blast_strength'], obs['bomb_life'])
        elif mode == constants.Mode.Attack:
            enemyPos = []
            enemyList = [x.value for x in obs.get('enemies')]
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if board[i][j] in enemyList and self.calDistance(i, j, myPos[0], myPos[1]) <= 4:
                        enemyPos.append(tuple((i,j)))
            maxAttack = 0
            for pos in enemyPos:
                maxAttack = max(maxAttack, self.attackScore(pos, obs))
            return maxAttack

    def evadeCondition(self, obs):
        bombLife = obs['bomb_life']
        pos = tuple(obs['position'])
        bombs = self.convert_bombs(np.array(obs['bomb_blast_strength']))
        bombCnt = 0
        tickCnt = 0
        for bomb in bombs:
            if self.calDistance(bomb['position'][0], bomb['position'][1], pos[0], pos[1]) <= bomb['blast_strength']:
                bombCnt += 1
                tickCnt += bombLife[bomb['position'][0]][bomb['position'][1]]
        if bombCnt == 0:
            return False
        return tickCnt < 5 + 2*bombCnt

    def evadeScore(self, pos, bombStrength, bombLife):
        score = 100
        for i in range(len(bombStrength)):
            for j in range(len(bombStrength[0])):
                if bombStrength[i][j] > 0 and bombStrength[i][j] >= (abs(pos[0]-i) + abs(pos[1]-j)):
                    score = score - (25 * (11 - bombLife[i][j])/10)
        return score

    def attackCondition(self, obs):
        myPos = tuple(obs['position'])
        board = np.array(obs['board'])
        ammo = int(obs['ammo'])
        enemyList = [x.value for x in obs.get('enemies')]
        if ammo == 0: return False
        # this can be improve using bfs
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in enemyList and self.calDistance(i, j, myPos[0], myPos[1]) <= 4:
                    return True
        return False


    def attackScore(self, pos, obs):
        fillArea = self.calFillArea(pos)
        emptySafeArea = self.calEmptySafeArea(pos, obs)
        return 100 * (1 - float(emptySafeArea)/fillArea)

    def calFillArea(self, pos):
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

    def calEmptySafeArea(self, pos, obs):
        board = np.zeros((11,11), dtype=int)
        agentList = []
        for enemy in obs['enemies']:
            agentList.append(enemy.value)
        for i in range(11):
            for j in range(11):
                if obs['board'][i][j] == 1 or obs['board'][i][j] == 2 or obs['board'][i][j] in agentList:
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
        # count emptySafeArea
        emptySafeArea = 0
        for i in range(5):
            for j in range(5):
                newR = pos[0]+i-2
                newC = pos[1]+j-2
                if newR >= 0 and newR < 11 and newC >= 0 and newC < 11 and self.calDistance(newR, newC, pos[0], pos[1]) <= 2:
                    if board[newR][newC] == 0:
                        emptySafeArea += 1
        return emptySafeArea

    def calDistance(self, pos1R, pos1C, pos2R, pos2C):
        return abs(pos1R - pos2R) + abs(pos1C - pos2C)
