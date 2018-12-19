'''The base simple agent use to train agents.
This agent is also the benchmark for other agents.
'''
from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent
from .. import constants
from .. import utility


class SimpleTeamAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """

    def __init__(self, *args, **kwargs):
        super(SimpleTeamAgent, self).__init__(*args, **kwargs)

        # Keep track of recently visited uninteresting positions so that we
        # don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        # Keep track of the previous direction to help with the enemy standoffs.
        self._prev_direction = None

    def act(self, obs, action_space):
        def convert_bombs(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                })
            return ret

        # 20181218
        def convert_flames(_board, bomb_map):
            locations = np.where(bomb_map > 0)
            ret = []  # np.array([[0 for j in range(11)] for i in range(11)])
            is_flame = []
            for r, c in zip(locations[0], locations[1]):
                blast_str = bomb_map[(r, c)]
                if not (r, c) in is_flame:
                    ret.append({
                        'position': (r, c),
                        'blast_strength': 1  # is_flame
                    })
                for _dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for _i in range(1, int(blast_str)):
                        _pos = (r + _dir[0] * _i, c + _dir[1] * _i)
                        if _pos in is_flame:
                            continue
                        if utility.position_on_board(_board, _pos):
                            if board[_pos] != constants.Item.Rigid and board[_pos] != constants.Item.Wood:
                                ret.append({
                                    'position': _pos,
                                    'blast_strength': 1
                                })
                            else:
                                break
                        else:
                            break
            return ret

        # 20181218
        # more aggressive flame map
        def convert_flames2(_board, bomb_map, bomb_life):
            locations = np.where(bomb_map > 0)
            tmp_ret = {}
            is_flame = []
            for r, c in zip(locations[0], locations[1]):
                blast_str = bomb_map[(r, c)]
                if not (r, c) in is_flame:
                    tmp_ret[(r, c)] = {
                        'position': (r, c),
                        'flame_time': bomb_life[(r, c)]  # is_flame
                    }
                    is_flame.append((r, c))
                else:
                    if bomb_life[(r, c)] < tmp_ret[(r, c)]['flame_time']:
                        tmp_ret[(r, c)]['flame_time'] = bomb_life[(r, c)]
                for _dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for _i in range(1, int(blast_str)):
                        _pos = (r + _dir[0] * _i, c + _dir[1] * _i)
                        if _pos in is_flame:
                            if bomb_life[_pos] < tmp_ret[_pos]['flame_time']:
                                tmp_ret[_pos]['flame_time'] = bomb_life[_pos]
                            continue
                        if utility.position_on_board(_board, _pos):
                            if board[_pos] != constants.Item.Rigid and board[_pos] != constants.Item.Wood:
                                tmp_ret[_pos] = {
                                    'position': _pos,
                                    'flame_time': bomb_life[_pos]
                                }
                                is_flame.append(_pos)
                            else:
                                break
                        else:
                            break
            # tmp_ret to ret
            ret = []
            for tm in tmp_ret:
                ret.append(tmp_ret[tm])
            return ret

        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        enemies = [constants.Item(e) for e in obs['enemies']]
        # 20181208
        teammate = constants.Item(obs['teammate']) # fix it
        tm_value = teammate.value
        # tm_position = np.where(board == teammate.value)
        tm_position = False
        tm_coordinates = (-1, -1)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[(i, j)] == tm_value:
                    tm_position = True
                    tm_coordinates = (i, j)
        # clean discarded code //20181218
        enemies2 = enemies.copy()
        enemies2.append(teammate)

        # 20181218
        flames = convert_flames(board, np.array(obs['bomb_blast_strength']))
        flames2 = convert_flames2(board, np.array(obs['bomb_blast_strength']),
                                  np.array(obs['bomb_life']))
        dang_move = []
        for _m in [constants.Action.Up, constants.Action.Down,
                   constants.Action.Left, constants.Action.Right]:
            _d = (int(_m.value < 3)*(_m.value-1.5)*2, int(_m.value > 2)*(_m.value-3.5)*2)
            new_pos = (my_position[0]+_d[0], my_position[1]+_d[1])
            if {'position': new_pos, 'blast_strength': 1} in flames:
                dang_move.append(_m)

        ammo = int(obs['ammo'])
        blast_strength = int(obs['blast_strength'])
        
        # 20181208
        items, dist, prev = self._djikstra(
            board, my_position, bombs, enemies2, depth=10)

        # Move if we are in an unsafe place.
        unsafe_directions = self._directions_in_range_of_bomb(
            board, my_position, bombs, dist)
        if unsafe_directions:
            # 20181208
            directions = self._find_safe_directions(
                board, my_position, unsafe_directions, bombs, enemies2)

            # 20181208
            if directions[0] == constants.Action.Stop and not tm_position:
                directions = [constants.Action.Bomb]  # .append(constants.Action.Bomb)
            elif directions[0] == constants.Action.Stop:
                if self._maybe_bomb(ammo, blast_strength, items, dist, tm_coordinates):
                    directions = [constants.Action.Bomb]

            return random.choice(directions).value

        # Lay pomme if we are adjacent to an enemy.
        if self._is_adjacent_enemy(items, dist, enemies) and self._maybe_bomb(
                ammo, blast_strength, items, dist, my_position):
            
            # 20181208
            if not tm_position:
                return constants.Action.Bomb.value
            else:
                if self._maybe_bomb(ammo, blast_strength, items, dist, tm_coordinates):
                    return constants.Action.Bomb.value

        # Move towards an enemy if there is one in exactly three reachable spaces.
        direction = self._near_enemy(my_position, items, dist, prev, enemies, 3)  # 3 -> 5 ->3
        # 20181218 remove dangerous moves
        #if direction is not None:
        #    for _dm in dang_move:
        #        if _dm == direction:
        #            direction = None  #.remove(_dm)
        if direction in dang_move:
            direction = None

        if direction is not None and (self._prev_direction != direction or
                                      random.random() < .5):
            self._prev_direction = direction
            return direction.value

        # Move towards a good item if there is one within two reachable spaces.
        direction = self._near_good_powerup(my_position, items, dist, prev, 3)  # 2 -> 3
        # 20181218 remove dangerous moves
        if direction in dang_move:
            direction = None

        if direction is not None:
            return direction.value

        # Maybe lay a bomb if we are within a space of a wooden wall.
        if self._near_wood(my_position, items, dist, prev, blast_strength - 1): # 1 -> blast_strength
            if self._maybe_bomb(ammo, blast_strength, items, dist, my_position):
                if not tm_position:
                    return constants.Action.Bomb.value
                else:
                    if self._maybe_bomb(ammo, blast_strength, items, dist, tm_coordinates):
                        return constants.Action.Bomb.value
                    return constants.Action.Stop.value
            else:
                return constants.Action.Stop.value

        # Move towards a wooden wall if there is one within two reachable spaces and you have a bomb.
        direction = self._near_wood(my_position, items, dist, prev, 4) # 2 -> 4
        # 20181218 remove dangerous moves
        if direction in dang_move:
            direction = None

        if direction is not None:
            directions = self._filter_unsafe_directions(board, my_position,
                                                        [direction], bombs)
            if directions:
                return directions[0].value

        # Choose a random but valid direction.
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]
        valid_directions = self._filter_invalid_directions(
            board, my_position, directions, enemies2)
        directions = self._filter_unsafe_directions(board, my_position,
                                                    valid_directions, bombs)
        directions = self._filter_recently_visited(
            directions, my_position, self._recently_visited_positions)
        if len(directions) > 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        # Add this position to the recently visited uninteresting positions so we don't return immediately.
        self._recently_visited_positions.append(my_position)
        self._recently_visited_positions = self._recently_visited_positions[
            -self._recently_visited_length:]

        # 20181218
        # remove dangerous moves
        for _dm in dang_move:
            if _dm in directions:
                directions.remove(_dm)
        if len(directions) == 0:
            directions = [constants.Action.Stop]

        return random.choice(directions).value

    @staticmethod
    def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1, p_2):
            '''Determines if two points are out of rang of each other'''
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.Queue()

        my_x, my_y = my_position
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
                position = (r, c)
                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude),
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)
                
                if position == my_position:
                    Q.put(position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf

        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            position = Q.get()

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                val = dist[(x, y)] + 1
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + x, col + y)
                    if new_position not in dist:
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position
                        Q.put(new_position)
                    elif (val == dist[new_position] and random.random() < .5):
                        dist[new_position] = val
                        prev[new_position] = position   

        return items, dist, prev

    def _directions_in_range_of_bomb(self, board, my_position, bombs, dist):
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                        constants.Action.Right,
                        constants.Action.Left,
                        constants.Action.Up,
                        constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])
            elif x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _find_safe_directions(self, board, my_position, unsafe_directions,
                              bombs, enemies):

        def is_stuck_direction(next_position, bomb_range, next_board, enemies):
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board,
                                                        new_position, enemies):
                        continue

                    dist = abs(row + position_x - next_x) + abs(col + position_y - next_y)
                    Q.put((dist, new_position))
            return is_stuck

        # All directions are unsafe. Return a position that won't leave us locked.
        safe = []

        # safe.append(constants.Action.Bomb)

        # TODO1: can_kick and lay_bomb
        # TODO2: find the nearest safe position
        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(
                    my_position, direction)
                next_x, next_y = next_position
                if not utility.position_on_board(next_board, next_position) or \
                   not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board,
                                          enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []  # The directions that will go off the board.

        for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row, y + col)
            direction = utility.get_direction(my_position, position)

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position,
                                            enemies) or utility.position_is_fog(
                                                board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    @staticmethod
    def _maybe_bomb(ammo, blast_strength, items, dist, my_position):
        """Returns whether we can safely bomb right now.

        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] == np.inf:
                continue

            # We can reach a passage that's outside of the bomb strength.
            if dist[position] > blast_strength:
                return True

            # We can reach a passage that's outside of the bomb scope.
            position_x, position_y = position
            if position_x != x and position_y != y:
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        dist_to = max(dist.values())

        for obj in objs:
            for position in items.get(obj, []):
                d = dist[position]
                if d <= radius and d <= dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    @staticmethod
    def _get_direction_towards_position(my_position, position, prev):
        if not position:
            return None

        next_position = position
        while prev[next_position] != my_position:
            next_position = prev[next_position]

        return utility.get_direction(my_position, next_position)

    @classmethod
    def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
        nearest_enemy_position = cls._nearest_position(dist, enemies, items,
                                                       radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_enemy_position, prev)

    @classmethod
    def _near_good_powerup(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_wood(cls, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                        board, position, enemies):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_unsafe_directions(board, my_position, directions, bombs):
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bomb_x, bomb_y = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bomb_x and abs(bomb_y - y) <= blast_strength) or \
                   (y == bomb_y and abs(bomb_x - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret
