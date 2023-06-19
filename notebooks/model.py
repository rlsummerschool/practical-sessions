#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21, 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

from agent import Agent
from display import display_position, display_board


class Environment:
    """Generic environment.
    Reward only depends on the next state.
    """
    def __init__(self):
        self.state = self.init_state()

    def init_state(self):
        state = None
        return state

    def reinit_state(self):
        self.state = self.init_state()  
    
    @staticmethod
    def get_actions(state):
        actions = [None]
        return actions
    
    @staticmethod
    def get_states():
        raise ValueError("Not available. The state space might be too large.")
    
    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [deepcopy(state)]
        return probs, states

    @staticmethod
    def get_reward(state):
        return None

    @staticmethod
    def is_terminal(state):
        return True

    @staticmethod
    def encode(state):
        return state
    
    @staticmethod
    def decode(state):
        return state
    
    @staticmethod
    def get_model(state, action):
        probs, states = Environment.get_transition(state, action)
        rewards = [Environment.get_reward(state) for state in states]
        return probs, states, rewards
    
    def step(self, action):
        """Apply action, get reward and modify state.
        Returns reward, stop (``True`` if terminal state).
        """
        reward = None
        stop = True
        if action is not None and action in self.get_actions(self.state):
            probs, states, rewards = self.get_model(self.state, action)
            i = np.random.choice(len(probs), p=probs)
            state = states[i]
            self.state = state
            reward = rewards[i]
            stop = self.is_terminal(state)
        return reward, stop


class Walk(Environment):
    """1D Walk."""

    Length = 10
    Reward_States = [2, 8]
    Reward_Values = [1, 2]

    def __init__(self):
        super(Walk, self).__init__()    

    @classmethod
    def set_parameters(cls, length, reward_states, reward_values):
        cls.Length = length
        cls.Reward_States = reward_states
        cls.Reward_Values = reward_values

    def init_state(self):
        return np.random.choice(Walk.Length)

    @staticmethod
    def get_states():
        states = np.arange(Walk.Length)
        return states
    
    @staticmethod
    def get_actions(state=None):
        actions = [1, -1]
        if state is not None:
            if state == Walk.Length - 1:
                actions = [-1]
            if state == 0:
                actions = [1]
        return actions

    @staticmethod
    def get_transition(state, action):
        state += action
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        reward = 0
        if state in Walk.Reward_States:
            reward = Walk.Reward_Values[Walk.Reward_States.index(state)]
        return reward

    @staticmethod
    def is_terminal(state):
        return False

    @staticmethod
    def get_model(state, action):
        probs, states = Walk.get_transition(state, action)
        rewards = [Walk.get_reward(state) for state in states]
        return probs, states, rewards

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        image = 200 * np.ones((1, Walk.Length, 3)).astype(int)
        if states is not None:
            positions = [(0, state) for state in states]
        else:
            positions = None
        position = (0, self.state)
        return display_position(image, position, positions, marker, marker_size, marker_color, interval)

    @staticmethod
    def display_values(values):
        values_scaled = np.array(values, dtype=float)
        values_scaled -= np.min(values_scaled)
        if np.max(values_scaled) > 0:
            values_scaled /= np.max(values_scaled)
        image = values_scaled.reshape(1,-1)
        plt.imshow(image, cmap='gray');
        plt.axis('off')

    def display_policy(self, policy):
        states = self.get_states()
        image = np.ones((1, len(states)))
        plt.imshow(image, cmap='gray', alpha=0.2)
        for state in states:
            _, actions = policy(state)
            action = actions[0]
            plt.arrow(state, 0, action, 0, color='r', width=0.15, length_includes_head=True)
        plt.axis('off')
        
    
class Maze(Environment):
    """Maze."""

    Map = np.ones((2, 2)).astype(int)
    Start_State = (0, 0)
    Exit_States = [(1, 1)]

    def __init__(self):
        super(Maze, self).__init__()

    @classmethod
    def set_parameters(cls, maze_map, start_state, exit_states):
        cls.Map = maze_map
        cls.Start_State = start_state
        cls.Exit_States = exit_states

    def init_state(self):
        return np.array(Maze.Start_State)

    @staticmethod
    def is_valid(state):
        n, m = Maze.Map.shape
        x, y = tuple(state)
        return 0 <= x < n and 0 <= y < m and Maze.Map[x, y]
    
    @staticmethod
    def get_states():
        n, m = Maze.Map.shape
        states = [np.array([x,y]) for x in range(n) for y in range(m) if Maze.is_valid(np.array([x,y]))]
        return states
    
    @staticmethod
    def encode(state):
        return tuple(state)
    
    @staticmethod
    def decode(state):
        return np.array(state)

    @staticmethod
    def get_actions(state=None):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if state is not None:
            actions = []
            for move in moves:
                if Maze.is_valid(state + move):
                    actions.append(move)
        else:
            actions = moves.copy()
        return actions

    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [state.copy() + action]
        return probs, states

    @staticmethod
    def get_reward(state):
        return -1

    @staticmethod
    def is_terminal(state):
        return tuple(state) in Maze.Exit_States

    @staticmethod
    def get_model(state, action):
        probs, states = Maze.get_transition(state, action)
        rewards = [Maze.get_reward(state) for state in states]
        return probs, states, rewards

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        shape = (*Maze.Map.shape, 3)
        image = np.zeros(shape).astype(int)
        for i in range(3):
            image[:, :, i] = 255 * Maze.Map
        return display_position(image, self.state, states, marker, marker_size, marker_color, interval)

    def display_values(self, values):
        image = np.zeros(self.Map.shape)
        values_scaled = np.array(values)
        values_scaled /= np.min(values)
        if np.max(values_scaled) > 0:
            values_scaled /= np.max(values_scaled)
        states = self.get_states()
        for state, value in zip(states, values_scaled):
            image[tuple(state)] = 1 - 0.8 * value 
            plt.imshow(image, cmap='gray');
            plt.axis('off')

    def display_policy(self, policy):
        image = np.zeros(self.Map.shape)
        states = self.get_states()
        for state in states:
            image[tuple(state)] = 1
        plt.imshow(image, cmap='gray')
        for state in states:
            if not self.is_terminal(state):
                _, actions = policy(state)
                action = actions[0]
                if action == (0,0):
                    plt.scatter(state[1], state[0], s=100, c='b')
                else:
                    plt.arrow(state[1], state[0] , action[1], action[0], color='r', width=0.15, length_includes_head=True)
        plt.axis('off')
        

class TicTacToe(Environment):
    """Tic-tac-toe game."""

    def __init__(self, play_first=True, adversary_policy=None):
        if play_first:
            self.first_player = 1
        else:
            self.first_player = -1
        self.adversary = Agent(self, adversary_policy)
        super(TicTacToe, self).__init__()

    def init_state(self):
        board = np.zeros((3, 3)).astype(int)
        return [self.first_player, board]

    @staticmethod
    def encode(state):
        player, board = state
        if player == 1:
            return b'\x01' + board.tobytes()
        else:
            return b'\x00' + board.tobytes()
        
    @staticmethod
    def decode(state_code):
        player_code = state_code[0]
        if player_code:
            player = 1
        else:
            player = -1
        board_code = state_code[1:]
        board = np.frombuffer(board_code, dtype=int).reshape((3,3))
        return player, board

    @staticmethod
    def get_actions(state=None, player=None):
        actions = [None]
        if state is not None:
            current_player, board = state
            if player is None or player == current_player:
                x_, y_ = np.where(board == 0)
                actions = [(x, y) for x, y in zip(x_, y_)]
        return actions
    
    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        board[action] = player
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        _, board = state
        sums = set(board.sum(axis=0)) | set(board.sum(axis=1))
        sums.add(board.diagonal().sum())
        sums.add(np.fliplr(board).diagonal().sum())
        if 3 in sums:
            reward = 1
        elif -3 in sums:
            reward = -1
        else:
            reward = 0
        return reward

    @staticmethod
    def is_terminal(state):
        return bool(TicTacToe.get_reward(state)) or not len(TicTacToe.get_actions(state))

    @staticmethod
    def get_model(state, action):
        probs, states = TicTacToe.get_transition(state, action)
        rewards = [TicTacToe.get_reward(state) for state in states]
        return probs, states, rewards

    def step(self, action=None):
        player, board = self.state
        if player == -1:
            # flip player roles 
            state = 1, -board
            action = self.adversary.get_action(state)
        return Environment.step(self, action)

    def display(self, states=None, marker1='X', marker2='o', marker_size=2000, color1='b', color2='r', interval=300):
        image = 200 * np.ones((3, 3, 3)).astype(int)
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class ConnectFour(Environment):
    """Connect Four game."""

    def __init__(self, play_first=True, adversary_policy=None):
        if play_first:
            self.first_player = 1
        else:
            self.first_player = -1
        self.adversary = Agent(self, adversary_policy)
        super(ConnectFour, self).__init__()

    def init_state(self):
        board = np.zeros((6, 7)).astype(int)
        state = [self.first_player, board]
        return state

    @staticmethod
    def encode(state):
        player, board = state
        if player == 1:
            return b'\x01' + board.tobytes()
        else:
            return b'\x00' + board.tobytes()
        
    @staticmethod
    def decode(state_code):
        player_code = state_code[0]
        if player_code:
            player = 1
        else:
            player = -1
        board_code = state_code[1:]
        board = np.frombuffer(board_code, dtype=int).reshape((6, 7))
        return player, board
        
    @staticmethod
    def get_actions(state=None, player=None):
        actions = [None]
        if state is not None:
            current_player, board = state
            if player is None or player == current_player:
                actions = np.argwhere(board[0] == 0).ravel()
        return actions

    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        row = np.argwhere(board[:, action] == 0).ravel()[-1]
        board[row, action] = player
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        _, board = state
        sep = ','
        sequence = np.array2string(board, separator=sep)
        sequence += np.array2string(board.T, separator=sep)
        sequence += ''.join([np.array2string(board.diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        sequence += ''.join([np.array2string(np.fliplr(board).diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        pattern_pos = sep.join(4 * [' 1'])
        pattern_neg = sep.join(4 * ['-1'])
        if pattern_pos in sequence:
            reward = 1
        elif pattern_neg in sequence:
            reward = -1
        else:
            reward = 0
        return reward

    @staticmethod
    def is_terminal(state):
        return bool(ConnectFour.get_reward(state)) or not len(ConnectFour.get_actions(state))

    @staticmethod
    def get_model(state, action):
        probs, states = ConnectFour.get_transition(state, action)
        rewards = [ConnectFour.get_reward(state) for state in states]
        return probs, states, rewards

    def step(self, action=None):
        player, board = self.state
        if player == -1:
            # flip player roles 
            state = 1, -board
            action = self.adversary.get_action(state)
        return Environment.step(self, action)

    def display(self, states=None, marker1='o', marker2='o', marker_size=1000, color1='gold', color2='r', interval=200):
        image = np.zeros((6, 7, 3)).astype(int)
        image[:, :, 2] = 255
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class Nim(Environment):
    """Nim game."""

    Init_State = [1, 3, 5, 7]

    @classmethod
    def set_init_state(cls, init_state):
        cls.Init_State = init_state

    def __init__(self, play_first=True, adversary_policy=None):
        if play_first:
            self.first_player = 1
        else:
            self.first_player = -1
        self.adversary = Agent(self, adversary_policy)
        super(Nim, self).__init__()

    def init_state(self):
        board = np.array(Nim.Init_State).astype(int)
        state = [self.first_player, board]
        return state

    @staticmethod
    def encode(state):
        player, board = state
        if player == 1:
            return b'\x01' + board.tobytes()
        else:
            return b'\x00' + board.tobytes()
        
    @staticmethod
    def decode(state_code):
        player_code = state_code[0]
        if player_code:
            player = 1
        else:
            player = -1
        board_code = state_code[1:]
        board = np.frombuffer(board_code, dtype=int).reshape((4,))
        return player, board
    
    @staticmethod
    def get_actions(state=None, player=None):
        actions = [None]
        if state is not None:
            current_player, board = state
            if player is None or player == current_player:
                rows = np.where(board)[0]
                actions = [(row, number + 1) for row in rows for number in range(board[row])]
        return actions

    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        row, number = action
        board[row] -= number
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        player, board = state
        if np.sum(board) > 0:
            reward = 0
        else:
            reward = player
        return reward

    @staticmethod
    def is_terminal(state):
        _, board = state
        return not np.sum(board)

    @staticmethod
    def get_model(state, action):
        probs, states = Nim.get_transition(state, action)
        rewards = [Nim.get_reward(state) for state in states]
        return probs, states, rewards

    def step(self, action=None):
        player, board = self.state
        if player == -1:
            # flip player roles 
            state = 1, board
            action = self.adversary.get_action(state)
        return Environment.step(self, action)

    def display(self, states=None, marker='d', marker_size=500, color='gold', interval=200):
        board = np.array(Nim.Init_State).astype(int)
        image = np.zeros((len(board), np.max(board), 3)).astype(int)
        image[:, :, 1] = 135
        if states is not None:
            positions = []
            for _, board in states:
                x = []
                y = []
                for row in np.where(board)[0]:
                    for col in range(board[row]):
                        x.append(row)
                        y.append(col)
                positions.append((x, y))
        else:
            positions = None
        _, board = self.state
        x = []
        y = []
        for row in np.where(board)[0]:
            for col in range(board[row]):
                x.append(row)
                y.append(col)
        position = x, y
        return display_position(image, position, positions, marker, marker_size, color, interval)
