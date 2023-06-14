"""Various utilities for Environments, mostly in the form of gym.Wrapper."""

import random
from typing import SupportsFloat, cast

import gymnasium as gym
import numpy as np
from minigrid.minigrid_env import MiniGridEnv


class DecodeObservation(gym.ObservationWrapper):
    """Decoded observation for minigrid.

    The observation is composed of agent 2D position and orientation.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.unwrapped, MiniGridEnv)
        self.minigrid = self.unwrapped
        self.observation_space = gym.spaces.MultiDiscrete(
            [self.minigrid.grid.height, self.minigrid.grid.width, 4], np.int_
        )

    def observation(self, observation: dict) -> np.ndarray:
        """Transform observation."""
        obs = (*self.minigrid.agent_pos, self.minigrid.agent_dir)
        return np.array(obs, dtype=np.int32)


class BinaryReward(gym.RewardWrapper):
    """1 if agent is at minigrid goal, 0 otherwise."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.unwrapped, MiniGridEnv)
        self.minigrid = self.unwrapped
        self._was_at_goal = False

    def reset(self, **kwargs):
        """Reset."""
        self._was_at_goal = False
        return super().reset(**kwargs)

    def reward(self, reward: SupportsFloat) -> float:
        """Compute reward."""
        current_cell = self.minigrid.grid.get(*self.minigrid.agent_pos)
        if current_cell is not None:
            at_goal = current_cell.type == "goal"
        else:
            at_goal = False
        rew = 1.0 if at_goal and not self._was_at_goal else 0.0
        self._was_at_goal = at_goal
        return rew


class FailProbability(gym.Wrapper):
    """Causes input actions to fail with some probability: a different action is executed."""

    def __init__(self, env: gym.Env, failure: float, seed: int, **kwargs):
        """Initialize."""
        super().__init__(env, **kwargs)
        self.failure = failure
        assert 0 <= self.failure <= 1
        self._n = int(cast(gym.spaces.Discrete, env.action_space).n)
        self._rng = random.Random(seed)

    def step(self, action):
        """Env step."""
        # Random?
        if self._rng.random() < self.failure:
            action = self._rng.randint(0, self._n - 1)
        return self.env.step(action)
