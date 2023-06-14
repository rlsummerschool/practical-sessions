"""Environment configurations, maps and transition functions"""

import itertools
import time
from collections import defaultdict
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from minigrid import envs
from minigrid.core.constants import DIR_TO_VEC
from minigrid.minigrid_env import MiniGridEnv

from rlss_practice.wrappers import BinaryReward, DecodeObservation, FailProbability


class MinigridBase(gym.Wrapper):
    """Base class for minigrid environments with explicit transition and reward functions.

    The agent is rewarded upon reaching the goal location.

    Action space:

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |

    Observation space:

    | Name | Description             |
    |------|-------------------------|
    | x    | x coordinate            |
    | y    | y coordinate (downward) |
    | dir  | cardinal direction      |

    The transition function is stored in `T`,
    where `T[state][action][next_state]` is the transition probability.
    The reward function is `R`. `R[state][action]` contains a reward.
    """

    StateT = tuple[int, int, int]
    ActionT = int

    def __init__(self, minigrid: MiniGridEnv, seed: int, failure=0.0):
        """Initialize.

        minigrid: an instantiated minigrid environment.
        seed: random seed
        failure: failure probability of the actions (another action is executed instead).
        """
        # Store
        self.minigrid = minigrid
        self.minigrid.highlight = False
        self.minigrid.action_space = gym.spaces.Discrete(3)
        self.failure = failure
        self.seed = seed

        # Transform and store
        env: gym.Env = FailProbability(self.minigrid, failure=failure, seed=seed)
        env = DecodeObservation(env=env)
        env = BinaryReward(env=env)
        env = TimeLimit(env=env, max_episode_steps=40)
        super().__init__(env=env)

        # The grid must be generated once
        env.reset(seed=self.seed)  # this creates the grid

        def _reset(*, seed=None, options=None):
            return minigrid.gen_obs(), {}

        minigrid.reset = _reset
        self._grid = (
            self.minigrid.grid.encode()
        )  # Just to check that the grid never changes

        # States and actions
        assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        obs_space = env.observation_space.nvec
        self.actions = list(range(env.action_space.n))
        self.states = list(
            itertools.product(*(range(obs_space[i]) for i in range(len(obs_space))))
        )
        self.states = [
            (x, y, o) for (x, y, o) in self.states if self._is_valid_position(x, y)
        ]

        # Explicit transition and rewards functions
        self._compute_model()

    def __str__(self):
        """Simplified to string."""
        OBJECT_TO_STR = {
            "wall": "W",
            "goal": "G",
        }
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    output += AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                tile = self.grid.get(i, j)
                if tile is None:
                    output += " "
                    continue
                output += OBJECT_TO_STR[tile.type]

            if j < self.grid.height - 1:
                output += "\n"

        return output

    def _is_valid_position(self, i: int, j: int) -> bool:
        """Testing whether a coordinate is a valid location."""
        if i < 0:
            return False
        if j < 0:
            return False
        if i >= self.minigrid.width:
            return False
        if j >= self.minigrid.height:
            return False
        cell = self.minigrid.grid.get(i, j)
        if cell is not None and not cell.can_overlap():
            return False
        return True

    def _state_step(self, state: StateT, action: ActionT) -> StateT:
        """Utility to move states one step forward, no side effect."""
        x, y, direction = state

        # Default transition to the sink failure state
        assert self._is_valid_position(x, y)

        # Transition left
        if action == self.minigrid.actions.left:
            direction -= 1
            if direction < 0:
                direction += 4
            return x, y, direction

        # Transition right
        elif action == self.minigrid.actions.right:
            direction = (direction + 1) % 4
            return x, y, direction

        # Transition forward
        elif action == self.minigrid.actions.forward:
            fwd_pos = np.array((x, y)) + DIR_TO_VEC[direction]
            if self._is_valid_position(*fwd_pos):
                return (fwd_pos[0], fwd_pos[1], direction)
            else:
                return state
        # Error
        else:
            assert False, "Invalid action"

    def _compute_model(self):
        """Compute explicit transition and reward functions for this environment."""
        # Compute matrices
        T: dict = defaultdict(lambda: defaultdict())
        R: dict = defaultdict(lambda: defaultdict())
        for state in self.states:
            for action in self.actions:
                # Reward
                pos = self.minigrid.grid.get(state[0], state[1])
                if pos is not None and pos.type == "goal":
                    R[state][action] = 1.0
                else:
                    R[state][action] = 0.0

                # Transition
                success_state = self._state_step(state, action)
                failure_states = [
                    self._state_step(state, a) for a in self.actions if a != action
                ]
                T[state][action] = {
                    s: 1 - self.failure
                    if s == success_state
                    else self.failure / len(failure_states)
                    if s in failure_states
                    else 0.0
                    for s in self.states
                }

            T[state] = dict(T[state])
            R[state] = dict(R[state])
        self.T = dict(T)
        self.R = dict(R)

    def reset(self, **kwargs):
        """Environment reset."""
        ret = super().reset(seed=self.seed, **kwargs)
        assert (
            self.minigrid.grid.encode() == self._grid
        ).all(), "The grid changed: this shouldn't happen"
        return ret

    def _pretty_print_T(self):
        """Prints the positive components of the transition function."""
        print("Transition function -- self.T")
        for state in self.states:
            if self._is_valid_position(state[0], state[1]):
                print(f"State {state}")
                for action in self.actions:
                    print(f"  action {action}")
                    for state2 in self.states:
                        if self.T[state][action][state2] > 0.0:
                            print(
                                f"    next state {state2}: {self.T[state][action][state2]}"
                            )


class Room(MinigridBase):
    """Single room environment with explicit model."""

    def __init__(self, failure=0.0, **kwargs):
        """Initialize.

        failure: failure probability of the actions (another action is executed instead).
        agent_start_pos: tuple with coordinates
        agent_start_dir: north or..
        size: room side length
        """
        minigrid = envs.EmptyEnv(**kwargs)
        super().__init__(minigrid=minigrid, seed=91273192, failure=failure)


class Rooms(MinigridBase):
    """Grid-world with multple rooms and explicit model."""

    def __init__(self, rooms: int, size: int, failure=0.0, **kwargs):
        """Initialize.

        rooms: how many rooms
        size: maximum room size
        failure: failure probability of the actions (another action is executed instead).
        """
        # Initialize
        minigrid = envs.MultiRoomEnv(
            minNumRooms=rooms, maxNumRooms=rooms, maxRoomSize=size, screen_size=1500, **kwargs
        )
        super().__init__(minigrid=minigrid, seed=91273187, failure=failure)

        # Remove doors and keys
        minigrid.grid.grid = [
            obj if obj is None or obj.type in ("wall", "goal") else None
            for obj in minigrid.grid.grid
        ]
        self._grid = self.minigrid.grid.encode()


def test(env: gym.Env, interactive: bool = False):
    """Environment rollouts with uniform policy for visualization.

    env: gym environment to test
    interactive: if True, the user selects the action
    """
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    def log():
        env.render()
        print("Env step")
        print("       Action:", action)
        print("  Observation:", observation)
        print("       Reward:", reward)
        print(
            "         Done:",
            "terminated" if terminated else "truncated" if truncated else "False",
        )
        print("         Info:", info)
        time.sleep(0.1)

    reward: SupportsFloat = 0.0
    action = None
    terminated = False
    truncated = False

    try:
        observation, info = env.reset()
        log()
        while True:
            # Action selection
            action = env.action_space.sample()
            if interactive:
                a = input(f"       Action (default {action}): ")
                if a:
                    action = int(a)
                if action < 0:
                    truncated = True

            # Step
            if action >= 0:
                observation, reward, terminated, truncated, info = env.step(action)
                log()

            # Reset
            if terminated or truncated:
                print("Reset")
                observation, info = env.reset()
                terminated = False
                truncated = False
                reward = 0.0
                log()
    finally:
        env.close()


if __name__ == "__main__":
    env = Rooms(
        failure=0.1,
        rooms=3,
        size=6,
        render_mode="human",
    )
    test(env, interactive=False)
