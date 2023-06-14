"""Environment configurations, maps and transition functions"""

import itertools
import time
from collections import defaultdict
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
from minigrid import envs
from minigrid.core.constants import DIR_TO_VEC

from rlss_practice.wrappers import BinaryReward, DecodeObservation, FailProbability


class Room(gym.Wrapper):
    """An Empty minigrid environment with explicit transition and reward functions.

    The agent is rewarded upon reaching the goal location.

    Action space:

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |

    Observation space:

    |-----|-------------------------|
    | x   | x coordinate            |
    | y   | y coordinate (downward) |
    | dir | cardinal direction      |

    The transition function is stored in `T`,
    where `T[state][action][next_state]` is the transition probability.
    The reward function is `R`. `R[state][action]` contains a reward.
    """

    StateT = tuple[int, int, int]
    ActionT = int

    def __init__(self, seed: int, failure=0.0, **kwargs):
        """Initialize.

        seed: random seed
        failure: failure probability of the actions (another action is executed instead).
        size: room side length
        agent_start_pos: tuple with coordinates
        agent_start_dir: north or..
        """
        # Create minigrid env
        self.minigrid = envs.EmptyEnv(highlight=False, **kwargs)
        self.minigrid.action_space = gym.spaces.Discrete(3)
        self.failure = failure

        # Transform appropriately
        env: gym.Env = FailProbability(self.minigrid, failure=failure, seed=seed)
        env = DecodeObservation(env=env)
        env = BinaryReward(env=env)

        # Sizes
        assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        obs_space = env.observation_space.nvec
        self.states = list(itertools.product(*(range(obs_space[i]) for i in range(len(obs_space)))))
        self.actions = list(range(env.action_space.n))

        # Store and compute functions
        super().__init__(env=env)
        self.reset()                             # This creates a fixed grid and goal
        self._grid = self.minigrid.grid.encode() # Just to check that the grid never changes
        self._compute_model()
        print(self)

    def __str__(self):
        """Simplified to string."""
        OBJECT_TO_STR = {
            "wall": "W",
            "goal": "G",
        }
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = "Room\n"

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
        if i < 0: return False
        if j < 0: return False
        if i >= self.minigrid.width: return False
        if j >= self.minigrid.height: return False
        cell = self.minigrid.grid.get(i, j)
        if cell is not None and not cell.can_overlap(): return False
        return True

    def _state_step(self, state: StateT, action: ActionT) -> StateT:
        """Utility to move states one step forward, no side effect."""
        x, y, direction = state

        # Default transition to the sink failure state
        if not self._is_valid_position(x, y):
            return (0, 0, 0)

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
                failure_states = [self._state_step(state, a) for a in self.actions if a != action]
                T[state][action] = {s: 1 - self.failure if s == success_state
                    else self.failure / len(failure_states) if s in failure_states
                    else 0.0
                    for s in self.states}

            T[state] = dict(T[state])
            R[state] = dict(R[state])
        self.T = dict(T)
        self.R = dict(R)

    def reset(self, **kwargs):
        ret = super().reset(**kwargs)
        if hasattr(self, "_grid"):
            assert (self.minigrid.grid.encode() == self._grid).all(), "The grid changed: this shouldn't happen"
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
                            print(f"    next state {state2}: {self.T[state][action][state2]}")




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
        print("         Done:", "terminated" if terminated else "truncated" if truncated else "False")
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


if __name__ == '__main__':

    env = Room(seed=19823283, failure=0.0, size=5, agent_start_dir=0, agent_start_pos=(1,1), render_mode='human')
    test(env, interactive=False)
