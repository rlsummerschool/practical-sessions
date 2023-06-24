"""Various utilities for Environments, mostly in the form of gym.Wrapper."""

import random
from base64 import b64encode
from pathlib import Path
from typing import Optional, SupportsFloat, Union, cast

import gymnasium as gym
import os
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo
from IPython.core.display import HTML, display
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


class BinaryReward(gym.Wrapper):
    """1 if agent is at minigrid goal, 0 otherwise; also, do not terminate trajecotories."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.unwrapped, MiniGridEnv)
        self.minigrid = self.unwrapped

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._last_pos = self.minigrid.agent_pos
        return observation, info

    def step(self, action):
        """Gym step with modified reward and termination"""
        observation, reward, terminated, truncated, info = super().step(action)

        # Were we at goal?
        current_cell = self.minigrid.grid.get(*self._last_pos)
        self._last_pos = self.minigrid.agent_pos
        if current_cell is not None:
            at_goal = current_cell.type == "goal"
        else:
            at_goal = False
        reward = 1.0 if at_goal else 0.0

        return observation, reward, False, truncated, info


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


class Renderer(gym.Wrapper):
    """Record a video and show it back in Jupyter notebook."""

    def __init__(self, env: gym.Env, path: Optional[Union[str, Path]] = None):
        """Initialize.

        env: environments to render.
        path: this can be omitted only in colab, otherwise a path for the recorded video is needed.
        """
        self._video_dir = Path(path) if path else Path("/content/video-out")
        self._video_path_format = self._video_dir / "rl-video-step-{}.mp4"
        super().__init__(
            RecordVideo(
                env=env,
                video_folder=str(self._video_dir),
                step_trigger=lambda step: True,
                disable_logger=True,
            )
        )

    def all_videos(self):
        """Prints all video paths."""
        videos = self._video_path_format.parent.glob("*.mp4")
        print(videos)
    
    def play(self, step: Optional[int] = None):
        """Reproduce a video in Jupyter.

        step: the video step ID (see the output of the video recurder)
            the last video by default
        """
        if step is None:
            videos = self._video_path_format.parent.glob("*.mp4")
            video_path = max(videos, key=os.path.getmtime)
        else:
            video_path = str(self._video_path_format).format(step)

        video = open(video_path, "rb").read()
        data = "data:video/mp4;base64," + b64encode(video).decode()
        display(HTML('<video  controls autoplay> <source src="%s" type="video/mp4"> </video>' % data))
