import pathlib
import random
from typing import Dict, OrderedDict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# from gymnasium.wrappers.record_video import RecordVideo as GymnasiumRecordVideo
from gymnasium.experimental.wrappers import RecordVideoV0 as GymnasiumRecordVideo
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete
from gymnasium.spaces.dict import Dict as GymnasiumDict


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


gym_envs = [
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Acrobot-v1",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "LunarLander-v2",
    "LunarLanderContinuous-v2",
]

mujoco_envs = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "InvertedDoublePendulum-v4",
    "InvertedPendulum-v4",
    "Humanoid-v4",
    "HumanoidStandup-v4",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Walker2d-v4",
]


def get_activation(activation_name: str):
    if activation_name == "tanh":
        activation = nn.Tanh
    elif activation_name == "relu":
        activation = nn.ReLU
    elif activation_name == "leakyrelu":
        activation = nn.LeakyReLU
    elif activation_name == "prelu":
        activation = nn.PReLU
    elif activation_name == "gelu":
        activation = nn.GELU
    elif activation_name == "sigmoid":
        activation = nn.Sigmoid
    elif activation_name in [None, "id", "identity", "linear", "none"]:
        activation = nn.Identity
    elif activation_name == "elu":
        activation = nn.ELU
    elif activation_name in ["swish", "silu"]:
        activation = nn.SiLU
    elif activation_name == "softplus":
        activation = nn.Softplus
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation_name)
        )
    return activation


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


def make_gymnasium_env(env_index, env_id, cfg):
    capture_video = cfg.experiment.capture_video

    def thunk():
        # TODO: Atari
        # if "Breakout" in env_id:
        #     env = gym.make(env_id)
        #     env = gym.wrappers.RecordEpisodeStatistics(env)
        #     env = NoopResetEnv(env, noop_max=30)
        #     env = MaxAndSkipEnv(env, skip=4)
        #     env = EpisodicLifeEnv(env)
        #     if "FIRE" in env.unwrapped.get_action_meanings():
        #         env = FireResetEnv(env)
        #     env = ClipRewardEnv(env)
        #     env = gym.wrappers.ResizeObservation(env, (84, 84))
        #     env = gym.wrappers.GrayScaleObservation(env)
        #     env = gym.wrappers.FrameStack(env, 4)
        #     return env
        # else:
        env = gym.make(env_id, render_mode="rgb_array")
        if capture_video and env_index == 0:
            video_path = cfg.paths.video
            print(video_path)
            video_path = pathlib.Path(video_path)
            train_path = pathlib.Path("train")
            video_save_path = str(video_path / train_path / env_id)
            print(f"{env_id}: is being recorded")
            env = GymnasiumRecordVideo(env, video_save_path, disable_logger=True)

        # env = gym.wrappers.TimeLimit(env, cfg.experiment.max_episode_steps)
        # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=cfg.ppo.gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        if isinstance(env.action_space, GymnasiumBox):
            env = gym.wrappers.ClipAction(env)

        return env

    return thunk


class GymnasiumNormalizedFlattenRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, cfg, deque_size=100):
        super().__init__(env)
        if (
            isinstance(env.observation_space, GymnasiumDict)
            or isinstance(env.observation_space, Dict)
            or isinstance(env.observation_space, OrderedDict)
        ):
            size, highs, lows = self.get_obs_space(env.observation_space)
            self.observation_space = GymnasiumBox(low=lows, high=highs)
            # print("observation is changed")
        self.observation_space = GymnasiumBox(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space._shape,
        )
        print(f"gymnasium env observation space: {env.observation_space}")
        print(f"gymnasium env observation space type: {type(env.observation_space)}")
        print(f"gymnasium env action space: {env.action_space}")
        print(f"gymnasium env action space type: {type(env.action_space)}")
        self.action_space = GymnasiumDiscrete(env.action_space.n)

        self.env_type = cfg.experiment.env_type
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.obs_rms = RunningMeanStd(
            shape=(self.num_envs, *self.observation_space.shape)
        )
        self.return_rms = RunningMeanStd(shape=(self.num_envs,))
        self.gamma = 0.98
        self.epsilon = 1e-8

    def normalize_obs(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def normalize_rew(self, rews):
        self.return_rms.update(self.returned_episode_returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

    def reset(self, **kwargs):
        # print(f"Wrapper reset kwargs: {kwargs}")
        observations, infos = super().reset()
        if (
            isinstance(observations, Dict)
            or isinstance(observations, GymnasiumDict)
            or isinstance(observations, OrderedDict)
        ):
            observations = self.flatten_dict(observations)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        observations = self.normalize_obs(observations)
        return observations, infos

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = np.logical_or(terminated, truncated)
        if (
            isinstance(observations, Dict)
            or isinstance(observations, GymnasiumDict)
            or isinstance(observations, OrderedDict)
        ):
            observations = self.flatten_dict(observations)
        self.episode_returns += rewards  # infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones  # infos["terminated"]
        self.episode_lengths *= 1 - dones  # infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        observations = self.normalize_obs(observations)
        rewards = self.normalize_rew(rewards)
        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )

    def flatten_dict(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=-1)
        obs = obs.reshape(self.num_envs, -1)
        return obs

    def get_obs_space(self, observation_space):
        shapes = []
        highs = []
        lows = []
        for _, box in observation_space.items():
            if len(box.shape) == 0:
                shapes.append(1)
                # highs.append()
                # lows.append()
                highs += [*np.expand_dims(box.high, axis=-1)]
                lows += [*np.expand_dims(box.low, axis=-1)]
            elif len(box.shape) == 1:
                shapes += [*box.shape]
                highs += [*box.high]
                lows += [*box.low]
            elif len(box.shape) > 1:
                # print(box)
                shapes += [np.prod(box.shape)]
                highs += [*(box.high.reshape(-1))]
                lows += [*(box.low.reshape(-1))]
        # print(shapes)
        # print(highs)
        # print(lows)
        size = np.sum(shapes, dtype=np.int32)
        highs = np.array(highs)
        lows = np.array(lows)
        return size, highs, lows


class GymNormalizedFlattenRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, cfg, deque_size=100):
        super().__init__(env)
        if (
            isinstance(env.observation_space, GymnasiumDict)
            or isinstance(env.observation_space, Dict)
            or isinstance(env.observation_space, OrderedDict)
        ):
            size, highs, lows = self.get_obs_space(env.observation_space)
            self.observation_space = GymnasiumBox(low=lows, high=highs)
            # print("observation is changed")
        self.env_type = cfg.experiment.env_type
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.obs_rms = RunningMeanStd(
            shape=(self.num_envs, *self.observation_space.shape)
        )
        self.return_rms = RunningMeanStd(shape=(self.num_envs,))
        self.gamma = 0.98
        self.epsilon = 1e-8

    def normalize_obs(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def normalize_rew(self, rews):
        self.return_rms.update(self.returned_episode_returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

    def reset(self, **kwargs):
        print(f"kwargs: {kwargs}")
        observations = super().reset(**kwargs)
        if (
            isinstance(observations, Dict)
            or isinstance(observations, GymnasiumDict)
            or isinstance(observations, OrderedDict)
        ):
            observations = self.flatten_dict(observations)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        observations = self.normalize_obs(observations)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        if (
            isinstance(observations, Dict)
            or isinstance(observations, GymnasiumDict)
            or isinstance(observations, OrderedDict)
        ):
            observations = self.flatten_dict(observations)
        self.episode_returns += rewards  # infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones  # infos["terminated"]
        self.episode_lengths *= 1 - dones  # infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        observations = self.normalize_obs(observations)
        rewards = self.normalize_rew(rewards)
        return (
            observations,
            rewards,
            dones,
            infos,
        )

    def flatten_dict(self, obs):
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        obs = np.concatenate(obs_pieces, axis=-1)
        obs = obs.reshape(self.num_envs, -1)
        return obs

    def get_obs_space(self, observation_space):
        shapes = []
        highs = []
        lows = []
        for _, box in observation_space.items():
            if len(box.shape) == 0:
                shapes.append(1)
                highs += [*np.expand_dims(box.high, axis=-1)]
                lows += [*np.expand_dims(box.low, axis=-1)]
            elif len(box.shape) == 1:
                shapes += [*box.shape]
                highs += [*box.high]
                lows += [*box.low]
            elif len(box.shape) > 1:
                shapes += [np.prod(box.shape)]
                highs += [*(box.high.reshape(-1))]
                lows += [*(box.low.reshape(-1))]
        size = np.sum(shapes, dtype=np.int32)
        highs = np.array(highs)
        lows = np.array(lows)
        return size, highs, lows
