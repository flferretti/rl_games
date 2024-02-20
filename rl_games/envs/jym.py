import functools
from typing import List

import gym
import jax
import jaxsim.typing as jtp
import jax.numpy as jnp
import jym_envs
import numpy as np
import torch.utils.dlpack as tpack
from jaxsim import mujoco
from jax._src.dlpack import from_dlpack, to_dlpack
from jym.jax.pytree_space import PyTree
from jym.wrappers.jax import ToNumPyWrapper
from torch import Tensor

from rl_games.common.ivecenv import IVecEnv


def jax_to_torch(tensor: jax.Array):
    tensor = to_dlpack(tensor)
    tensor = tpack.from_dlpack(tensor)
    return tensor.float()


def torch_to_jax(tensor: Tensor):
    tensor = tpack.to_dlpack(tensor.contiguous())
    tensor = from_dlpack(tensor)
    return tensor


class JymEnv(IVecEnv):
    """
    A wrapper around the JymEnv environment to make it compatible with the rl_games framework.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_name, num_actors, log_rewards=False, **kwargs) -> None:
        """"""

        env_name = kwargs.pop("env_name", "cartpole")
        self.seed = kwargs.pop("seed", 0)
        self.batch_size = num_actors

        print(f"Creating {num_actors} environments.")

        self.env = jym_envs.create_jax_env(
            env_name=env_name, batch_size=self.batch_size, seed=self.seed
        )

        single_env_action_space: PyTree = self.env.unwrapped.single_action_space

        single_env_observation_space: PyTree = (
            self.env.unwrapped.single_observation_space
        )

        self.logger_rewards = [] if log_rewards else None

        self.observation_space = single_env_observation_space.to_box()

        self.action_space = single_env_action_space.to_box()

    def reset(self):
        obs, _ = self.env.reset()
        return jax_to_torch(obs)

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def tree_inverse_transpose(pytree: jtp.PyTree, batch_size: int) -> List[jtp.PyTree]:
        return tuple(
            jax.tree_util.tree_map(lambda leaf: leaf[i], pytree)
            for i in range(batch_size)
        )

    def step(self, actions):
        (
            observations,
            rewards,
            terminals,
            truncated,
            step_infos,
        ) = self.env.step(actions=torch_to_jax(actions))

        done = jnp.logical_or(terminals, truncated)

        list_of_step_infos = self.tree_inverse_transpose(
            pytree=step_infos, batch_size=self.env.num_envs
        )

        list_of_step_infos_numpy = [
            ToNumPyWrapper.pytree_to_numpy(pytree=pt) for pt in list_of_step_infos
        ]

        if self.logger_rewards is not None:
            self.logger_rewards.append(np.array(rewards).mean())

        obs = jax_to_torch(observations)
        reward = jax_to_torch(rewards)
        done = jax_to_torch(done)

        return obs, reward, done, list_of_step_infos_numpy

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {
            "action_space": self.env.unwrapped.single_action_space.to_box(),
            "observation_space": self.env.unwrapped.single_observation_space.to_box(),
        }
        return info


def create_jym_env(**kwargs):
    return JymEnv("", kwargs.pop("num_actors", 256), **kwargs)
