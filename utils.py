"""Useful functions."""

import importlib
from functools import partial
from typing import Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
import pax
import numpy as np

from games.env import Enviroment as E

T = TypeVar("T")


@pax.pure
def batched_policy(agent, states):
    """Apply a policy to a batch of states.

    Also return the updated agent.
    """
    return agent, agent(states, batched=True)


def replicate(value: T, repeat: int) -> T:
    """Replicate along the first axis."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * repeat), value)


@pax.pure
def reset_env(env: E) -> E:
    """Return a reset enviroment."""
    env.reset()
    return env


@jax.jit
def env_step(env: E, action: chex.Array) -> Tuple[E, chex.Array]:
    """Execute one step in the enviroment."""
    env, reward = env.step(action)
    return env, reward


def import_class(path: str) -> E:
    """Import a class from a python file.

    For example:
    >> Game = import_class("connect_two_game.Connect2Game")

    Game is the Connect2Game class from `connection_two_game.py`.
    """
    names = path.split(".")
    mod_path, class_name = names[:-1], names[-1]
    mod = importlib.import_module(".".join(mod_path))
    return getattr(mod, class_name)


def select_tree(pred: jnp.ndarray, a, b):
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_util.tree_map(partial(jax.lax.select, pred), a, b)


num_recent_positions = 8


def stack_last_state(state: np.ndarray):
    stacked = np.stack([state] * num_recent_positions)
    stacked = np.concatenate((stacked, np.zeros_like(state)[None]))
    return np.moveaxis(stacked, 0, -1)


def add_new_state(stacked_pos: np.ndarray, state: jnp.ndarray, color_to_play: int):
    moved_axis = np.moveaxis(stacked_pos, -1, 0)
    new_stacked_pos = np.concatenate((moved_axis[1:-1], state[None]))
    new_stacked_pos = np.concatenate((new_stacked_pos, np.ones_like(state)[None]))
    # TODO: Understand these multiplications
    new_stacked_pos = new_stacked_pos * color_to_play
    return np.moveaxis(new_stacked_pos, 0, -1)


def add_new_stack_previous_state(stacked_pos: np.ndarray, state: np.ndarray, color_to_play: int):
    moved_axis = np.moveaxis(stacked_pos, -1, 0)
    previous_state = np.stack([moved_axis[-2]] * (num_recent_positions - 1))
    new_stacked_pos = np.concatenate((previous_state, state[None]))
    new_stacked_pos = np.concatenate((new_stacked_pos, np.ones_like(state)[None]))
    new_stacked_pos = new_stacked_pos * color_to_play
    return np.moveaxis(new_stacked_pos, 0, -1)
