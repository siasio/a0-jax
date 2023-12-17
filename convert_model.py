import os.path

import torch
import warnings
from utils import import_class
#from jaxlib import xla_extension
#jax.config.update('jax_array', False)
import jax.numpy as jnp
import pickle
from fire import Fire


def main(
        game_class: str = "games.go_game.GoBoard9x9",
        agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
        root_dir="/home/test/PycharmProjects/a0-jax/",
        new_ckpt_filename: str = "new_ckpt.ckpt",
        old_ckpt_filename: str = "go_agent_9x9_128_sym.ckpt",
        torch_ckpt_filename: str = "torczyk.pt",
        to_torch=False,
):
    """Load agent's weight from disk and start the game."""
    warnings.filterwarnings("ignore")
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    def to_np(d, counter=0):
        print(counter)
        if counter > 20:
            print('Counter above 20!!')
            return
        class_to_search_for = torch.Tensor #xla_extension.DeviceArray if to_torch else torch.Tensor
        function_to_use = torch.tensor if to_torch else jnp.array
        changed_to_text = 'torch' if to_torch else 'jax'
        if isinstance(d, class_to_search_for):
            nparr = function_to_use(d.numpy())
            print(f'Changed to {changed_to_text}')
            return nparr
        if isinstance(d, list):
            return list([to_np(el, counter + 1) for el in d])
        if isinstance(d, tuple):
            return tuple([to_np(el, counter + 1) for el in d])
        if isinstance(d, dict):
            return dict({key: to_np(d[key], counter + 1) for key in d})

    loader = pickle if to_torch else torch
    ckpt_to_convert = old_ckpt_filename if to_torch else torch_ckpt_filename
    with open(os.path.join(root_dir, ckpt_to_convert), "rb") as f:
        sd = torch.load(f)

    if not to_torch:
        sd = sd["agent"]
    sd = to_np(sd)

    converted_ckpt = torch_ckpt_filename if to_torch else new_ckpt_filename
    if to_torch:
        torch.save(sd, converted_ckpt)
    else:
        with open(os.path.join(root_dir, converted_ckpt), "wb") as writer:
            pickle.dump(sd, writer)


if __name__ == "__main__":
    Fire(main)
