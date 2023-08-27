"""
AlphaZero training script.

Train agent by self-play only.
"""
import datetime
import json
import os
import pickle
import random
import shutil
import zipfile
from functools import partial
from typing import Optional

import chex
import click
import cloudpickle
import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import numpy as np
import opax
import optax
import pax

from games.env import Enviroment
from play import PlayResults, agent_vs_agent_multiple_games
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import batched_policy, env_step, import_class, replicate, reset_env, stack_last_state
from policies.resnet_policy import TransferResnet, ResnetPolicyValueNet128
from local_pos_masks import AnalyzedPosition
from typing import Tuple
import datetime

EPSILON = 1e-9  # a very small positive value
TRAIN_DIR = "zip_logs"
TEST_DIR = "test_dir"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.79'


@chex.dataclass(frozen=True)
class TrainingExample:
    """AlphaZero training example.

    state: the current state of the game.
    action_weights: the target action probabilities from MCTS policy.
    value: the target value from self-play result.
    """

    state: chex.Array
    action_weights: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class TrainingOwnershipExample:
    """AlphaZero training example.

    state: the current state of the game.
    DEPRECATED move: the next move played locally, one-hot-encoded, shape: [2 * num_actions + 1].
    mask: the binary mask of the local position which the network should evaluate.
    board_mask: the binary mask of the board on which the game was played (all boards are padded to 19x19).
    value: the expected ownership per intersection.
    """

    state: chex.Array
    color_to_move: chex.Numeric
    has_next_move: chex.Numeric
    next_move_coords: chex.Array
    next_move_color: chex.Numeric
    mask: chex.Array
    board_mask: chex.Array
    value: chex.Array
    # TODO: Think whether concatenating board mask to backbone output is the best solution

    @classmethod
    def from_obj(cls, obj: 'TrainingOwnershipExample', **kwargs):
        init_fields = [f.name for f in cls.__dataclass_fields__.values() if f.init]
        updated_fields = {field: kwargs.get(field, getattr(obj, field)) for field in init_fields}
        return cls(**updated_fields)  # type : ignore - why does it not work?


@chex.dataclass(frozen=True)
class TrainingOwnershipDatapoint:
    """AlphaZero training example.

    state: the current state of the game.
    move: the next move played locally, one-hot-encoded, shape: [2 * num_actions + 1].
    mask: the binary mask of the local position which the network should evaluate.
    board_mask: the binary mask of the board on which the game was played (all boards are padded to 19x19).
    value: the expected ownership per intersection.
    """

    state: chex.Array
    color_to_move: chex.Numeric
    move: chex.Array
    mask: chex.Array
    board_mask: chex.Array
    value: chex.Array

    @classmethod
    def from_obj(cls, obj: 'TrainingOwnershipExample', **kwargs):
        init_fields = [f.name for f in cls.__dataclass_fields__.values() if f.init and f.name != 'move']
        updated_fields = {field: kwargs.get(field, getattr(obj, field)) for field in init_fields}
        move = construct_move_target(obj.has_next_move, obj.next_move_coords, obj.next_move_color)
        updated_fields['move'] = move
        return cls(**updated_fields)  # type : ignore - why does it not work?


@chex.dataclass(frozen=True)
class MoveOutput:
    """The output of a single self-play move.

    state: the current state of game.
    reward: the reward after execute the action from MCTS policy.
    terminated: the current state is a terminated state (bad state).
    action_weights: the action probabilities from MCTS policy.
    """

    state: chex.Array
    reward: chex.Array
    terminated: chex.Array
    action_weights: chex.Array


@partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=(3, 4))
def collect_batched_self_play_data(
    agent,
    env: Enviroment,
    rng_key: chex.Array,
    batch_size: int,
    num_simulations_per_move: int,
):
    """Collect a batch of self-play data using mcts."""

    def single_move(prev, inputs):
        """Execute one self-play move using MCTS.

        This function is designed to be compatible with jax.scan.
        """
        env, rng_key, step = prev
        del inputs
        rng_key, rng_key_next = jax.random.split(rng_key, 2)
        state = jax.vmap(lambda e: e.canonical_observation())(env)
        terminated = env.is_terminated()
        policy_output = improve_policy_with_mcts(
            agent,
            env,
            rng_key,
            recurrent_fn,
            num_simulations_per_move,
        )
        env, reward = jax.vmap(env_step)(env, policy_output.action)
        return (env, rng_key_next, step + 1), MoveOutput(
            state=state,
            action_weights=policy_output.action_weights,
            reward=reward,
            terminated=terminated,
        )

    env = reset_env(env)
    env = replicate(env, batch_size)
    step = jnp.array(1)
    _, self_play_data = pax.scan(
        single_move,
        (env, rng_key, step),
        None,
        length=env.max_num_steps(),
        time_major=False,
    )
    return self_play_data


def prepare_training_data(data: MoveOutput, env: Enviroment):
    """Preprocess the data collected from self-play.

    1. remove states after the enviroment is terminated.
    2. compute the value at each state.
    """
    buffer = []
    num_games = len(data.terminated)
    for i in range(num_games):
        state = data.state[i]
        is_terminated = data.terminated[i]
        action_weights = data.action_weights[i]
        reward = data.reward[i]
        num_steps = len(is_terminated)
        value: Optional[chex.Array] = None
        for idx in reversed(range(num_steps)):
            if is_terminated[idx]:
                continue
            if value is None:
                value = reward[idx]
            else:
                value = -value
            s = np.copy(state[idx])
            a = np.copy(action_weights[idx])
            for augmented_s, augmented_a in env.symmetries(s, a):
                buffer.append(
                    TrainingExample(  # type: ignore
                        state=augmented_s,
                        action_weights=augmented_a,
                        value=np.array(value, dtype=np.float32),
                    )
                )

    return buffer


def collect_self_play_data(
    agent,
    env,
    rng_key: chex.Array,
    batch_size: int,
    data_size: int,
    num_simulations_per_move: int,
):
    """Collect self-play data for training."""
    num_iters = data_size // batch_size
    devices = jax.local_devices()
    num_devices = len(devices)
    rng_key_list = jax.random.split(rng_key, num_iters * num_devices)
    rng_keys = jnp.stack(rng_key_list).reshape((num_iters, num_devices, -1))  # type: ignore
    data = []

    with click.progressbar(range(num_iters), label="  self play     ") as bar:
        for i in bar:
            batch = collect_batched_self_play_data(
                agent,
                env,
                rng_keys[i],
                batch_size // num_devices,
                num_simulations_per_move,
            )
            batch = jax.device_get(batch)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, *x.shape[2:])), batch
            )
            data.extend(prepare_training_data(batch, env=env))
    return data


def collect_ownership_data(log_path):
    def get_position(gtp_row) -> AnalyzedPosition:
        try:
            a0position = AnalyzedPosition.from_gtp_log(gtp_data=gtp_row)
        except:
            return None
        return a0position

    data = []
    for tupla in os.walk(log_path):
        dir, inside_dirs, files = tupla
        for file in files:
            if not file.endswith('.log'):
                continue
            file = os.path.join(dir, file)
            with open(file, 'r') as f:
                gtp_games = f.read().splitlines()
                for gtp_game in gtp_games:
                    a0pos = get_position(json.loads(gtp_game))
                    if a0pos is None:
                        print(f'Bad log found in {file}: {gtp_game}')
                        continue
                    move_num = a0pos.move_num
                    try:
                        data_list = a0pos.get_single_local_pos(move_num=move_num)
                    except AssertionError as e:
                        print(e)
                        continue
                    for datapoint in data_list:
                        mask, position_list, coords, color, value = datapoint
                        has_next_move = True
                        if coords is None or color is None:
                            assert coords is None and color is None
                            coords = (a0pos.pad_size, a0pos.pad_size)
                            color = 0
                            has_next_move = False

                        state = jnp.moveaxis(jnp.array(position_list), 0, -1)

                        # value = jnp.array(a0pos.continous_ownership)  # .flatten()
                        example = TrainingOwnershipExample(state=state,
                                                           has_next_move=jnp.array(has_next_move),
                                                           next_move_color=jnp.array(color),
                                                           next_move_coords=jnp.array(coords),
                                                           color_to_move=jnp.array(-a0pos.last_color),
                                                           # move=jnp.array(move),
                                                           mask=jnp.array(mask),
                                                           board_mask=jnp.array(a0pos.board_mask),
                                                           value=jnp.array(value))
                        data.append(example)
    return data


def loss_fn(net, data: TrainingExample):
    """Sum of value loss and policy loss."""
    net, (action_logits, value) = batched_policy(net, data.state)

    # value loss (mse)
    mse_loss = optax.l2_loss(value, data.value)
    mse_loss = jnp.mean(mse_loss)

    # policy loss (KL(target_policy', agent_policy))
    target_pr = data.action_weights
    # to avoid log(0) = nan
    target_pr = jnp.where(target_pr == 0, EPSILON, target_pr)
    action_logits = jax.nn.log_softmax(action_logits, axis=-1)
    kl_loss = jnp.sum(target_pr * (jnp.log(target_pr) - action_logits), axis=-1)
    kl_loss = jnp.mean(kl_loss)

    # return the total loss
    return mse_loss + kl_loss, (net, (mse_loss, kl_loss))


def apply_random_flips(d: TrainingOwnershipExample):
    # TODO: add color flip?
    state, mask, board_mask, has_next_move, coords, value, color_to_move, next_color = d.state, d.mask, d.board_mask, d.has_next_move, d.next_move_coords, d.value, d.color_to_move, d.next_move_color
    if random.choice([0, 1]):
        state = state[:, ::-1, :]
        mask = mask[:, ::-1]
        board_mask = board_mask[:, ::-1]
        value = value[:, ::-1]
        if has_next_move == 1:
            coords = jnp.array((coords[0], mask.shape[1] - coords[1] - 1))
    if random.choice([0, 1]):
        state = state[::-1, :, :]
        mask = mask[::-1, :]
        board_mask = board_mask[::-1, :]
        value = value[::-1, :]
        if has_next_move == 1:
            coords = jnp.array((mask.shape[0] - coords[0] - 1, coords[1]))
    new_example = TrainingOwnershipExample.from_obj(
        d,
        state=state,
        mask=mask,
        board_mask=board_mask,
        value=value,
        next_move_coords=coords,
    )
    return new_example  # TrainingOwnershipExample(state=state, mask=mask, board_mask=board_mask, next_move_coords=coords, value=value, color_to_move=color_to_move, next_color=next_color)


#def zero_grads():
#    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
#    def init_fn(_):
#        return ()
#    def update_fn(updates, state, params=None):
#        return jax.tree_map(jnp.zeros_like, updates), ()
#    return optax.GradientTransformation(init_fn, update_fn)

#tx = optax.multi_transform({'adam': optax.adam(0.1), 'zero': zero_grads()},
#                           create_mask(params, lambda s: s.startswith('frozen')))

def construct_move_target(has_next_move, coords, color):
    target_pr = jnp.zeros([2 * 19 * 19])
    if has_next_move == 0:
        pass
        # move_loc = 2 * 19 * 19
    else:
        x, y = coords
        move_loc = 19 * x + y

        # If next move is Black (1), then current move is White (-1), and we add 361 to the index in the possible next move coordinates array
        if color == 1:
            move_loc += 19 * 19
        target_pr = target_pr.at[move_loc].set(1)
    return target_pr


def flatten_mask(mask_361):
    single_flat = mask_361.reshape(mask_361.shape[0], -1)
    full_flat = jnp.ones((single_flat.shape[0], 722))
    full_flat = full_flat.at[..., :361].set(single_flat)
    full_flat = full_flat.at[..., 361:722].set(single_flat)
    return full_flat


def flatten_preds(preds_361):
    first_flat = preds_361[..., 0].reshape(preds_361[..., 0].shape[0], -1)
    full_flat = jnp.ones((first_flat.shape[0], 722))
    full_flat = full_flat.at[..., :361].set(first_flat)
    second_flat = preds_361[..., 1].reshape(preds_361[..., 1].shape[0], -1)
    full_flat = full_flat.at[..., 361:722].set(second_flat)
    return full_flat


def construct_flat_mask(data: TrainingOwnershipDatapoint):
    allowed_moves_mask = (data.mask == 1) & (data.state[..., 7] == 0)
    return flatten_mask(allowed_moves_mask)


def construct_training_datapoint(d: TrainingOwnershipExample):
    new_datapoint = TrainingOwnershipDatapoint.from_obj(d)
    return new_datapoint


def ownership_loss_fn(net, data: TrainingOwnershipDatapoint):
    """Sum of value loss and policy loss."""
    net, (action_logits, ownership_map) = batched_policy(net, (data.state, data.mask, data.board_mask))

    target_pr = data.move
    # to avoid log(0) = nan
    # target_pr = jnp.where(target_pr == 0, EPSILON, target_pr)
    flattened_ownership_mask = data.mask.reshape(data.mask.shape[0], -1)
    flattened_ownership_map = ownership_map.reshape(ownership_map.shape[0], -1)
    mse_loss = optax.l2_loss(flattened_ownership_map * flattened_ownership_mask, data.value.reshape(data.value.shape[0], -1) * flattened_ownership_mask)
    mse_loss = jnp.mean(mse_loss)

    # policy loss (KL(target_policy', agent_policy))
    action_logits = flatten_preds(action_logits)

    log_action_logits = jax.nn.log_softmax(action_logits, axis=-1)
    soft_action_logits = jax.nn.softmax(action_logits, axis=-1)
    # TODO: Is KL loss a good choice when my target is categorical (one-hot encoded) and not an array of probabilities?
    #  ANSWER: No

    flat_mask = construct_flat_mask(data)
    penalty_weight = 2.

    # target_pr_valid = target_pr[flat_mask]
    # action_logits_valid = action_logits[flat_mask]

    cross_entropy_loss = - jnp.sum(target_pr * log_action_logits, axis=-1)
    cross_entropy_loss = jnp.mean(cross_entropy_loss)

    # Penalty term for moves outside the mask
    action_logits_invalid = soft_action_logits * (1 - flat_mask)
    invalid_moves_penalty = penalty_weight * jnp.sum(action_logits_invalid, axis=-1)
    invalid_moves_penalty = jnp.mean(invalid_moves_penalty)

    total_loss = cross_entropy_loss + invalid_moves_penalty

    # return the total loss
    return mse_loss + total_loss, (net, (mse_loss, total_loss))


@partial(jax.pmap, axis_name="i")
def train_step(net, optim, data: TrainingExample):
    """A training step."""
    (_, (net, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


@partial(jax.pmap, axis_name="i")
def train_ownership_step(net, optim, data: TrainingOwnershipDatapoint):
    """A training step."""
    #data.state = net.backbone(data.state, data.mask, data.board_mask, batched=True)
    (_, (net, losses)), grads = jax.value_and_grad(ownership_loss_fn, has_aux=True)(net, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    #for name in grads.pytree_attributes:
        #value = getattr(grads, name)
        #value = jax.tree_util.tree_map(
        #    _module_or_array,
        #    value,
        #    state_dict[name],
        #    is_leaf=lambda x: isinstance(x, Module),
        #)
    #    setattr(grads, name, 0)
    #grads = jax.tree_util.tree_map(lambda u, mt: u * , grads, multitransformer)
    #net = pax.unfreeze_parameters(net)
    net, optim = opax.apply_gradients(net, optim, grads)
    #net.freeze_parameters()
    return net, optim, losses


#@partial(jax.pmap, axis_name="i")
def test_ownership(net, data: TrainingOwnershipDatapoint):
    # This function has outdated code
    """Evaluation on test set."""
    # SF: This if is redundant, it's not a list anyway, it's a dynamic jaxpr tracer - NO, IT'S NOT
    if isinstance(data, list):
        # print('VERY UNEXPECTED IF USAGE')
        state = jnp.stack(list(d.state for d in data))
        mask = jnp.stack(list(d.mask for d in data))
        board_mask = jnp.stack(list(d.board_mask for d in data))
        move = jnp.stack(list(d.move for d in data))
        value = jnp.stack(list(d.value for d in data))
    else:
        state = data.state
        mask = data.mask
        board_mask = data.board_mask
        move = data.move
        value = data.value
    action_logits, ownership_map = net((state, mask, board_mask), batched=True)
    action_logits = flatten_preds(action_logits)
    top_1_acc = (action_logits.argmax(axis=1) == move.argmax(axis=1)).mean()
    flattened_ownership_mask = mask.reshape(mask.shape[0], -1)
    flattened_ownership_map = ownership_map.reshape(ownership_map.shape[0], -1)
    mse_loss = optax.l2_loss(flattened_ownership_map * flattened_ownership_mask, value.reshape(value.shape[0], -1) * flattened_ownership_mask)
    return top_1_acc, mse_loss


@pax.pure
def train(
    game_class="games.go_game.GoBoard9x9",
    agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
    selfplay_batch_size: int = 128,
    training_batch_size: int = 128,  # Originally 128 but maybe I'm getting OOM
    num_iterations: int = 20000,
    num_simulations_per_move: int = 32,
    num_self_plays_per_iteration: int = 128 * 100,
    learning_rate: float = 1e-3,  # Originally 0.01,
    ckpt_filename: str = "go_agent_9x9_128_sym.ckpt",
    trained_ckpt_filename: str = "trained.ckpt",
    root_dir: str = ".",
    random_seed: int = 42,
    weight_decay: float = 1e-4,
    lr_decay_steps: int = 100_000,
):
    if root_dir == ".":
        root_dir = os.path.dirname(os.getcwd())

    """Train an agent by self-play."""
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    # STAS-08: moved the following lines loading weights before transfer_model definition
    ckpt_path = os.path.join(root_dir, ckpt_filename)
    trained_ckpt_path = os.path.join(root_dir, trained_ckpt_filename)
    if os.path.isfile(ckpt_path):
        if os.path.isfile(trained_ckpt_path):
            agent = ResnetPolicyValueNet128(input_dims=(9, 9, 9), num_actions=82)
            start_iter = 0  # TODO: it should be different
        else:
            print("Loading weights at", ckpt_filename)
            with open(ckpt_path, "rb") as f:
                dic = pickle.load(f)
                if "agent" in dic:
                    dic = dic["agent"]
                agent = agent.load_state_dict(dic)
                #optim = optim.load_state_dict(dic["optim"])
                start_iter = 0 #dic["iter"] + 1
    else:
        start_iter = 0

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    transfer_model = TransferResnet(agent)
    #multitransformer = TransferResnet()

    if os.path.isfile(trained_ckpt_path):
        print('Loading trained weights at', trained_ckpt_filename)
        with open(trained_ckpt_path, "rb") as f:
            loaded_agent = pickle.load(f)
            if "agent" in loaded_agent:
                loaded_agent = loaded_agent["agent"]
            transfer_model = transfer_model.load_state_dict(loaded_agent)

    optim = opax.chain(
        opax.add_decayed_weights(weight_decay),
        opax.sgd(lr_schedule, momentum=0.9),
    ).init(transfer_model.parameters())



    rng_key = jax.random.PRNGKey(random_seed)
    shuffler = random.Random(random_seed)
    devices = jax.local_devices()
    num_devices = jax.local_device_count()

    def _stack_and_reshape(*xs):
        x = np.stack(xs)
        x = np.reshape(x, (num_devices, -1) + x.shape[1:])
        return x

    print(f"  time {datetime.datetime.now().strftime('%H:%M:%S')}")

    pickle_test_path = os.path.join(root_dir, TEST_DIR, 'test_data.pkl')
    if os.path.isfile(pickle_test_path):
        with open(pickle_test_path, "rb") as f:
            test_data = cloudpickle.load(f)
    else:
        test_data = collect_ownership_data(os.path.join(root_dir, TEST_DIR))
        with open(pickle_test_path, "wb") as f:
            cloudpickle.dump(test_data, f)

    # transfer_model = transfer_model.eval()
    #
    #
    # accs = []
    # mses = []
    # ids = range(0, len(test_data) - training_batch_size, training_batch_size)
    # with click.progressbar(ids, label="  test model   ") as progressbar:
    #     for idx in progressbar:
    #         batch = test_data[idx: (idx + training_batch_size)]
    #         batch = [TrainingOwnershipExample(state=d.state, move=d.move, mask=d.mask, board_mask=d.board_mask,
    #                                           value=d.value) for d in batch]
    #         # batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
    #         top_1_acc, mse = test_ownership(transfer_model, batch)
    #         accs.append(top_1_acc)
    #         mses.append(mse)
    #
    # top_1_acc = np.mean(accs)
    # mse = np.mean(mses)
    #
    # print(
    #     f"  test top 1 accuracy {top_1_acc:.3f}"
    #     f"  test ownership map MSE {mse:.3f}"
    #     f"  time {datetime.datetime.now().strftime('%H:%M:%S')}"
    # )

    already_pickled = [int(filename[9:-4]) for filename in os.listdir(os.path.join(root_dir, TRAIN_DIR)) if filename.endswith('.pkl')]
    try:
        small_counter = max(already_pickled)
    except ValueError:
        small_counter = 0
    start_iter += 1
    num_iterations += 1
    unpacked = [int(filename) for filename in os.listdir(os.path.join(root_dir, TRAIN_DIR)) if os.path.isdir(os.path.join(root_dir, TRAIN_DIR, filename))]
    try:
        last_unpacked = max(unpacked)
    except ValueError:
        last_unpacked = 0
    print(f"Unpacked: {unpacked}")

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")
        pickle_path = os.path.join(root_dir, TRAIN_DIR, f'datasmall{iteration:04}.pkl')
        if not os.path.isfile(pickle_path):
            last_unpacked += 1
            zip_path = os.path.join(root_dir, TRAIN_DIR, f'{last_unpacked:04}.zip')
            if not os.path.exists(zip_path[:-4]):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(path=zip_path[:-4])
            unzipped = zip_path[:-4]
            for folder in os.listdir(unzipped):
                small_counter += 1
                cur_pickle_path = os.path.join(root_dir, TRAIN_DIR, f'datasmall{small_counter:04}.pkl')
                partial_data = collect_ownership_data(os.path.join(unzipped, folder))
                with open(cur_pickle_path, "wb") as f:
                    cloudpickle.dump(partial_data, f)
                shutil.rmtree(os.path.join(unzipped, folder), ignore_errors=True)
                #data.extend(partial_data)
        with open(pickle_path, "rb") as f:
            data = cloudpickle.load(f)

        print(f"  time {datetime.datetime.now().strftime('%H:%M:%S')}")
        shuffler.shuffle(data)
        # old_model = jax.tree_util.tree_map(jnp.copy, transfer_model)
        transfer_model, losses = transfer_model.train(), []
        # transfer_model.backbone = transfer_model.backbone.eval()
        # transfer_model.backbone = pax.freeze_parameters(transfer_model.backbone)
        transfer_model, optim = jax.device_put_replicated((transfer_model, optim), devices)
        ids = range(0, len(data) - training_batch_size, training_batch_size)
        with click.progressbar(ids, label="  train model   ") as progressbar:
            for idx in progressbar:
                batch = data[idx: (idx + training_batch_size)]
                batch = [apply_random_flips(d) for d in batch]
                batch = [construct_training_datapoint(d) for d in batch]
                # batch = [TrainingOwnershipExample(state=d.state, move=d.move, mask=d.mask, board_mask=d.board_mask,
                #                                   value=d.value) for d in batch]
                batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                transfer_model, optim, loss = train_ownership_step(transfer_model, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = np.mean(sum(jax.device_get(loss))) / len(loss)
        policy_loss = np.mean(sum(jax.device_get(policy_loss))) / len(policy_loss)
        transfer_model, optim = jax.tree_util.tree_map(lambda x: x[0], (transfer_model, optim))

        if iteration % 10 == 0:
            transfer_model = transfer_model.eval()
            # save agent's weights to disk
            with open(trained_ckpt_path, "wb") as writer:
                dic = {
                    "agent": jax.device_get(transfer_model.state_dict()),
                    "optim": jax.device_get(transfer_model.state_dict()),
                    "iter": iteration,
                }
                pickle.dump(dic, writer)

            accs = []
            mses = []
            ids = range(0, len(test_data) - training_batch_size, training_batch_size)
            with click.progressbar(ids, label="  test model   ") as progressbar:
                for idx in progressbar:
                    batch = test_data[idx: (idx + training_batch_size)]
                    # I needed to move axis
                    batch = [#TrainingOwnershipExample(state=d.state, move=d.move, mask=d.mask, board_mask=d.board_mask,
                             #                         value=d.value)
                             construct_training_datapoint(d) for d in batch]
                    # batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                    top_1_acc, mse = test_ownership(transfer_model, batch)
                    accs.append(top_1_acc)
                    mses.append(mse)

            top_1_acc = np.mean(accs)
            mse = np.mean(mses)

            print(
                f"  ownership loss {value_loss:.3f}"
                f"  policy loss {policy_loss:.3f}"
                f"  test top 1 accuracy {top_1_acc:.3f}"
                f"  test ownership map MSE {mse:.3f}"
                f"  learning rate {optim[1][-1].learning_rate:.1e}"
                f"  time {datetime.datetime.now().strftime('%H:%M:%S')}"
            )
    print("Done!")


"""
    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        agent = agent.eval()
        data = collect_self_play_data(
            agent,
            env,
            rng_key_1,  # type: ignore
            selfplay_batch_size,
            num_self_plays_per_iteration,
            num_simulations_per_move,
        )
        data = list(data)
        shuffler.shuffle(data)
        old_agent = jax.tree_util.tree_map(jnp.copy, agent)
        agent, losses = agent.train(), []
        agent, optim = jax.device_put_replicated((agent, optim), devices)
        ids = range(0, len(data) - training_batch_size, training_batch_size)
        with click.progressbar(ids, label="  train agent   ") as progressbar:
            for idx in progressbar:
                batch = data[idx : (idx + training_batch_size)]
                batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                agent, optim, loss = train_step(agent, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = np.mean(sum(jax.device_get(value_loss))) / len(value_loss)
        policy_loss = np.mean(sum(jax.device_get(policy_loss))) / len(policy_loss)
        agent, optim = jax.tree_util.tree_map(lambda x: x[0], (agent, optim))
        # new agent is player 1
        result_1: PlayResults = agent_vs_agent_multiple_games(
            agent.eval(),
            old_agent,
            env,
            rng_key_2,
            num_simulations_per_move=32,
        )
        # old agent is player 1
        result_2: PlayResults = agent_vs_agent_multiple_games(
            old_agent,
            agent.eval(),
            env,
            rng_key_3,
            num_simulations_per_move=32,
        )
        print(
            "  evaluation      {} win - {} draw - {} loss".format(
                result_1.win_count + result_2.loss_count,
                result_1.draw_count + result_2.draw_count,
                result_1.loss_count + result_2.win_count,
            )
        )
        print(
            f"  value loss {value_loss:.3f}"
            f"  policy loss {policy_loss:.3f}"
            f"  learning rate {optim[1][-1].learning_rate:.1e}"
        )
        # save agent's weights to disk
        with open(ckpt_filename, "wb") as writer:
            dic = {
                "agent": jax.device_get(agent.state_dict()),
                "optim": jax.device_get(optim.state_dict()),
                "iter": iteration,
            }
            pickle.dump(dic, writer)
    print("Done!")
"""


if __name__ == "__main__":
    if "COLAB_TPU_ADDR" in os.environ:
        jax.tools.colab_tpu.setup_tpu()
    print("Cores:", jax.local_devices())

    fire.Fire(train)
