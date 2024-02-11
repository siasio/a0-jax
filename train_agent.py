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
from typing import Optional, Callable

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
from opax.transform import GradientTransformation

from games.env import Enviroment
from play import PlayResults, agent_vs_agent_multiple_games
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import batched_policy, env_step, import_class, replicate, reset_env, stack_last_state
from policies.resnet_policy import TransferResnet, ResnetPolicyValueNet128, BareHead
from local_pos_masks import AnalyzedPosition
from typing import Tuple
import datetime

EPSILON = 1e-9  # a very small positive value
TRAIN_DIR = "zip_logs_new"
TEST_DIR = "test_dir"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.79'


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


def collect_ownership_data(log_path, use_only_19x19=True):
    def get_position(gtp_row) -> AnalyzedPosition:
        try:
            a0position = AnalyzedPosition.from_gtp_log(gtp_data=gtp_row)
        except:
            return None
        return a0position
    exception_counter = 0
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
                        # print(f'Bad log found in {file}: {gtp_game}')
                        exception_counter += 1
                        continue
                    if use_only_19x19 and (a0pos.size_x != 19 or a0pos.size_y != 19):
                        continue
                    move_num = a0pos.move_num
                    try:
                        data_list = a0pos.get_single_local_pos(move_num=move_num)
                    except Exception as e:
                        # print(e)
                        exception_counter += 1
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

                        # value = jnp.array(a0pos.continuous_ownership)  # .flatten()
                        example = TrainingOwnershipExample(state=state,
                                                           has_next_move=jnp.array(has_next_move),
                                                           next_move_color=jnp.array(color),
                                                           next_move_coords=jnp.array(coords),
                                                           color_to_move=jnp.array(a0pos.color_to_play),  # -a0pos.last_color),
                                                           # move=jnp.array(move),
                                                           mask=jnp.array(mask),
                                                           board_mask=jnp.array(a0pos.board_mask),
                                                           value=jnp.array(value))
                        data.append(example)
    print(f'{log_path} Exception counter: {exception_counter}')
    return data


def apply_random_flips(d: TrainingOwnershipExample):
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


def construct_move_target(has_next_move, coords, color):
    target_pr = jnp.zeros([2 * 19 * 19 + 1])
    if has_next_move == 0:
        move_loc = 2 * 19 * 19
    else:
        x, y = coords
        move_loc = 19 * x + y

        # If next move is Black (1), then current move is White (-1), and we add 361 to the index in the possible next move coordinates array
        if color == 1:
            move_loc += 19 * 19
    target_pr = target_pr.at[move_loc].set(1)
    return target_pr


def flatten_mask(mask_361, allow_last=True):
    single_flat = mask_361.reshape(mask_361.shape[0], -1)
    full_flat = jnp.ones((single_flat.shape[0], 723)) if allow_last else jnp.zeros((single_flat.shape[0], 723))
    full_flat = full_flat.at[..., :361].set(single_flat)
    full_flat = full_flat.at[..., 361:722].set(single_flat)
    return full_flat


def flatten_preds(preds_361):
    if preds_361.shape[-1] == 2:
        first_flat = preds_361[..., 0].reshape(preds_361[..., 0].shape[0], -1)
        full_flat = jnp.ones((first_flat.shape[0], 722))
        full_flat = full_flat.at[..., :361].set(first_flat)
        second_flat = preds_361[..., 1].reshape(preds_361[..., 1].shape[0], -1)
        full_flat = full_flat.at[..., 361:722].set(second_flat)
        return full_flat
    return preds_361


def construct_flat_mask(data: TrainingOwnershipDatapoint):
    if data.move[-1] == 1:
        return flatten_mask(jnp.zeros_like(data.mask), allow_last=True)
    else:
        allowed_moves_mask = (data.mask == 1) & (data.state[..., 7] == 0)
        return flatten_mask(allowed_moves_mask, allow_last=False)


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
    mse_loss = optax.l2_loss(flattened_ownership_map * flattened_ownership_mask,
                             data.value.reshape(data.value.shape[0], -1) * flattened_ownership_mask)
    mse_loss = jnp.mean(mse_loss)

    # policy loss (KL(target_policy', agent_policy))
    action_logits = flatten_preds(action_logits)

    log_action_logits = jax.nn.log_softmax(action_logits, axis=-1)
    soft_action_logits = jax.nn.softmax(action_logits, axis=-1)

    flat_mask = construct_flat_mask(data)
    penalty_weight = 2.

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
def train_ownership_step(net, optim, data: TrainingOwnershipDatapoint):
    """A training step."""
    (_, (net, losses)), grads = jax.value_and_grad(ownership_loss_fn, has_aux=True)(net, data)
    grads = jax.lax.pmean(grads, axis_name="i")
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


@partial(jax.pmap, axis_name="i")
def test_ownership(net, data: TrainingOwnershipDatapoint):
    """Evaluation on test set."""
    (top_1_acc, (top_2_acc, mse_loss)), _ = jax.value_and_grad(calculate_metrics, has_aux=True)(net, data)
    return top_1_acc, top_2_acc, mse_loss


def calculate_metrics(net, data: TrainingOwnershipDatapoint):
    net, (action_logits, ownership_map) = batched_policy(net, (data.state, data.mask, data.board_mask))

    action_logits = flatten_preds(action_logits)
    top_1_acc = (action_logits.argmax(axis=1) == data.move.argmax(axis=1)).mean()
    top_2_acc = (jnp.argsort(action_logits, axis=1)[:, -2:] == data.move.argmax(axis=1)[..., None]).any(axis=1).mean()
    flattened_ownership_mask = data.mask.reshape(data.mask.shape[0], -1)
    flattened_ownership_map = ownership_map.reshape(ownership_map.shape[0], -1)
    mse_loss = optax.l2_loss(flattened_ownership_map * flattened_ownership_mask,
                             data.value.reshape(data.value.shape[0], -1) * flattened_ownership_mask)
    return top_1_acc, (top_2_acc, mse_loss)


def multi_transform(schedule_fn: Callable[[jnp.ndarray], jnp.ndarray]):
    """
    Freeze backbone weights in the beginning of training
    """

    count: jnp.ndarray
    backbone_multiplier: jnp.ndarray

    class MultiTransform(GradientTransformation):
        def __init__(self, params):
            super().__init__(params=params)
            self.schedule_fn = schedule_fn
            self.count = jnp.array(0, dtype=jnp.int32)
            self.backbone_multiplier = self.schedule_fn(self.count)

        def __call__(self, updates, params=None):
            del params
            self.count = self.count + 1
            self.backbone_multiplier = self.schedule_fn(self.count)

            updates = jax.tree_util.tree_map_with_path(
                lambda path, u: self.backbone_multiplier * u if "backbone" in jax.tree_util.keystr(path) else u, updates
            )
            return updates

    return MultiTransform


def save_model(trained_ckpt_path, model, iteration):
    with open(trained_ckpt_path, "wb") as writer:
        dic = {
            "agent": jax.device_get(model.state_dict()),
            "optim": jax.device_get(model.state_dict()),
            "iter": iteration,
        }
        pickle.dump(dic, writer)


def test_model(test_data, batch_size, model, optim, value_loss, policy_loss, _stack_and_reshape, devices):
    transfer_model = model.eval()
    transfer_model, optim = jax.device_put_replicated((transfer_model, optim), devices)
    accs1 = []
    accs2 = []
    mses = []
    ids = range(0, len(test_data) - batch_size, batch_size)
    with click.progressbar(ids, label="  test model   ") as progressbar:
        for idx in progressbar:
            batch = test_data[idx: (idx + batch_size)]
            # I needed to move axis
            batch = [construct_training_datapoint(d) for d in batch]
            batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
            top_1_acc, top_2_acc, mse = test_ownership(transfer_model, batch)
            accs1.append(top_1_acc)
            accs2.append(top_2_acc)
            mses.append(mse)

    top_1_acc = np.mean(accs1)
    top_2_acc = np.mean(accs2)
    mse = np.mean(mses)

    if value_loss is not None or policy_loss is not None:
        text_to_print = f"  ownership loss {value_loss:.3f}  policy loss {policy_loss:.3f}"
    else:
        text_to_print = ""
    lr = optim[1][-1].learning_rate[0]
    backbone_multiplier = optim[2].backbone_multiplier[0]
    print(text_to_print +
          f"  test top 1 acc {top_1_acc:.3f}"
          f"  test top 2 acc {top_2_acc:.3f}"
          f"  test ownership MSE {mse:.4f}"
          f"  learning rate {lr:.1e}"
          f"  backbone multiplier {backbone_multiplier:.1f}"
          f"  time {datetime.datetime.now().strftime('%H:%M:%S')}"
          )
    return value_loss, policy_loss, top_1_acc, mse, lr, backbone_multiplier


def plot_stats(filename, root_dir):
    with open(filename, "rb") as f:
        v_losses, p_losses, t1_accs, mses, lrs, bms, indices = pickle.load(f)
    os.makedirs(os.path.dirname(root_dir), exist_ok=True)
    run_name = os.path.basename(filename).rsplit('.', 1)[0]
    import matplotlib.pyplot as plt
    plt.plot(v_losses)
    plt.plot(p_losses)
    plt.plot(t1_accs)
    plt.plot([None] + [mse * 1000 for mse in mses])
    plt.plot([lr * 100 for lr in lrs])
    plt.plot(bms)
    plt.xticks(indices)
    plt.legend(['Value loss', 'Policy loss', 'Top 1 accuracy', 'MSE * 1000', 'Learning rate * 100', 'Backbone multiplier'])
    plt.savefig(os.path.join(root_dir, f'metrics-{run_name}.png'))
    # print the list of floats with only 3 decimals
    print("Value losses:", ", ".join([f"{v_loss:.3f}" for v_loss in v_losses if v_loss is not None]))
    print("Policy losses:", ", ".join([f"{p_loss:.3f}" for p_loss in p_losses if p_loss is not None]))
    print("Top 1 accuracies:", ", ".join([f"{t1_acc:.3f}" for t1_acc in t1_accs if t1_acc is not None]))
    print("MSEs:", ", ".join([f"{mse:.3f}" for mse in mses if mse is not None]))
    print("Learning rates:", ", ".join([f"{lr:.3f}" for lr in lrs if lr is not None]))
    print("Backbone multipliers:", ", ".join([f"{bm:.3f}" for bm in bms if bm is not None]))


@pax.pure
def train(
        trained_ckpt_filename: str,
        game_class="games.go_game.GoBoard9x9",
        agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
        training_batch_size: int = 128,  # Originally 128 but maybe I'm getting OOM
        num_iterations: int = 361,
        learning_rate: float = 1e-2,  # Originally 0.01,
        ckpt_filename: str = "go_agent_9x9_128_sym.ckpt",
        root_dir: str = ".",
        random_seed: int = 42,
        weight_decay: float = 1e-4,
        lr_decay_steps: int = 10_000,  # My full epoch is likely shorter than 100_000 steps
        backbone_lr_steps: int = 15_000,  # was 25_000 in August
        use_only_19x19: bool = True,
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
    trained_ckpt_path = os.path.join(root_dir, 'models', trained_ckpt_filename)
    os.makedirs(os.path.dirname(trained_ckpt_path), exist_ok=True)
    # check_backbone(ckpt_path, trained_ckpt_path, agent)
    # return
    start_iter = 1
    if os.path.isfile(ckpt_path) and not os.path.isfile(trained_ckpt_path):
        print("Loading weights at", ckpt_filename)
        with open(ckpt_path, "rb") as f:
            dic = pickle.load(f)
            if "agent" in dic:
                dic = dic["agent"]
            agent = agent.load_state_dict(dic)

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    def lr_backbone_schedule(step):
        return step > backbone_lr_steps

    transfer_model = TransferResnet(agent, include_boardmask=not use_only_19x19)
    # transfer_model = BareHead(include_boardmask=not use_only_19x19)

    optim = opax.chain(
        opax.add_decayed_weights(weight_decay),
        opax.sgd(lr_schedule, momentum=0.9),
        multi_transform(lr_backbone_schedule)
    ).init(transfer_model.parameters())

    if os.path.isfile(trained_ckpt_path):
        print('Loading trained weights at', trained_ckpt_filename)
        with open(trained_ckpt_path, "rb") as f:
            loaded_agent = pickle.load(f)
            start_iter = loaded_agent["iter"] + 1
            optim = optim.load_state_dict(loaded_agent["optim"])
            transfer_model = transfer_model.load_state_dict(loaded_agent["agent"])


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
        test_data = collect_ownership_data(os.path.join(root_dir, TEST_DIR), use_only_19x19=use_only_19x19)
        with open(pickle_test_path, "wb") as f:
            cloudpickle.dump(test_data, f)

    already_pickled = [int(filename[9:-4]) for filename in os.listdir(os.path.join(root_dir, TRAIN_DIR)) if
                       filename.endswith('.pkl')]
    try:
        small_counter = max(already_pickled)
    except ValueError:
        small_counter = 0
    unpacked = [int(filename) for filename in os.listdir(os.path.join(root_dir, TRAIN_DIR)) if
                os.path.isdir(os.path.join(root_dir, TRAIN_DIR, filename))]
    try:
        last_unpacked = max(unpacked)
    except ValueError:
        last_unpacked = 0
    print(f"Unpacked: {unpacked}")

    value_loss, policy_loss = None, None

    v_losses, p_losses, t1_accs, mses, lrs, bms, indices = [], [], [], [], [], [], []
    stats_pickle_name = trained_ckpt_path.rsplit('.', 1)[0] + '_stats.pkl'

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")

        if iteration % 20 == 1 or iteration in (6, 11, 16):
            value_loss, policy_loss, top_1_acc, mse, lr, backbone_multiplier = test_model(test_data, training_batch_size, transfer_model, optim, value_loss, policy_loss, _stack_and_reshape, devices)
            v_losses.append(value_loss)
            p_losses.append(policy_loss)
            t1_accs.append(top_1_acc)
            mses.append(mse)
            lrs.append(lr)
            bms.append(backbone_multiplier)
            indices.append(iteration)
        pickle_path = os.path.join(root_dir, TRAIN_DIR, f'datasmall{iteration:04}.pkl')
        if not os.path.isfile(pickle_path):
            last_unpacked += 1
            zip_path = os.path.join(root_dir, TRAIN_DIR, f'{last_unpacked:04}.zip')
            if not os.path.exists(zip_path[:-4]):
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(path=zip_path[:-4])
                except:
                    value_loss, policy_loss, top_1_acc, mse, lr, backbone_multiplier = test_model(test_data,
                                                                                                  training_batch_size,
                                                                                                  transfer_model, optim,
                                                                                                  value_loss,
                                                                                                  policy_loss,
                                                                                                  _stack_and_reshape,
                                                                                                  devices)
                    v_losses.append(value_loss)
                    p_losses.append(policy_loss)
                    t1_accs.append(top_1_acc)
                    mses.append(mse)
                    lrs.append(lr)
                    bms.append(backbone_multiplier)
                    indices.append(iteration)
                    break
            unzipped = zip_path[:-4]
            for folder in os.listdir(unzipped):
                small_counter += 1
                cur_pickle_path = os.path.join(root_dir, TRAIN_DIR, f'datasmall{small_counter:04}.pkl')
                partial_data = collect_ownership_data(os.path.join(unzipped, folder), use_only_19x19=use_only_19x19)
                with open(cur_pickle_path, "wb") as f:
                    cloudpickle.dump(partial_data, f)
                shutil.rmtree(os.path.join(unzipped, folder), ignore_errors=True)
                # data.extend(partial_data)
        with open(pickle_path, "rb") as f:
            data = cloudpickle.load(f)

        print(f"  time {datetime.datetime.now().strftime('%H:%M:%S')}")
        shuffler.shuffle(data)
        transfer_model, losses = transfer_model.train(), []
        transfer_model, optim = jax.device_put_replicated((transfer_model, optim), devices)
        ids = range(0, len(data) - training_batch_size, training_batch_size)
        with click.progressbar(ids, label="  train model   ") as progressbar:
            for idx in progressbar:
                batch = data[idx: (idx + training_batch_size)]
                batch = [apply_random_flips(d) for d in batch]
                batch = [construct_training_datapoint(d) for d in batch]
                batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                transfer_model, optim, loss = train_ownership_step(transfer_model, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = np.mean(sum(jax.device_get(loss))) / len(loss)
        policy_loss = np.mean(sum(jax.device_get(policy_loss))) / len(policy_loss)
        transfer_model, optim = jax.tree_util.tree_map(lambda x: x[0], (transfer_model, optim))

        if iteration % 20 == 1:
            # save agent's weights to disk
            save_model(trained_ckpt_path, transfer_model, iteration)
        with open(stats_pickle_name, "wb") as f:
            pickle.dump((v_losses, p_losses, t1_accs, mses, lrs, bms, indices), f)
    save_model(trained_ckpt_path, transfer_model, num_iterations - 1)
    # Plot metrics
    plot_stats(stats_pickle_name, os.path.join(root_dir, 'stats'))

    print("Done!")


def check_backbone(ckpt_path, trained_ckpt_path, agent):
    """
    Simple utility to check if the backbone weights were indeed frozen
    """
    print("Loading weights at", ckpt_path)
    with open(ckpt_path, "rb") as f:
        dic = pickle.load(f)
        if "agent" in dic:
            dic = dic["agent"]
        agent = agent.load_state_dict(dic)
    jax.tree_util.tree_map(lambda p: print(np.sum(p)), agent)
    transfer_model = TransferResnet(agent)

    if os.path.isfile(trained_ckpt_path):
        print('Loading trained weights at', trained_ckpt_path)
        with open(trained_ckpt_path, "rb") as f:
            loaded_agent = pickle.load(f)
            if "agent" in loaded_agent:
                loaded_agent = loaded_agent["agent"]
            transfer_model = transfer_model.load_state_dict(loaded_agent)

    jax.tree_util.tree_map(lambda p: print(np.sum(p)), transfer_model.module_dict["backbone"])


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
