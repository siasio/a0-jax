"""
AlphaZero training script.

Train agent by self-play only.
"""
import dataclasses
import gc
import itertools
import json
import os
import pickle
import random
import shutil
import time
import traceback
import zipfile
from functools import partial
from typing import Callable

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
import yaml
from opax.transform import GradientTransformation
from tqdm import tqdm

# from torch.fx.experimental.symbolic_shapes import eval_is_non_overlapping_and_dense

import data_vis
from benchmark import get_positions, run_benchmark
from evaluators.a0jax_evaluator import color_to_pl
from jax_utils import batched_policy, import_class
from policies.resnet_policy import TransferResnet
from legacy.local_pos_masks import AnalyzedPosition
import datetime

from sgf_utils.game import IllegalMoveException

EPSILON = 1e-9  # a very small positive value
TRAIN_DIR = "zip_logs_new"
TEST_DIR = "test_dir"
VIS_DIR = "vis_pdf"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.59'


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
    num_moves_played: chex.Numeric
    move: chex.Array
    mask: chex.Array


def training_data_visualize(log_path, use_only_19x19=True, vis_path=None, num_pos=100):
    """
    Deprecated, to be updated or removed
    """
    def get_position(gtp_row) -> AnalyzedPosition:
        try:
            a0position = AnalyzedPosition.from_gtp_log(gtp_data=gtp_row)
        except:
            return None
        return a0position

    exception_counter = 0
    num_games = num_positions = 0
    for tupla in os.walk(log_path):
        dir, inside_dirs, files = tupla
        for file in files:
            if not file.endswith('.log'):
                continue
            file = os.path.join(dir, file)
            with open(file, 'r') as f:
                gtp_games = f.read().splitlines()
                for game_num, gtp_game in enumerate(gtp_games):
                    a0pos = get_position(json.loads(gtp_game))
                    if a0pos is None:
                        # print(f'Bad log found in {file}: {gtp_game}')
                        exception_counter += 1
                        continue
                    if use_only_19x19 and (a0pos.size_x != 19 or a0pos.size_y != 19):
                        continue
                    move_num = a0pos.move_num
                    try:
                        data_list = a0pos.get_vis(move_num=move_num, use_secure_territories=False, )
                    except IllegalMoveException as e:
                        # print(a0pos.game.root.sgf())
                        # print(e)
                        exception_counter += 1
                        continue
                    except Exception as e:
                        # traceback.print_exc()
                        exception_counter += 1
                        continue
                    if len(data_list) == 0:
                        # print("No data list found")
                        continue
                    num_games += 1

                    num_positions += len(data_list)

                    for vis_num, game in enumerate(data_list):
                        pdf_name = f'{vis_path.rsplit(".", 1)[0]}_{num_games:03}_{vis_num:02}.pdf'

                        data_vis.visualize_one(game, pdf_name)
                    if num_games >= num_pos:
                        break


def construct_move_target(has_next_move, coords, color):
    target_pr = np.zeros([2 * 19 * 19 + 1])
    if has_next_move == 0:
        move_loc = 2 * 19 * 19
    else:
        x, y = coords
        # print(x, y, end=', ')
        move_loc = 19 * x + y

        # If next move is Black (1), then current move is White (-1), and we add 361 to the index in the possible next move coordinates array
        if color == 1:
            move_loc += 19 * 19
    # print(move_loc) #, end=', ')
    target_pr[move_loc] = 1
    # target_pr = target_pr.at[move_loc].set(1)
    return target_pr


# names are mixed up, should be the other way around. but usage in code is correct
no_pass_allowed_mask = jnp.ones((723,))
no_pass_allowed_mask = no_pass_allowed_mask.at[..., 722].set(0)
no_move_allowed_mask = jnp.zeros((723,))
# no_move_allowed_mask = no_move_allowed_mask.at[..., 722].set(1)


def flatten_mask(mask_361, allow_last=True):
    single_flat = mask_361.reshape(mask_361.shape[0], -1)
    full_flat = jnp.where(allow_last, jnp.ones((single_flat.shape[0], 723)), jnp.zeros((single_flat.shape[0], 723)))
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
    # allowed_moves_mask = (data.state[..., 7] == 0)  # (data.mask == 1) &
    # mask_for_non_terminal = flatten_mask(allowed_moves_mask, allow_last=False)
    # mask_for_terminal = flatten_mask(jnp.zeros_like(data.mask), allow_last=True)
    masked_batch = jnp.where((data.move[..., -1] == 0)[..., None], no_move_allowed_mask, no_pass_allowed_mask)
    return masked_batch


def ownership_loss_fn(net, data: TrainingOwnershipDatapoint):
    """Sum of value loss and policy loss."""
    net, (action_logits,) = batched_policy(net, (data.state, data.mask))  # was: net, (action_logits, ownership_map)

    target_pr = data.move
    # to avoid log(0) = nan
    # target_pr = jnp.where(target_pr == 0, EPSILON, target_pr)
    # flattened_ownership_mask = data.mask.reshape(data.mask.shape[0], -1)
    # flattened_ownership_map = ownership_map.reshape(ownership_map.shape[0], -1)
    # mse_loss = optax.l2_loss(flattened_ownership_map * flattened_ownership_mask,
    #                          data.value.reshape(data.value.shape[0], -1) * flattened_ownership_mask)

    # mse_loss = jnp.sum(mse_loss, axis=-1) / jnp.sum(flattened_ownership_mask, axis=-1)

    # mse_loss = jnp.mean(mse_loss)
    # mse_weight = 50.
    # mse_loss = mse_loss * mse_weight

    # policy loss (KL(target_policy', agent_policy))
    action_logits = flatten_preds(action_logits)

    log_action_logits = jax.nn.log_softmax(action_logits, axis=-1)
    soft_action_logits = jax.nn.softmax(action_logits, axis=-1)

    flat_mask = construct_flat_mask(data)
    penalty_weight = 6.

    cross_entropy_loss = - jnp.sum(target_pr * log_action_logits, axis=-1)
    num_player = jnp.sum(target_pr[..., :361])
    num_opponent = jnp.sum(target_pr[..., 361:722])
    num_pass = jnp.sum(target_pr[..., 722])
    num_first = jnp.sum(data.num_moves_played == 0)
    ce_loss_player = jnp.where(num_player > 0, jnp.sum(
        jnp.where(jnp.sum(target_pr[..., :361], axis=-1) > 0, cross_entropy_loss, 0)) / num_player, 0)
    ce_loss_opponent = jnp.where(num_opponent > 0, jnp.sum(
        jnp.where(jnp.sum(target_pr[..., 361:722], axis=-1) > 0, cross_entropy_loss, 0)) / num_opponent, 0)
    ce_loss_pass = jnp.where(num_pass > 0,
                             jnp.sum(jnp.where(target_pr[..., 723] > 0, cross_entropy_loss, 0)) / num_pass, 0)
    ce_loss_root = jnp.where(num_first > 0,
                             jnp.sum(jnp.where(data.num_moves_played == 0, cross_entropy_loss, 0)) / num_first, 0)
    cross_entropy_loss = jnp.mean(cross_entropy_loss)

    # Penalty term for moves outside the mask
    action_logits_invalid = soft_action_logits * flat_mask
    invalid_moves_penalty = penalty_weight * jnp.sum(action_logits_invalid, axis=-1)
    invalid_moves_penalty = jnp.mean(invalid_moves_penalty)

    total_loss = cross_entropy_loss  # + invalid_moves_penalty
    # jax.debug.breakpoint()

    # return the total loss
    return total_loss, (net, (
        ce_loss_player, ce_loss_opponent, ce_loss_pass, ce_loss_root, invalid_moves_penalty, num_player, num_opponent,
        num_pass))  # was mse_loss + total_loss ... (mse_loss, )


@partial(jax.jit)  # pmap, axis_name="i")
def train_ownership_step(net, optim, data: TrainingOwnershipDatapoint):
    """A training step."""
    (_, (net, losses)), grads = jax.value_and_grad(ownership_loss_fn, has_aux=True)(net, data)
    # grads = jax.lax.pmean(grads, axis_name="i")
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses


@partial(jax.pmap, axis_name="i")
def test_ownership(net, data: TrainingOwnershipDatapoint):
    """Evaluation on test set."""
    (top_1_acc, top_2_acc), _ = jax.value_and_grad(calculate_metrics, has_aux=True)(net,
                                                                                    data)  # was: (top_1_acc, (top_2_acc, mse_loss))
    return top_1_acc, top_2_acc  # , mse_loss


def calculate_metrics(net, data: TrainingOwnershipDatapoint):
    net, (action_logits,) = batched_policy(net, (
        data.state, data.mask))  # was: net, (action_logits, ownership_map) =  # was also data.boardmask

    action_logits = flatten_preds(action_logits)
    top_1_acc = (action_logits.argmax(axis=1) == data.move.argmax(axis=1)).mean()
    top_2_acc = (jnp.argsort(action_logits, axis=1)[:, -2:] == data.move.argmax(axis=1)[..., None]).any(axis=1).mean()
    flattened_ownership_mask = data.mask.reshape(data.mask.shape[0], -1)
    # flattened_ownership_map = ownership_map.reshape(ownership_map.shape[0], -1)
    # mse_loss = optax.l2_loss(flattened_ownership_map * flattened_ownership_mask,
    #                          data.value.reshape(data.value.shape[0], -1) * flattened_ownership_mask)

    # mse_loss = jnp.sum(mse_loss, axis=-1) / jnp.sum(flattened_ownership_mask, axis=-1)
    # mse_loss = jnp.mean(mse_loss)

    return top_1_acc, top_2_acc  # was: return top_1_acc, (top_2_acc, mse_loss)


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


# def test_model(test_data, batch_size, model, optim, ownership_loss, policy_loss_pl, policy_loss_op, policy_loss_pass, policy_loss_root, invalid_moves_penalty, _stack_and_reshape, devices):
#     transfer_model = model.eval()
#     transfer_model, optim = jax.device_put_replicated((transfer_model, optim), devices)
#     accs1 = []
#     accs2 = []
#     mses = []
#     ids = range(0, len(test_data) - batch_size, batch_size)
#     with click.progressbar(ids, label="  test model   ") as progressbar:
#         for idx in progressbar:
#             batch = test_data[idx: (idx + batch_size)]
#             # I needed to move axis
#             batch = [construct_training_datapoint(d) for d in batch]
#             batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
#             top_1_acc, top_2_acc = test_ownership(transfer_model, batch)  # top_1_acc, top_2_acc, mse
#             mse = 0
#             accs1.append(top_1_acc)
#             accs2.append(top_2_acc)
#             mses.append(mse)
#
#     top_1_acc = np.mean(accs1)
#     top_2_acc = np.mean(accs2)
#     mse = np.mean(mses)
#
#     if ownership_loss is not None or policy_loss_pl is not None:
#         text_to_print = f"  ownership loss {ownership_loss:.3f}  policy losses: player {policy_loss_pl:.3f}, opponent {policy_loss_op:.3f}, pass {policy_loss_pass:.3f}, root {policy_loss_root:.3f} invalid moves penalty {invalid_moves_penalty:.3f}"
#     else:
#         text_to_print = ""
#     lr = optim[1][-1].learning_rate[0]
#     backbone_multiplier = optim[2].backbone_multiplier[0]
#     print(text_to_print +
#           f"  test top 1 acc {top_1_acc:.3f}"
#           f"  test top 2 acc {top_2_acc:.3f}"
#           f"  test ownership MSE {mse:.4f}"
#           f"  learning rate {lr:.1e}"
#           f"  backbone multiplier {backbone_multiplier:.1f}"
#           f"  time {datetime.datetime.now().strftime('%H:%M:%S')}"
#           )
#     return top_1_acc, mse, lr, backbone_multiplier


def plot_stats(filename, root_dir):
    with open(filename, "rb") as f:
        o_losses, p_losses_pl, p_losses_op, p_losses_pass, p_losses_root, i_losses, t1_accs, mses, lrs, bms, indices, loss_indices = pickle.load(
            f)
    os.makedirs(os.path.dirname(root_dir), exist_ok=True)
    run_name = os.path.basename(filename).rsplit('.', 1)[0]
    import matplotlib.pyplot as plt
    plt.xticks(indices)
    plt.plot(loss_indices, o_losses)
    plt.plot(loss_indices, p_losses_pl)
    plt.plot(loss_indices, p_losses_op)
    plt.plot(loss_indices, p_losses_pass)
    plt.plot(loss_indices, p_losses_root)
    plt.plot(loss_indices, i_losses)
    plt.plot(indices, t1_accs)
    plt.plot(indices, [mse * 500 for mse in mses])
    plt.plot(indices, [lr * 100 for lr in lrs])
    plt.plot(indices, bms)
    plt.ylim(0, 3)
    plt.legend(
        ['Ownership loss', 'Player loss', 'Opponent loss', 'Pass loss', 'Invalid loss', 'Top 1 accuracy', 'MSE * 500',
         'Learning rate * 100', 'Backbone multiplier'])
    plt.savefig(os.path.join(root_dir, f'metrics-{run_name}.png'))
    # print the list of floats with only 3 decimals
    print("Ownership losses:", ", ".join([f"{v_loss:.3f}" for v_loss in o_losses if v_loss is not None]))
    print("Player losses:", ", ".join([f"{p_loss:.3f}" for p_loss in p_losses_pl if p_loss is not None]))
    print("Opponent losses:", ", ".join([f"{p_loss:.3f}" for p_loss in p_losses_op if p_loss is not None]))
    print("Pass losses:", ", ".join([f"{p_loss:.3f}" for p_loss in p_losses_pass if p_loss is not None]))
    print("Root losses:", ", ".join([f"{p_loss:.3f}" for p_loss in p_losses_root if p_loss is not None]))
    print("Invalid moves penalties:", ", ".join([f"{i_loss:.3f}" for i_loss in i_losses if i_loss is not None]))
    print("Top 1 accuracies:", ", ".join([f"{t1_acc:.3f}" for t1_acc in t1_accs if t1_acc is not None]))
    print("MSEs:", ", ".join([f"{mse:.3f}" for mse in mses if mse is not None]))
    print("Learning rates:", ", ".join([f"{lr:.3f}" for lr in lrs if lr is not None]))
    print("Backbone multipliers:", ", ".join([f"{bm:.3f}" for bm in bms if bm is not None]))


@pax.pure
def train(
        trained_ckpt_filename: str,
        game_class="games.go_game.GoBoard9x9",
        agent_class="policies.resnet_policy.ResnetPolicyValueNet128",
        training_batch_size: int = 256,  # Originally 128 but maybe I'm getting OOM
        num_steps: int = 80_000,
        learning_rate: float = 1e-3 / 4,  # Originally 0.01,
        ckpt_filename: str = "go_agent_9x9_128_sym.ckpt",
        root_dir: str = ".",
        random_seed: int = 42,
        weight_decay: float = 1e-4,
        lr_decay_steps: int = 10_000,  # My full epoch is likely shorter than 100_000 steps
        backbone_lr_steps: int = 0,  # 3_000,  # 3_000, #15_000,  # was 25_000 in August
        # use_only_19x19: bool = True,
        only_prepared_data: bool = True,
):
    if root_dir == ".":
        root_dir = os.path.dirname(os.getcwd())

    """Train an agent by self-play."""
    env = import_class(game_class)()
    agent = import_class(agent_class)(
        input_dims=env.observation().shape,
        num_actions=env.num_actions(),
    )

    conf_file = 'analysis_config/a0kata_estimated_1024a.yaml'
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)
    config['evaluator_kwargs']['a0_ckpt'] = f'{trained_ckpt_filename}'
    benchmark_dir_small = 'assets/endgame-for-nerds-large-masks-no_solutions/01-value-of-move'
    positions_small = get_positions(benchmark_dir_small)
    benchmark_dir_big = 'assets/endgame-for-nerds-large-masks-no_solutions'
    positions_big = get_positions(benchmark_dir_big)
    max_depth = 10
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
            # w_init = jax.nn.initializers.normal(stddev=1.0 / 100.0)
            # w_rng_key = jax.random.PRNGKey(8)
            # dic.backbone.modules[0].weight = dic.backbone.modules[0].weight.at[0, 0, -1, :].set(
            #     w_init(w_rng_key, (128,)))
            agent = agent.load_state_dict(dic)
    elif os.path.isfile(trained_ckpt_path):
        print("Will load already pre-finetuned weights at", trained_ckpt_path)
    else:
        print("Not loading weight since no file was found at", ckpt_filename)

    def lr_schedule(step):
        e = jnp.floor(step * 1.0 / lr_decay_steps)
        return learning_rate * jnp.exp2(-e)

    def lr_backbone_schedule(step):
        return step > backbone_lr_steps

    transfer_model = TransferResnet(agent, include_boardmask=False)  # not use_only_19x19)
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
            start_iter = 1  # loaded_agent["iter"] + 1
            # optim = optim.load_state_dict(loaded_agent["optim"])
            transfer_model = transfer_model.load_state_dict(loaded_agent["agent"])

    rng_key = jax.random.PRNGKey(random_seed)
    shuffler = random.Random(random_seed)
    devices = jax.local_devices()
    num_devices = jax.local_device_count()

    def _stack_and_reshape(*xs):
        x = np.stack(xs)
        # x = np.reshape(x, (num_devices, -1) + x.shape[1:])
        return x

    print(f"  time {datetime.datetime.now().strftime('%H:%M:%S')}")

    ownership_loss, policy_loss_pl, policy_loss_op, policy_loss_pass, policy_loss_root, invalid_moves_penalty = None, None, None, None, None, None

    o_losses, p_losses_pl, p_losses_op, p_losses_pass, p_losses_root, i_losses, t1_accs, mses, lrs, bms, indices, loss_indices = [], [], [], [], [], [], [], [], [], [], [], []
    stats_pickle_name = trained_ckpt_path.rsplit('.', 1)[0] + '_stats.pkl'

    if only_prepared_data:
        prepared_dir = 'prep'
        pickle_files = [os.path.join(prepared_dir, f) for f in os.listdir(prepared_dir)]
    else:
        training_data_dir = '../KifuMining/endgame_data/processed'
        pickle_files = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir)]

    chunk_iterator = itertools.cycle(enumerate(pickle_files[:80], 1))

    datapoints_iterator = iter([])

    losses = []
    batch = []
    log_period = 10
    checkpoint_period = 500
    small_benchmark_period = 2000
    big_benchmark_period = 10_000
    tqdm_bar = tqdm(range(1, num_steps + 1))
    epoch_num = 0

    def get_new_datapoints_iterator(custom_chunk_iterator):

        file_num, pickle_file = next(custom_chunk_iterator)
        # print(file_num, pickle_file)
        try:
            datapoints_iterator = iter(ChunkData(pickle_file, cache=True, only_prepared_data=only_prepared_data))
            return datapoints_iterator, file_num
        except:
            gc.collect()
            print(f"Error unpickling {pickle_file}")
            traceback.print_exc()
            return get_new_datapoints_iterator(custom_chunk_iterator)

    for step in tqdm_bar:
        for _ in range(training_batch_size):
            try:
                sample = next(datapoints_iterator)
            except StopIteration:

                del datapoints_iterator
                gc.collect()
                time.sleep(10)
                datapoints_iterator, file_num = get_new_datapoints_iterator(chunk_iterator)

                if file_num == 1:
                    epoch_num += 1
                try:
                    sample = next(datapoints_iterator)
                except:
                    # This shouldn't happen but just in case...
                    continue
            # sample = TrainingOwnershipDatapoint(**{key: jnp.array(value) for key, value in sample.items()})
            batch.append(sample)
            if len(batch) == training_batch_size:
                batch = jax.tree_util.tree_map(_stack_and_reshape, *batch)
                transfer_model, optim, loss = train_ownership_step(transfer_model, optim, batch)
                losses.append(loss)
                batch = []
        if len(losses) == log_period:
            policy_loss_pl, policy_loss_op, policy_loss_pass, policy_loss_root, invalid_moves_penalty, num_player, num_opponent, num_pass = zip(
                *losses)  # was: ownership_loss, ...
            # ownership_loss = np.mean(sum(jax.device_get(ownership_loss))) / len(ownership_loss)
            ownership_loss = 0
            policy_loss_pl = np.mean(sum(jax.device_get(policy_loss_pl))) / len(policy_loss_pl)
            policy_loss_op = np.mean(sum(jax.device_get(policy_loss_op))) / len(policy_loss_op)
            policy_loss_pass = np.mean(sum(jax.device_get(policy_loss_pass))) / len(policy_loss_pass)
            policy_loss_root = np.mean(sum(jax.device_get(policy_loss_root))) / len(policy_loss_root)
            invalid_moves_penalty = np.mean(sum(jax.device_get(invalid_moves_penalty))) / len(invalid_moves_penalty)
            num_player = np.mean(sum(jax.device_get(num_player))) / len(num_player)
            num_opponent = np.mean(sum(jax.device_get(num_opponent))) / len(num_opponent)
            num_pass = np.mean(sum(jax.device_get(num_pass))) / len(num_pass)
            o_losses.append(float(ownership_loss))
            p_losses_pl.append(float(policy_loss_pl))
            p_losses_op.append(float(policy_loss_op))
            p_losses_pass.append(float(policy_loss_pass))
            p_losses_root.append(float(policy_loss_root))
            i_losses.append(float(invalid_moves_penalty))
            loss_indices.append(int(step))
            # print(f"Avg number of examples with move of player on turn: {num_player:.1f} oppponent: {num_opponent:.1f} pass: {num_pass:.1f}")
            text_to_print = f"[Epoch {epoch_num} Chunk {file_num} Step {step}] pl {policy_loss_pl:.3f}, opp {policy_loss_op:.3f}, pass {policy_loss_pass:.3f} , root {policy_loss_root:.3f} inv {invalid_moves_penalty:.3f}"
            tqdm_bar.set_description(text_to_print)
            losses = []
            # gc.collect()
        if step % checkpoint_period == 0:
            print(
                f"[Epoch {epoch_num} Chunk {file_num} Step {step}] pl {policy_loss_pl:.3f}, opp {policy_loss_op:.3f}, pass {policy_loss_pass:.3f}, root {policy_loss_root:.3f}")  # inv {invalid_moves_penalty:.3f}")
            with open(stats_pickle_name, "wb") as f:
                pickle.dump((o_losses, p_losses_pl, p_losses_op, p_losses_pass, p_losses_root, i_losses, t1_accs, mses,
                             lrs, bms, indices, loss_indices), f)
            save_model(trained_ckpt_path, transfer_model, step)
        if step % big_benchmark_period == 0:
            results = run_benchmark(positions_big, config, max_depth, benchmark_dir_big,
                                    f'{trained_ckpt_filename}-{step}', verbose=False)
            if results is not None:
                print(
                    f"Big benchmark: {results['correct_count']}/{results['num_positions']} correct, {results['error_count']} errors")
            else:
                print("Big benchmark failed")
        elif step % small_benchmark_period == 0:
            results = run_benchmark(positions_small, config, max_depth, benchmark_dir_small,
                                    f'{trained_ckpt_filename}-{step}', verbose=False)
            if results is not None:
                print(
                    f"Small benchmark: {results['correct_count']}/{results['num_positions']} correct, {results['error_count']} errors")
            else:
                print("Small benchmark failed")

    return

    # Earlier I also ran tests on a test dataset but it makes little sense,
    # given that in training the model rarely sees same data,
    # and that benchmark on GoMagic endgame problems is a much better metric
    if iteration % 20 == 2 or iteration in (1,):  # 6, 11, 16):  # was (1, 6, 11, 16)
        top_1_acc, mse, lr, backbone_multiplier = test_model(test_data, training_batch_size, transfer_model, optim,
                                                             ownership_loss, policy_loss_pl, policy_loss_op,
                                                             policy_loss_pass, policy_loss_root,
                                                             invalid_moves_penalty, _stack_and_reshape, devices)
        t1_accs.append(float(top_1_acc))
        mses.append(float(mse))
        lrs.append(float(lr))
        bms.append(float(backbone_multiplier))
        indices.append(float(iteration - 1))
    plot_stats(stats_pickle_name, os.path.join(root_dir, 'stats'))



def profiling_decorator(requested_profiling):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if requested_profiling:
                import cProfile, pstats
                profiler = cProfile.Profile()
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                stats = pstats.Stats(profiler)
                stats.sort_stats("cumulative").print_stats(50)
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ChunkData:
    @profiling_decorator(False)
    def __init__(self, pickle_file, cache=False, only_prepared_data=True):
        if only_prepared_data:
            with open(pickle_file, 'rb') as f:
                self.items = pickle.load(f)
        else:
            prep_path = os.path.join('prep', os.path.basename(pickle_file[:-4]) + '-prep.pkl')
            if os.path.isfile(prep_path):
                with open(prep_path, 'rb') as f:
                    self.items = pickle.load(f)
            else:
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                self.data = list(data.values())
                # random.shuffle(self.data)

                stats = []
                for datapoint in self.data:
                    lp = datapoint.get('local_pos', None)
                    if lp is None:
                        continue
                    stats.extend(list(lp.values()))

                sizes = np.array([s['size'] for s in stats])
                mtls = np.array([s['moves_till_end'] for s in stats])
                lms = []
                for s in stats:
                    lms.extend(s.get('local_moves', []) or [])
                ims = np.array([l[1] for l in lms])
                size_counts = np.unique(sizes, return_counts=True, equal_nan=False)
                mtls_counts = np.unique(mtls, return_counts=True, equal_nan=False)
                ims_counts = np.unique(ims, return_counts=True, equal_nan=False)

                sente_gote_balance = 1.
                ims_factor = ims_counts[1][ims_counts[0] == True][0] / ims_counts[1][ims_counts[0] == False][0]
                if ims_factor < sente_gote_balance:
                    self.sente_multiplier, self.gote_multiplier = 1., sente_gote_balance * ims_factor
                else:
                    self.sente_multiplier, self.gote_multiplier = 1. / ims_factor, sente_gote_balance

                desired_size = 4
                count_of_desired_size = np.max(size_counts[1][size_counts[0] >= desired_size])
                self.size_balances = {1: .5, 2: .75, 3: 1.}
                self.size_factors = {
                    i: min(self.size_balances[i] / (size_counts[1][size_counts[0] == i][0] / count_of_desired_size), 1.)
                    for i in self.size_balances
                }

                desired_mtl = 4
                count_of_desired_mtl = np.max(mtls_counts[1][mtls_counts[0] >= desired_mtl])
                self.mtl_balances = {0: .2, 1: .5, 2: .75, 3: 1.}
                self.mtl_factors = {
                    i: min(self.mtl_balances[i] / (mtls_counts[1][mtls_counts[0] == i][0] / count_of_desired_mtl), 1.)
                    for i in self.mtl_balances
                }
                self.items = self.make_items()
                if cache:
                    with open(prep_path, 'wb') as f:
                        pickle.dump(self.items, f)
        self.num_items = len(self.items)
        random.shuffle(self.items)
        self.current = -1
        self.high = len(self.items)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.high:
            return self.items[self.current]
        raise StopIteration

    def make_items(self):
        items = []
        for datapoint in self.data:
            prev_pos = datapoint.get('prev_pos', None)
            # print(prev_pos.dtype)
            segments = datapoint.get('segments', None)
            # print(segments.dtype)
            distances = datapoint.get('distances', None)
            local_pos = datapoint.get('local_pos', None)
            if any(x is None for x in (prev_pos, segments, distances, local_pos)):
                print('Malformed datapoint', datapoint.keys())
                continue
            prev_pos = prev_pos.astype(np.int8)
            segments = segments.astype(np.uint8)
            distances = distances.astype(np.uint8)
            for identifier, lp in local_pos.items():
                size_factor = self.size_factors.get(lp['size'], 1.)
                mtl = lp['moves_till_end']
                next_pos = lp['next_pos']
                if next_pos is None or len(next_pos) == 0:
                    next_pos = np.zeros((0, 19, 19), dtype=np.int8)
                # print(next_pos.dtype)

                # This way we won't show to the network positions with 0 moves played, and pass as a target
                # But it shouldn't matter, I think
                local_moves = lp.get('local_moves', [])
                if len(local_moves) < 8:
                    local_moves += [(None, False, 0)]
                for i, (move, sente, dist) in enumerate(local_moves, 1):
                    mtl_factor = self.mtl_factors.get(mtl, 1.)
                    sente_factor = 1.
                    if move is not None:
                        sente_factor = self.sente_multiplier if sente else self.gote_multiplier
                    if random.random() < size_factor * mtl_factor * sente_factor:
                        try:
                            training_datapoint = self.prepare_training_datapoint(prev_pos, segments, distances,
                                                                                 identifier, next_pos, sente, i)
                        except AssertionError as e:
                            raise e
                            continue
                        items.append(training_datapoint)
                    mtl -= 1
        return items

    def prepare_training_datapoint(self, prev_pos, segments, distances, identifier, next_pos, sente, i):
        random_prob = np.random.rand()
        random_binary = np.random.choice([0, 1], size=(19, 19), p=[random_prob, 1 - random_prob]).astype(np.uint8)

        min_num_prev_pos_to_use = 2 if (sente and i == 1) else 1
        max_num_prev_pos_to_use = 9 - i
        num_prev_pos_to_use = random.randint(min_num_prev_pos_to_use, max_num_prev_pos_to_use)
        num_replicated_first_pos = max_num_prev_pos_to_use - num_prev_pos_to_use
        replicated_first_pos = np.stack([prev_pos[
                                             num_replicated_first_pos]] * num_replicated_first_pos) if num_replicated_first_pos > 0 else np.zeros(
            (0, 19, 19), dtype=np.int8)
        full_pos = np.concatenate((replicated_first_pos, prev_pos[-num_prev_pos_to_use:], next_pos[:i - 1]))
        has_next_move = len(next_pos) >= i
        mask_dist = random.randint(0, 5)
        border_mask = np.logical_and(segments == identifier, distances == mask_dist).astype(np.uint8)
        updated_border_mask = np.logical_and(border_mask, random_binary).astype(np.uint8)
        if np.all(updated_border_mask == 0):
            updated_border_mask = border_mask
        mask = np.logical_and(segments == identifier, distances < mask_dist).astype(np.uint8)
        mask = np.logical_or(mask, updated_border_mask).astype(np.uint8)
        cur_arrangement = full_pos[-1]
        prev_arrangement = full_pos[-2] if not (num_prev_pos_to_use == 1 and i == 1) else prev_pos[-2]
        _, prev_color = self.detect_move(prev_arrangement, cur_arrangement)
        if prev_color == 0:
            prev_color = 1
        color_to_play = -prev_color
        # assert prev_color != 0, f'{np.any(cur_arrangement != prev_arrangement)}, {(num_prev_pos_to_use == 1 and i == 1)}'
        if has_next_move:
            target_arrangement = next_pos[i - 1]
            move_coords, move_color = self.detect_move(cur_arrangement, target_arrangement)
        else:
            move_coords, move_color = (0, 0), 0

        # randomly remove distant stones
        if np.random.rand() < 0.8:
            random_mask = np.logical_or(random_binary, segments == identifier)
            full_pos = np.where(random_mask[None, :, :], full_pos, 0)

        full_pos = np.concatenate((full_pos, np.ones((1, 19, 19), dtype=np.int8)))
        full_pos *= color_to_play
        full_pos = np.moveaxis(full_pos, 0, -1)
        move_color *= color_to_play
        move = construct_move_target(has_next_move, move_coords, move_color)
        # return {
        #     'state': full_pos.astype(np.int8),
        #     'mask': mask.astype(np.uint8),
        #     'num_moves_played': i - 1,
        #     'color_to_move': np.array(color_to_play).astype(np.int8),
        #     'move': np.array(move).astype(np.uint8),
        # }
        return TrainingOwnershipDatapoint(
            state=jnp.array(full_pos.astype(np.int8)),  # .astype(jnp.int8),
            mask=jnp.array(mask.astype(np.uint8)),  # .astype(jnp.uint8),
            # next_move_coords=jnp.array(move_coords),
            # next_move_color=jnp.array(move_color),
            num_moves_played=i - 1,
            color_to_move=jnp.array(color_to_play),  # .astype(jnp.int8),
            move=jnp.array(move.astype(np.uint8)),  # .astype(jnp.uint8),
            # has_next_move=jnp.array(has_next_move),
        )

    @staticmethod
    def detect_move(prev_pos: np.ndarray, pos: np.ndarray):
        assert prev_pos.shape == pos.shape, "Shapes must be equal"
        assert prev_pos.ndim == 2, "Arrays must be 2D"
        new_stones = np.logical_and(pos != 0, prev_pos == 0)
        x, y = np.unravel_index(np.argmax(abs(new_stones), axis=None), new_stones.shape)
        return (x, y), pos[x, y]


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


if __name__ == "__main__":
    if "COLAB_TPU_ADDR" in os.environ:
        jax.tools.colab_tpu.setup_tpu()
    print("Cores:", jax.local_devices())

    fire.Fire(train)
