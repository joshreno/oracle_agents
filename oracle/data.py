from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from enum import IntEnum
from glob import glob
import itertools
import re

import numpy as np
import pandas as pd


block_names = [
    'Ground', 'Stair', 'Treetop', 'Block', 'Bar', 'Koopa', 'Koopa 2',
    'PipeBody', 'Pipe', 'Question', 'Coin', 'Goomba', 'CannonBody',
    'Cannon', 'Lakitu', 'Bridge', 'Hard Shell', 'SmallCannon', 'Plant'
]
decoration_names = [
    'Waves', 'Hill', 'Castle', 'Snow Tree 2', 'Cloud 2', 'Cloud', 'Bush',
    'Tree 2', 'Bush 2', 'Tree', 'Snow Tree', 'Fence', 'Bark', 'Flag', 'Mario'
]
action_names = ['Nothing'] + block_names + decoration_names[:-2]
Block = IntEnum('Block', block_names, start=1)
Decoration = IntEnum('Decoration', decoration_names, start=1+len(block_names))
Action = IntEnum('Action', action_names, start=0)


def is_block(tile_name):
    return tile_name in Block.__members__


def is_decoration(tile_name):
    return tile_name in Decoration.__members__


def read_total_results():
    results = pd.read_csv("./totalResults.csv")
    results = results[['id', 'First_Reuse']]
    results['id'] = results['id'].map(lambda id: '{:04}'.format(id))
    results1 = results.copy()
    results2 = results.copy()
    results1['id'] += '-1'
    results2['id'] += '-2'
    results2['First_Reuse'] *= -1
    results = results1.append(results2, ignore_index=True).set_index('id')
    results.loc['2106-2b'] = -1
    return results


def parse_log_file(filename):
    with open(filename) as file:
        content = file.readlines()
    if not content or "Study Start" not in content[0]:
        return []
    time_logged_re = re.compile(r'(?P<action>.*):\s*time = (?P<time>.*)')
    added_re = re.compile(r'Added (?P<tile>.*) at (?P<x>\d+), (?P<y>\d+)')
    # states and actions composed of (block, decoration) tuples of coord dicts
    state = {}, {}
    prev_state = {}, {}
    action = {}, {}
    transitions = []
    player_control = True
    design_mode = True
    last_hit = 0
    for line in content:
        line = line.strip()
        match = time_logged_re.match(line)
        logged_action = line if match is None else match.group('action')
        time = np.NINF if match is None else float(match.group('time'))
        design_mode = design_mode or last_hit > 0 and time > last_hit
        if logged_action.startswith("Added"):
            match = added_re.match(logged_action)
            if match is None:
                continue
            tile = match.group('tile')
            coord = int(match.group('x')), int(match.group('y'))
            substate = state[0] if is_block(tile) else state[1]
            subaction = action[0] if is_block(tile) else action[1]
            tile = Block[tile] if is_block(tile) else Decoration[tile]
            if design_mode and player_control:
                if coord in substate and substate[coord] == tile:
                    del substate[coord]
                else:
                    substate[coord] = tile
            elif design_mode and not player_control:
                subaction[coord] = tile
                substate[coord] = tile
        elif logged_action.startswith("Ended Turn"):
            transitions.append((prev_state, action, state))
            player_control = False
            prev_state = deepcopy(state)
            action = {}, {}
        elif logged_action.startswith("Player Turn"):
            player_control = True
        elif logged_action.startswith("Starting Run"):
            design_mode = False
            last_hit = 0
        elif logged_action.startswith("Grid Cleared"):
            last_hit = time
        else:
            pass
    if action[0] or action[1]:
        transitions.append((prev_state, action, state))
    return transitions


def distribute_reward(data, score, gamma=0.1):
    training_sequence = []
    for i, (state, action, next_state) in enumerate(reversed(data)):
        X_0 = sparse_to_dense(state[0], (100, 15), dtype=np.uint8)
        X_1 = sparse_to_dense(state[1], (100, 15), dtype=np.uint8)
        X = np.stack((X_0, X_1), axis=-1)
        Y = np.zeros((100, 15, len(Action)), dtype=np.float32)
        # For blocks and decorations
        for substate, subaction, subnextstate in zip(state, action, next_state):
            for coord in itertools.product(range(100), range(15)):
                state_at_coord = substate.get(coord, None)
                action_at_coord = subaction.get(coord, None)
                next_state_at_coord = subnextstate.get(coord, None)
                pred_state_at_coord = None if state_at_coord == action_at_coord else action_at_coord
                Q = score * (gamma ** i)
                if action_at_coord is None and next_state_at_coord is None:
                    Q -= 0.1
                if action_at_coord is None and next_state_at_coord is not None:
                    Q += 0.1
                if action_at_coord is not None and pred_state_at_coord != next_state_at_coord:
                    # the final block didn't match our prediction
                    # the creator made unexpected changes or undid our changes
                    Q -= 0.1
                action_at_coord = Action.Nothing if action_at_coord is None else action_at_coord
                Y[coord][action_at_coord] = Q
        training_sequence.append((X, Y))
    return training_sequence


def sparse_to_dense(coord_dict, dims, default=0, dtype=np.float32):
    dense = np.full(dims, default, dtype=dtype)
    for coord in coord_dict:
        dense[coord] = coord_dict[coord]
    return dense


def train_test_split(data_by_player):
    test_ids = {
        '0215', '1206', '2114', '2005', '1204', '2101',
        '2001', '1200', '1002', '1203', '0103'
    }
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for player_id, player_data in data_by_player.items():
        X, Y = tuple(zip(*player_data))
        if player_id in test_ids:
            X_test += X
            Y_test += Y
        else:
            X_train += X
            Y_train += Y
    X_train = np.stack(X_train)
    Y_train = np.stack(Y_train)
    X_test = np.stack(X_test)
    Y_test = np.stack(Y_test)
    return (X_train, Y_train), (X_test, Y_test)


def create_player_training_data(args):
    filename, score = args
    data = parse_log_file(filename)
    data = distribute_reward(data, score)
    return data


def global_to_local_focus(X_batch, Y_batch):
    local_X_batch = []
    local_Y_batch = []
    for X, Y in zip(X_batch, Y_batch):
        min_x = np.inf
        max_x = 0
        for coord in itertools.product(range(100), range(15)):
            max_value_index = Y[coord].argmax()
            min_value_index = Y[coord].argmin()
            max_value = Y[coord][max_value_index]
            min_value = Y[coord][min_value_index]
            if (max_value > 0 and max_value_index > 0) or (min_value < 0 and min_value_index > 0):
                min_x = min(min_x, coord[0])
                max_x = max(max_x, coord[0])
        if min_x == np.inf:
            x_start = np.random.randint(60+1)
            x_end = x_start + 40
            local_X_batch.append(X[x_start:x_end])
            local_Y_batch.append(Y[x_start:x_end])
        elif max_x - min_x > 40:
            for i in range(max_x - min_x - 40 + 1):
                x_start = i
                x_end = x_start + 40
                local_X_batch.append(X[x_start:x_end])
                local_Y_batch.append(Y[x_start:x_end])
        else:
            x_start = (min_x + max_x) // 2 - 20
            x_start = max(0, x_start)
            x_start = min(60, x_start)
            x_end = x_start + 40
            local_X_batch.append(X[x_start:x_end])
            local_Y_batch.append(Y[x_start:x_end])
    return np.stack(local_X_batch), np.stack(local_Y_batch)

def one_hot_encoding(X, n_classes):
    return (np.expand_dims(X, -1) == np.arange(n_classes)).astype(np.uint8)

def retrieve_dataset(local=True, one_hot=True):
    total_results = read_total_results()
    filenames = glob(r"./Log Data/*/*.txt")
    filename_id_re = re.compile(r'(.*[/\\])?(.*)\.txt$')
    player_ids = [filename_id_re.match(name).group(2) for name in filenames]
    player_ids, filenames = zip(*(t for t in zip(player_ids, filenames) if t[0] in total_results.index))
    scores = [total_results.loc[id]['First_Reuse'] for id in player_ids]
    # load and process files in parallel since loading files is disk-bound
    with ThreadPoolExecutor() as executor:
        training_data = executor.map(create_player_training_data,
                                     zip(filenames, scores))

    data_by_player = {}
    for player_id, player_training_data in zip(player_ids, training_data):
        if player_training_data:
            base_id = player_id.split('-')[0]
            if base_id not in data_by_player:
                data_by_player[base_id] = []
            data_by_player[base_id] += player_training_data

    # approx 90-10 split
    (X_train, Y_train), (X_test, Y_test) = train_test_split(data_by_player)

    if local:
        # go from 100 x 15 to 40 x 15 windows
        X_train, Y_train = global_to_local_focus(X_train, Y_train)
        X_test, Y_test = global_to_local_focus(X_test, Y_test)

    if one_hot:
        # categorical -> one-hot encoding
        X_train = np.concatenate([one_hot_encoding(X_train[:, :, :, 0], len(Block)),
                                  one_hot_encoding(X_train[:, :, :, 1], len(Decoration))],
                                 axis=-1)
        X_test = np.concatenate([one_hot_encoding(X_test[:, :, :, 0], len(Block)),
                                 one_hot_encoding(X_test[:, :, :, 1], len(Decoration))],
                                axis=-1)

    return (X_train, Y_train), (X_test, Y_test)


def main():
    (X_train, Y_train), (X_test, Y_test) = retrieve_dataset()
    assert X_train.shape[1:] == (40, 15, len(Block) + len(Decoration))
    assert X_test.shape[1:] == (40, 15, len(Block) + len(Decoration))
    assert Y_train.shape[1:] == (40, 15, 33)
    assert Y_test.shape[1:] == (40, 15, 33)
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert np.isfinite(Y_train).all()
    assert np.isfinite(Y_test).all()


if __name__ == '__main__':
    main()
