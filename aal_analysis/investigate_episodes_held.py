"""
Load episodes from replay buffer.
Quantify features of the episodes (such as how many magenta pixels).
See download_ckpt.sh to get necessary files from remote server for running this locally.
"""
import functools
import os
import time

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import argparse

from os.path import expanduser

from tqdm import tqdm


def run():
    """
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-416
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-417
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-418
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-419
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-420
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-421
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-422
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-423
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-463
    /home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-464
    :return:
    """

    group = 'dv2_crafter_g08_a07_losses'

    priority_checkpoint = '910000'
    run_length = 910000

    run_list = ['DRF-416', 'DRF-417', 'DRF-418', 'DRF-419', 'DRF-420', 'DRF-421', 'DRF-422', 'DRF-423',
                'DRF-463', 'DRF-464']

    avg_full_priorities = np.zeros((len(run_list), run_length))
    total_resource_priorities = {}

    for r, run_name in enumerate(run_list):
        run_resource_priorities = {}

        print(f'Starting {run_name}...')

        priorities_npz = f'/home/cd/remote-download/icml2023crafter/{group}/{run_name}/train_episodes_priority.npz_{priority_checkpoint}'
        priorities_pkl = f'/home/cd/remote-download/icml2023crafter/{group}/{run_name}/train_episodes_priority.npz_{priority_checkpoint}.pkl'
        priorities_npz_loaded = np.load(priorities_npz)
        priorities_pkl_loaded = pickle.load(open(priorities_pkl, 'rb'))

        traj_dir = f'/home/cd/remote-download/icml2023crafter/{group}/{run_name}/train_episodes'
        traj_names = os.listdir(traj_dir)
        traj_names.sort()

        idx_from_file, priorities = reform_priority(priorities_npz_loaded, priorities_pkl_loaded)
        get_priority_2_fn = functools.partial(get_priority_2, idx_from_file, priorities, run_name)

        past_the_end_of_priority_array = False

        avg_full_priorities[r, :] = priorities_npz_loaded['arr'][:run_length]

        for episode_num, traj_npz in enumerate(traj_names):
            # Load episodes
            ep, traj_name, traj_timestamp = load_ep(traj_dir, traj_npz)

            start = time.time()
            imgs = ep['image']
            has_wood = has_resource('wood', imgs)
            has_wood_pickaxe = has_resource('wood_pickaxe', imgs)
            has_stone = has_resource('stone', imgs)
            has_stone_pickaxe = has_resource('stone_pickaxe', imgs)
            # has_coal = has_resource('coal', imgs)
            has_iron = has_resource('iron', imgs)
            # has_wood_sword = has_resource('wood_sword', imgs)
            has_stone_sword = has_resource('stone_sword', imgs)
            has_iron_sword = has_resource('iron_sword', imgs)
            end = time.time()
            #print(f'has_resource in {end-start} seconds')

            has_resource_dict = {
                'wood': has_wood,
                'wood_pickaxe': has_wood_pickaxe,
                'stone': has_stone,
                'stone_pickaxe': has_stone_pickaxe,
                'iron': has_iron,
                'stone_sword': has_stone_sword,
                'iron_sword': has_iron_sword,
            }

            has_resource_indices = {resource: np.where(has_resource_dict[resource])[0] for resource in has_resource_dict}

            if has_stone_sword.sum() > 0:
                print(f'Found stone sword in {traj_name}')
            if has_iron_sword.sum() > 0:
                print(f'Found iron sword in {traj_name}')

            for resource, resource_index in has_resource_indices.items():
                prios = get_priority_2_fn(traj_npz, resource_index)
                if prios is None:
                    past_the_end_of_priority_array = True
                    break

                if resource not in run_resource_priorities.keys():
                    run_resource_priorities[resource] = []

                run_resource_priorities[resource].append(prios)

            if past_the_end_of_priority_array:
                print(f'Past the end of priority array at episode {episode_num}')
                break

        for resource, run_priorities in run_resource_priorities.items():
            if resource not in total_resource_priorities:
                total_resource_priorities[resource] = []
            scaled_run_priorities = (np.mean(np.concatenate(run_priorities)) /
                                     np.mean(avg_full_priorities[r, :]))
            total_resource_priorities[resource].append(scaled_run_priorities)

    for r, p in total_resource_priorities.items():
        print(f'---')
        if len(p) == 0:
            continue
        print(f'Number of {r} steps: {len(p)}')
        print(f'Mean: {np.nanmean(p)}')
        print(f'Median: {np.nanmedian(p)}')
        print(f'Min: {np.nanmin(p)}')
        print(f'Max: {np.nanmax(p)}')
        print(f'Std: {np.nanstd(p)}')
        print(f'Count non-NaN: {np.count_nonzero(~np.isnan(p))}')

    print(' ')


    print(f'Total steps in dataset: {len(avg_full_priorities.flatten())}')
    print(f'Mean: {np.mean(avg_full_priorities)}')
    print(f'Median: {np.median(avg_full_priorities)}')
    print(f'Min: {np.min(avg_full_priorities)}')
    print(f'Max: {np.max(avg_full_priorities)}')
    print(f'Std: {np.std(avg_full_priorities)}')

    print('Done.')

def load_ep(traj_dir, which_ep, timestamp=None):
    """Load an episode from replay buffer"""
    # print(f'{basedir}, Ep: {which_ep}')
    traj_names = os.listdir(traj_dir)
    traj_names.sort()
    if timestamp is None:
        if isinstance(which_ep, str):
            traj_name = which_ep
        else:
            traj_name = traj_names[which_ep]
    else:
        traj_name = [x for x in traj_names if timestamp in x]
        traj_name = traj_name[0]
    #print(traj_name)
    traj_timestamp = traj_name.split('-')[0]
    traj_path = f'{traj_dir}/{traj_name}'
    ep = np.load(traj_path)
    return ep, traj_name, traj_timestamp


def has_resource(resource, images):
    image_size = 64
    blocks = 9
    x_offset, y_offset = 0, 0

    if resource == 'wood':
        block_x = 6
        block_y = 8
    elif resource == 'stone':
        block_x = 7
        block_y = 8
    elif resource == 'coal':
        block_x = 8
        block_y = 8
    elif resource == 'iron':
        block_x = 9
        block_y = 8
    elif resource == 'wood_pickaxe': # guess
        block_x = 2
        block_y = 9
        x_offset = -2
    elif resource == 'stone_pickaxe': # guess
        block_x = 3
        block_y = 9
        x_offset = -2
    elif resource == 'iron_pickaxe': # guess
        block_x = 4
        block_y = 9
    elif resource == 'wood_sword': # guess
        block_x = 5
        block_y = 9
    elif resource == 'stone_sword': # guess
        block_x = 6
        block_y = 9
    elif resource == 'iron_sword': # guess
        block_x = 7
        block_y = 9
    else:
        raise ValueError(f'Unknown resource {resource}')

    def block_to_pixel(block, offset=0):
        block_size = image_size // blocks
        return ((block - 1) * block_size + block_size // 2) + offset

    def check_pixel_not_black(images, col, row):
        black_pixel = np.array([0, 0, 0], dtype=np.uint8)
        not_black = (images[:, row, col] != black_pixel).any(axis=1).astype(np.uint8)
        return not_black

    pixel_x, pixel_y = block_to_pixel(block_x, x_offset), block_to_pixel(block_y, y_offset)

    result = check_pixel_not_black(images, pixel_x, pixel_y)

    return result


def get_priority(priority_npz_loaded, priority_pkl_loaded, run_name, episode_name, step):
    prefix = f'/home/ikauvar/logs/{run_name}/train_episodes'
    try:
        idx = priority_pkl_loaded['idx_from_ep'][f'{prefix}/{episode_name}:{step}']
    except KeyError:
        print(f'Could not find {episode_name}:{step} in priority pkl')
        return None

    return priority_npz_loaded['arr'][idx]


def reform_priority(priorities_npz_loaded, priority_pkl_loaded):
    keys = list(priority_pkl_loaded['idx_from_ep'].keys())
    values = priorities_npz_loaded['arr']

    split_keys = [x.split(':') for x in keys]
    split_keys = [[x[0], int(x[1])] for x in split_keys]
    # zip
    split_file, split_step = list(zip(*split_keys))

    # create dictionary of index where each file starts
    idx_from_file = {}
    for i, file in enumerate(split_file):
        if file not in idx_from_file:
            idx_from_file[file] = i

    priorities = np.array(values)

    return idx_from_file, priorities


def get_priority_2(idx_from_file, priorities, run_name, episode_name, indices):
    prefix = f'/home/ikauvar/logs/{run_name}/train_episodes'

    try:
        offset = idx_from_file[f'{prefix}/{episode_name}']
    except KeyError:
        print(f'Could not find {episode_name} in priority pkl')
        return None

    try:
        result = priorities[offset + indices - 1]
    except IndexError:
        print(f'Could not find {episode_name}:{indices} in priority pkl')
        return None
    return result

def get_avg_episode_priority(priority_npz_loaded, priority_pkl_loaded, run_name, episode_name, episode_length):
    prefix = f'/home/ikauvar/logs/{run_name}/train_episodes'
    start_idx = priority_pkl_loaded['idx_from_ep'][f'{prefix}/{episode_name}:1']
    end_idx = priority_pkl_loaded['idx_from_ep'][f'{prefix}/{episode_name}:{episode_length}']

    return np.mean(priority_npz_loaded['arr'][start_idx:end_idx])


if __name__ == "__main__":
    run()

