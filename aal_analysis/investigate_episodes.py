"""
Load episodes from replay buffer.
Quantify features of the episodes (such as how many magenta pixels).
See download_ckpt.sh to get necessary files from remote server for running this locally.
"""
import functools
import os
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
    priority_checkpoint = '1010000'
    run_list = ['DRF-416', 'DRF-417', 'DRF-418', 'DRF-419', 'DRF-420', 'DRF-421', 'DRF-422', 'DRF-423',
                'DRF-463', 'DRF-464']
    run_length = 1000000

    iron_priorities = []
    norm_iron_priorities = []
    avg_full_priorities = np.zeros((len(run_list), run_length))

    for r, run_name in enumerate(run_list):

        print(f'Starting {run_name}...')

        priorities_npz = f'/home/cd/remote-download/icml2023crafter/{group}/{run_name}/train_episodes_priority.npz_{priority_checkpoint}'
        priorities_pkl = f'/home/cd/remote-download/icml2023crafter/{group}/{run_name}/train_episodes_priority.npz_{priority_checkpoint}.pkl'
        priorities_npz_loaded = np.load(priorities_npz)
        priorities_pkl_loaded = pickle.load(open(priorities_pkl, 'rb'))

        traj_dir = f'/home/cd/remote-download/icml2023crafter/{group}/{run_name}/train_episodes'
        traj_names = os.listdir(traj_dir)
        traj_names.sort()

        get_priority_fn = functools.partial(get_priority, priorities_npz_loaded, priorities_pkl_loaded, run_name)

        past_the_end_of_priority_array = False

        avg_full_priorities[r, :] = priorities_npz_loaded['arr'][:run_length]
        mean_priority = priorities_npz_loaded['arr'][:run_length].mean()

        for episode_num, traj_npz in tqdm(enumerate(traj_names), total=len(traj_names)):
            # Load episodes
            ep, traj_name, traj_timestamp = load_ep(traj_dir, traj_npz)

            imgs = ep['image']
            has_wood = has_resource('wood', imgs)
            has_stone = has_resource('stone', imgs)
            has_coal = has_resource('coal', imgs)
            has_iron = has_resource('iron', imgs)

            if has_iron.sum() == 0:
                continue

            def get_acquisition_indices(has_resource_array):
                # identify indices where the current index is 1 and the previous index is 0
                return np.where((has_resource_array[:-1] == 0) & (has_resource_array[1:] == 1))[0] + 1

            wood_acquisition_indices = get_acquisition_indices(has_wood)
            stone_acquisition_indices = get_acquisition_indices(has_stone)
            coal_acquisition_indices = get_acquisition_indices(has_coal)
            iron_acquisition_indices = get_acquisition_indices(has_iron)



            for idx in iron_acquisition_indices:
                priority = get_priority_fn(traj_npz, idx)
                if priority is None:
                    past_the_end_of_priority_array = True
                    break
                iron_priorities.append(priority)
                norm_iron_priorities.append(priority / mean_priority)

            if past_the_end_of_priority_array:
                print(f'Past the end of priority array at episode {episode_num}')
                break

            # plt.imshow(ep['image'][-1])
            # plt.axis('off')
            # plt.show()

    # Find stats on iron priorities
    print('---')
    print(f'Number of iron acquisition steps: {len(iron_priorities)}')
    print(f'Mean: {np.mean(iron_priorities)}')
    print(f'Median: {np.median(iron_priorities)}')
    print(f'Min: {np.min(iron_priorities)}')
    print(f'Max: {np.max(iron_priorities)}')
    print(f'Std: {np.std(iron_priorities)}')

    print(' ')

    print(f'Scaled priorities')
    print(f'Mean: {np.mean(norm_iron_priorities)}')
    print(f'Median: {np.median(norm_iron_priorities)}')
    print(f'Min: {np.min(norm_iron_priorities)}')
    print(f'Max: {np.max(norm_iron_priorities)}')
    print(f'Std: {np.std(norm_iron_priorities)}')


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
    else:
        raise ValueError(f'Unknown resource {resource}')

    def block_to_pixel(block):
        block_size = image_size // blocks
        return (block - 1) * block_size + block_size // 2

    def check_pixel_not_black(images, col, row):
        black_pixel = np.array([0, 0, 0], dtype=np.uint8)
        not_black = (images[:, row, col] != black_pixel).any(axis=1).astype(np.uint8)
        return not_black

    pixel_x, pixel_y = block_to_pixel(block_x), block_to_pixel(block_y)

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


def get_avg_episode_priority(priority_npz_loaded, priority_pkl_loaded, run_name, episode_name, episode_length):
    prefix = f'/home/ikauvar/logs/{run_name}/train_episodes'
    start_idx = priority_pkl_loaded['idx_from_ep'][f'{prefix}/{episode_name}:1']
    end_idx = priority_pkl_loaded['idx_from_ep'][f'{prefix}/{episode_name}:{episode_length}']

    return np.mean(priority_npz_loaded['arr'][start_idx:end_idx])

if __name__ == "__main__":
    run()

