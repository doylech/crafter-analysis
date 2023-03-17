"""
Load episodes from replay buffer.
Quantify features of the episodes (such as how many magenta pixels).
See download_ckpt.sh to get necessary files from remote server for running this locally.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import argparse

from os.path import expanduser

def run():
    traj_dir = '/home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-423/train_episodes'

    neps = len(os.listdir(traj_dir))
    for i in range(neps):
        # Load episodes
        episode_num = neps - i - 1
        episode_num = 6850
        ep, traj_name, traj_timestamp = load_ep(traj_dir, episode_num)

        imgs = ep['image']
        has_wood = has_resource('wood', imgs)
        has_stone = has_resource('stone', imgs)
        has_coal = has_resource('coal', imgs)
        has_iron = has_resource('iron', imgs)

        def get_acquisition_indices(has_resource_array):
            # identify indices where the current index is 1 and the previous index is 0
            return np.where((has_resource_array[:-1] == 0) & (has_resource_array[1:] == 1))[0] + 1


        # if has_iron.sum() > 0:
        #     print(neps - i - 1)

        plt.imshow(ep['image'][-1])
        plt.axis('off')
        plt.show()

        break
        if i > 500:
            break

    # print(f'Loaded {neps} episodes')
    priorities_npz = '/home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-423/train_episodes_priority.npz_110000'
    priorities_pkl = '/home/cd/remote-download/icml2023crafter/dv2_crafter_g08_a07_losses/DRF-423/train_episodes_priority.npz_110000.pkl'

    priorities_npz_loaded = np.load(priorities_npz)
    priorities_pkl_loaded = pickle.load(open(priorities_pkl, 'rb'))

    priorities_npz_loaded['arr']

    print('loaded')

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


if __name__ == "__main__":
    run()

        #
        #
        # colors = {'yellow':  {'color_max': [255, 255, 20], 'color_min': [160, 160, 0]},
        #           'magenta': {'color_max': [255, 20, 255], 'color_min': [75, 0, 75]}}
        # yellow = amount_of_color(imgs, colors['yellow']['color_min'], colors['yellow']['color_max']).numpy()
        # magenta = amount_of_color(imgs, colors['magenta']['color_min'], colors['magenta']['color_max']).numpy()
        #
        # npx_magenta.append(magenta)
        # npx_yellow.append(yellow)
