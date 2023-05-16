from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def run():
    sliding_window = 20000
    max_episodes = 10000
    budget = 1000000
    threshold_level = 0.01

    groups = {
        'dv3': [
            f'/home/cd/remote-download/crafter_collect_20230313/dv3/a100-01/crafter-dv3-20230312-071153/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230313/dv3/a100-02/crafter-dv3-20230312-071202/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230313/dv3/a100-03/crafter-dv3-20230312-071207/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230313/dv3/a100-04/crafter-dv3-20230312-071213/stats.jsonl'
        ],
        'cr-per-key': [
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-01/crafter-dv3-20230309-234939/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-02/crafter-dv3-20230310-001122/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-03/crafter-dv3-20230310-001124/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-04/crafter-dv3-20230310-001127/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-05/crafter-dv3-20230313-043140/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-06/crafter-dv3-20230313-043143/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-07/crafter-dv3-20230313-043144/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-08/crafter-dv3-20230313-043145/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-09/crafter-dv3-20230313-043146/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-10/crafter-dv3-20230313-043148/stats.jsonl',
        ],
        'cr-full': [
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123329/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123335/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123340/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123345/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123354/stats.jsonl',
        ],
    }

    """
      all_ids['dv2_crafter_U'] = {
        'DRF-59': 't4-tf-16',
        'DRF-58': 't4-tf-15',
        'DRF-57': 't4-tf-14',
        'DRF-60': 't4-tf-1c',
        'DRF-82': 't4-tf-11b',
        'DRF-81': 't4-tf-10b',
        'DRF-80': 't4-tf-9b',
        'DRF-79': 't4-tf-8b',
    
        'DRF-78': 't4-tf-7b',
        'DRF-77': 't4-tf-6b',
      }
      
        all_ids['dv2_crafter_D'] = { # New D
        'DRF-300': 't4-tf-3b',
        'DRF-299': 't4-tf-2b',
        'DRF-298': 't4-tf-1b',
        'DRF-297': 't4-tf-16',
        'DRF-294': 't4-tf-13',
        'DRF-293': 't4-tf-12',
        'DRF-292': 't4-tf-11',
        'DRF-291': 't4-tf-16c',
    
        'DRF-466': 't4-tf-16b',
        'DRF-465': 't4-tf-15b',
    
      }

      
        all_ids['dv2_crafter_g08_a07_losses'] = {
        'DRF-423': 't4-tf-16b',
        'DRF-422': 't4-tf-15b',
        'DRF-421': 't4-tf-14b',
        'DRF-420': 't4-tf-13b',
        'DRF-419': 't4-tf-12b',
        'DRF-418': 't4-tf-11b',
        'DRF-417': 't4-tf-10b',
        'DRF-416': 't4-tf-9b',
    
        'DRF-464': 't4-tf-14b',
        'DRF-463': 't4-tf-13b',
      }
  
    """


    groups = {
        'dv2_crafter_U': [
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-59_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-58_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-57_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-60_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-82_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-81_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-80_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-79_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-78_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-77_train_env0.jsonl',
        ],
        # 'dv2_crafter_D': [
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-300_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-299_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-298_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-297_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-294_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-293_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-292_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-291_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-466_train_env0.jsonl',
        #     '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-465_train_env0.jsonl',
        # ],
        'dv2_crafter_g08_a07_losses': [
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-423_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-422_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-421_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-420_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-419_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-418_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-417_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-416_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-464_train_env0.jsonl',
            '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-463_train_env0.jsonl',
        ],
        'dv3': [
            f'/home/cd/remote-download/crafter_collect_20230313/dv3/a100-01/crafter-dv3-20230312-071153/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230313/dv3/a100-02/crafter-dv3-20230312-071202/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230313/dv3/a100-03/crafter-dv3-20230312-071207/stats.jsonl',
            '/home/cd/remote-download/crafter_collect_20230313/dv3/a100-04/crafter-dv3-20230312-071213/stats.jsonl'
        ],
        # 'dv3-cr-per-seq': [
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-01/crafter-dv3-20230309-234939/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-02/crafter-dv3-20230310-001122/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-03/crafter-dv3-20230310-001124/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-04/crafter-dv3-20230310-001127/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-05/crafter-dv3-20230313-043140/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-06/crafter-dv3-20230313-043143/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-07/crafter-dv3-20230313-043144/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-08/crafter-dv3-20230313-043145/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-09/crafter-dv3-20230313-043146/stats.jsonl',
        #     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key/a100-10/crafter-dv3-20230313-043148/stats.jsonl',
        # ],
        'cr-full': [
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123329/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123335/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123340/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123345/stats.jsonl',
            '/home/cd/remote-download/crafter_priority_info_20230508/node1/crafter-dv3-20230503-123354/stats.jsonl',
        ],

    }

  #   """
  #   all_ids['p2e_crafter_U'] = {
  #   'DRF-169': 't4-tf-8b',
  #   'DRF-151': 't4-tf-7b',
  #   'DRF-150': 't4-tf-6b',
  #   'DRF-149': 't4-tf-5b',
  #   'DRF-147': 't4-tf-4b',
  #   'DRF-146': 't4-tf-3b',
  #   'DRF-145': 't4-tf-2b',
  #   'DRF-144': 't4-tf-1b',
  #
  #   'DRF-470': 't4-tf-16b',
  #   'DRF-469': 't4-tf-15b',
  # }
  # all_ids['p2e_crafter_D'] = { # New D
  #   'DRF-289': 't4-tf-14c',
  #   'DRF-288': 't4-tf-13c',
  #   'DRF-287': 't4-tf-12c',
  #   'DRF-286': 't4-tf-11c',
  #   'DRF-285': 't4-tf-10c',
  #   'DRF-284': 't4-tf-9c',
  #   'DRF-283': 't4-tf-8c',
  #   'DRF-282': 't4-tf-7c',
  #
  #   'DRF-468': 't4-tf-14b',
  #   'DRF-467': 't4-tf-13b',
  # }
  #
  # all_ids['p2e_crafter_g08_a07_losses'] = {
  #   'DRF-431': 't4-tf-16',
  #   'DRF-430': 't4-tf-15',
  #   'DRF-429': 't4-tf-14',
  #   'DRF-428': 't4-tf-13',
  #   'DRF-427': 't4-tf-12',
  #   'DRF-426': 't4-tf-11',
  #   'DRF-425': 't4-tf-10',
  #   'DRF-424': 't4-tf-9',
  #
  #   'DRF-486': 't4-tf-2c',
  #   'DRF-485': 't4-tf-1c',
  #
  # }
  #
  #   """
  #   groups = {
  #       'p2e_crafter_U': [
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-169_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-151_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-150_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-149_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-147_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-146_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-145_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-144_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-470_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-469_train_env0.jsonl',
  #       ],
  #       'p2e_crafter_D': [
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-289_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-288_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-287_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-286_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-285_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-284_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-283_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-282_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-468_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-467_train_env0.jsonl',
  #       ],
  #       'p2e_crafter_g08_a07_losses': [
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-431_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-430_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-429_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-428_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-427_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-426_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-425_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-424_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-486_train_env0.jsonl',
  #           '/home/cd/remote-download/icml2023crafter/crafter_stats/stats/stats_DRF-485_train_env0.jsonl',
  #       ],}

    unlock_results = {}
    threshold = {}
    threshold_avg = {}
    threshold_delta = {}
    threshold_delta_avg = {}
    threshold_delta_std = {}

    for group in groups:
        unlock_results[group] = []
        threshold[group] = []
        threshold_avg[group] = {}
        threshold_delta[group] = {}
        threshold_delta_avg[group] = {}
        threshold_delta_std[group] = {}

        for filename in groups[group]:
            avg_unlocks_over_time = moving_average_unlocks(max_episodes, budget, filename, sliding_window)
            unlock_results[group].append(avg_unlocks_over_time)
            threshold[group].append(threshold_calc(avg_unlocks_over_time, threshold_level))

        for key in threshold[group][0]:
            censored = [threshold[group][i][key] for i in range(len(threshold[group])) if threshold[group][i][key] > 0]
            threshold_avg[group][key] = np.mean(censored)
            # if 0 < len(censored) < len(threshold[group]):
            #     print(f'WARNING: {key} has {len(threshold[group]) - len(censored)} missing values', flush=True)

        for d in [('achievement_collect_wood', 'achievement_place_table'),
                  ('achievement_place_table', 'achievement_make_wood_pickaxe'),
                  ('achievement_make_wood_pickaxe','achievement_collect_stone'),]:
            threshold_delta[group][d] = [threshold[group][i][d[1]] - threshold[group][i][d[0]]
                                         for i in range(len(threshold[group]))
                                         if threshold[group][i][d[1]] > 0 and threshold[group][i][d[0]] > 0]
            threshold_delta_avg[group][d] = np.mean(threshold_delta[group][d])
            threshold_delta_std[group][d] = np.std(threshold_delta[group][d])

    print(threshold_avg)

    # x = steps_to_threshold_all_achievements(group, threshold_avg, threshold_level)
    #
    # filtered, x = step_delta_simple(groups, sliding_window, threshold_avg, threshold_level, x)
    #
    # step_delta_to_chained_achievement_thresholds(filtered, groups, sliding_window, threshold_avg, threshold_delta_avg,
    #                                              threshold_delta_std, threshold_level, x)
    #
    min_steps = 0
    max_steps = 1000000
    step_interval = 500

    achievements_list = ['achievement_collect_wood',
                         'achievement_place_table',
                         'achievement_make_wood_pickaxe',
                         'achievement_collect_stone',
                         'achievement_make_stone_pickaxe',
                         'achievement_collect_iron']
    #
    # proportion_of_ep_split_by_achievement(achievements_list, max_steps, min_steps, step_interval, unlock_results)
    proportion_of_eps_split_by_group(achievements_list, max_steps, min_steps, step_interval, unlock_results)

    # Print out a table of the following achievement chain:
    chain = ['achievement_collect_wood',
             'achievement_place_table',
             'achievement_make_wood_pickaxe',
             'achievement_collect_stone',
             'achievement_make_stone_pickaxe',
             'achievement_collect_iron']

    gs = ['dv2_crafter_U', 'dv2_crafter_g08_a07_losses']

    print(' Steps to 1% achievement threshold')
    print('| Achievement | dv2_crafter_U | dv2_crafter_g08_a07_losses |')
    print('| --- | --- | --- |')
    for achievement in chain:
        g_vals_str = ''
        for g in gs:
            g_vals_str += f' {threshold_avg[g][achievement]:.0f} |'

        print(f'| {achievement} |{g_vals_str}')

    gs = ['dv2_crafter_U', 'dv2_crafter_g08_a07_losses', 'dv3', 'cr-full']

    print(' Steps to 1% achievement threshold')
    print('| Achievement | dv2_crafter_U | dv2_crafter_g08_a07_losses | dv3 | cr-full |')
    print('| --- | --- | --- |')
    for achievement in chain:
        g_vals_str = ''
        for g in gs:
            g_vals_str += f' {threshold_avg[g][achievement]:.0f} |'

        print(f'| {achievement} |{g_vals_str}')

    return


def steps_to_threshold_all_achievements(group, threshold_avg, threshold_level):
    ## ALL ACHIEVEMENTS ##
    # make a multibar plot of threshold_avg, one bar for each achievement, and one color for each group
    fig, ax = plt.subplots()
    width = 0.25
    multiplier = 0
    for group in threshold_avg:
        x = np.arange(len(list(threshold_avg[group].keys())))
        offset = width * multiplier
        ax.bar(x + offset, threshold_avg[group].values(), width, label=group)
        multiplier += 1
    # make xtick labels vertical
    ax.set_xticks(x + width / 2, [k.split('_', 1)[1] for k in threshold_avg[group].keys()])
    plt.xticks(rotation=90)
    plt.legend()
    plt.title(f'Steps to {threshold_level} achievement rate (avg over 50k steps)')
    plt.tight_layout()
    plt.show()
    return x


def step_delta_simple(groups, sliding_window, threshold_avg, threshold_level, x):
    ## DELTA ##
    fig, ax = plt.subplots()
    width = 0.1
    multiplier = 0
    for group in threshold_avg:
        # calculate deltas
        # 'place_table - collect_wood'
        # 'make_wood_pickaxe - place_table'
        # 'collect_stone - make_wood_pickaxe'
        filtered = {
            'place_table -\n collect_wood': threshold_avg[group]['achievement_place_table'] - threshold_avg[group][
                'achievement_collect_wood'],
            'make_wood_pickaxe -\n place_table': threshold_avg[group]['achievement_make_wood_pickaxe'] -
                                                 threshold_avg[group]['achievement_place_table'],
            'collect_stone -\n make_wood_pickaxe': threshold_avg[group]['achievement_collect_stone'] -
                                                   threshold_avg[group]['achievement_make_wood_pickaxe']}
        x = np.arange(len(list(filtered.keys())))
        offset = width * multiplier
        ax.bar(x + offset, filtered.values(), width, label=group)
        multiplier += 1
    # make xtick labels vertical
    ax.set_xticks(x + (len(groups) * width) / 2, filtered.keys())
    # plt.xticks(rotation=90)
    plt.legend()
    plt.title(f'Steps to {threshold_level} achievement rate (avg over {sliding_window} steps)')
    plt.tight_layout()
    plt.show()
    return filtered, x


def step_delta_to_chained_achievement_thresholds(filtered, groups, sliding_window, threshold_avg, threshold_delta_avg,
                                                 threshold_delta_std, threshold_level, x):
    ## DELTA BY RUN ##
    fig, ax = plt.subplots()
    width = 0.1
    multiplier = 0
    for group in threshold_avg:
        # calculate deltas
        # 'place_table - collect_wood'
        # 'make_wood_pickaxe - place_table'
        # 'collect_stone - make_wood_pickaxe'
        filtered = {'place_table -\n collect_wood':
                        threshold_delta_avg[group][('achievement_collect_wood', 'achievement_place_table')],
                    'make_wood_pickaxe -\n place_table':
                        threshold_delta_avg[group][('achievement_place_table', 'achievement_make_wood_pickaxe')],
                    'collect_stone -\n make_wood_pickaxe':
                        threshold_delta_avg[group][('achievement_make_wood_pickaxe', 'achievement_collect_stone')]}
        x = np.arange(len(list(filtered.keys())))
        offset = width * multiplier
        ax.bar(x + offset, filtered.values(), width, label=group)
        # add error bars
        ax.errorbar(x + offset, filtered.values(), yerr=list(threshold_delta_std[group].values()), fmt='none',
                    ecolor='black', capsize=5)
        multiplier += 1
    # make xtick labels vertical
    ax.set_xticks(x + (len(groups) * width) / 2, filtered.keys())
    # plt.xticks(rotation=90)
    plt.legend()
    plt.title(f'Steps to {threshold_level} achievement rate (avg over {sliding_window} steps)')
    plt.tight_layout()
    plt.show()
    # plot achievement rate over time for each achievement
    # plt.figure(figsize=(10, 6))
    # for g in groups:
    #     for key in unlock_results:
    #         if key.startswith('achievement_collect_wood') or key.startswith('achievement_place_table'):
    #             plt.plot(unlock_results['cumulative_steps'], unlock_results[key], label=key)
    #             plt.legend()
    #             plt.show()


def proportion_of_ep_split_by_achievement(achievements_list, max_steps, min_steps, step_interval, unlock_results):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    axs = axs.flatten()
    for ax, achievement in zip(axs, achievements_list):
        # Define a common set of cumulative steps
        common_cumulative_steps = np.arange(min_steps, max_steps + 1, step_interval)

        # Initialize an empty DataFrame for storing resampled achievement values and corresponding common steps
        resampled_data = pd.DataFrame(columns=['Cumulative Steps', 'Achievement Value', 'Group'])

        # Loop through each group in unlock_results
        for group in unlock_results.keys():
            # Loop through each run in the current group
            for i, r in enumerate(unlock_results[group]):
                cumulative_steps = r['cumulative_steps']
                achievements = r[achievement]

                # Interpolate the achievement values to the common set of cumulative steps
                interpolated_achievements = np.interp(common_cumulative_steps, cumulative_steps, achievements)

                # Add the interpolated achievement values and corresponding common steps to the DataFrame
                run_data = pd.DataFrame({'Cumulative Steps': common_cumulative_steps,
                                         'Achievement Value': interpolated_achievements,
                                         'Group': group})
                resampled_data = resampled_data.append(run_data, ignore_index=True)

        # Create a line plot with a shaded area to illustrate uncertainty for each group
        sns.lineplot(x='Cumulative Steps', y='Achievement Value', hue='Group', errorbar='sd', data=resampled_data,
                     ax=ax)

        # Add labels to the subplot
        ax.set_xlabel('Cumulative Steps')
        ax.set_ylabel('Proportion of episodes')
        ax.set_title(f'{achievement} vs. Step')
        ax.grid()
        ax.legend(title='Groups')
        # ax.set_yscale("log")
    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()


def proportion_of_eps_split_by_group(achievements_list, max_steps, min_steps, step_interval, unlock_results,
                                     threshold=0.01):
    # Create a 2x2 grid of subplots
    #fig, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    fig, axs = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    axs = axs.flatten()

    step_cross_threshold = {a: {} for a in achievements_list}

    for ax, group in zip(axs, unlock_results.keys()):

        # Define a common set of cumulative steps
        common_cumulative_steps = np.arange(min_steps, max_steps + 1, step_interval)

        # Initialize an empty DataFrame for storing resampled achievement values and corresponding common steps
        resampled_data = pd.DataFrame(columns=['Cumulative Steps', 'Achievement Value', 'Group'])

        # Loop through each group in unlock_results
        for achievement in achievements_list:
            for i, r in enumerate(unlock_results[group]):
                cumulative_steps = r['cumulative_steps']
                achievements = r[achievement]

                # Interpolate the achievement values to the common set of cumulative steps
                interpolated_achievements = np.interp(common_cumulative_steps, cumulative_steps, achievements)

                # Add the interpolated achievement values and corresponding common steps to the DataFrame
                run_data = pd.DataFrame({'Cumulative Steps': common_cumulative_steps,
                                         'Achievement Value': interpolated_achievements,
                                         'Achievement': achievement})
                resampled_data = resampled_data.append(run_data, ignore_index=True)
                # no legend

            step_cross_threshold[achievement][group] = np.NaN
            # find average achievement value at each step
            for step in common_cumulative_steps:
                # find the mean of run_data where cumulative_steps == step and achievement == achievement
                v = resampled_data[(resampled_data['Cumulative Steps'] == step) &
                                   (resampled_data['Achievement'] == achievement)]['Achievement Value'].mean()
                if v >= threshold:
                    step_cross_threshold[achievement][group] = step
                    break

        # Create a line plot with a shaded area to illustrate uncertainty for each group
        sns.lineplot(x='Cumulative Steps', y='Achievement Value', hue='Achievement', errorbar='sd', data=resampled_data,
                     ax=ax)

        # Add labels to the subplot
        ax.set_xlabel('Cumulative Steps')
        ax.set_ylabel('Proportion of episodes')
        ax.set_title(f'{group}')
        ax.grid()
        # turn off legend
        #ax.legend().set_visible(False)

        # ax.set_yscale("log")
    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

    print(step_cross_threshold)
    # print out the dictionary as a markdown table in the same order as achievement_list and group_list
    groups = list(unlock_results.keys())
    print(f'| Achievement | {groups[0]} | {groups[1]} | {groups[2]} | {groups[3]} |')
    print('| ----------- | ------- | ------- | ------- | ------- |')
    for achievement in achievements_list:
        print('|', achievement, '| ', end='')
        for group in unlock_results.keys():
            print(step_cross_threshold[achievement][group], '| ', end='')
        print()


def moving_average_unlocks(MAX_EPISODES, budget, filename, sliding_window):
    filename = Path(filename)
    data = dict()
    data['episode_lengths'] = np.zeros(MAX_EPISODES)
    i = 0
    cumulative_steps = 0
    for line in filename.read_text().split('\n'):
        if not line.strip():
            continue
        episode = json.loads(line)
        cumulative_steps += episode['length']
        if cumulative_steps > budget:
            break

        data['episode_lengths'][i] = episode['length']
        for key, value in episode.items():
            if key.startswith('achievement_'):
                if key not in data:
                    data[key] = np.zeros(MAX_EPISODES)
                data[key][i] = value >= 1

        i += 1
    sum_of_episode_lengths = 0
    indices = []
    start_episode = 0
    results = {key: np.zeros(i) for key in data}
    results['cumulative_steps'] = np.zeros(i)
    for end_episode in range(i):

        sum_of_episode_lengths += data['episode_lengths'][end_episode]

        if sum_of_episode_lengths < sliding_window:
            continue

        while (sum_of_episode_lengths - data['episode_lengths'][start_episode]) >= sliding_window:
            sum_of_episode_lengths -= data['episode_lengths'][start_episode]
            start_episode += 1

        for key in data:
            results[key][end_episode] = np.mean(data[key][start_episode:end_episode + 1])

        results['cumulative_steps'][end_episode] = np.sum(data['episode_lengths'][0:end_episode + 1])

        # debugging help
        indices.append((start_episode, end_episode, np.sum(data['episode_lengths'][start_episode:end_episode + 1])))
    return results




def threshold_calc(avg_unlocks_over_time, threshold):
    results = {}
    for k, v in avg_unlocks_over_time.items():
        if k.startswith('achievement_'):
            step_to_threshold = np.argmax(avg_unlocks_over_time[k] > threshold)
            results[k] = avg_unlocks_over_time['cumulative_steps'][step_to_threshold]
    return results

if __name__ == '__main__':
    run()
