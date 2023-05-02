# IK CODE / IN PROGRESS

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import common
import scipy
def plot_spectrum_ik(all_percent, all_tasks, outpath, legend, colors,
                     order_by_difficulty=True, just_high_difficulty=False,
                     show_error_bars=False):
  df = pd.DataFrame()
  for method in all_percent.keys():
    percents = all_percent[method]
    for seed in np.arange(percents.shape[1]):
      for taskind in np.arange(percents.shape[2]):
        task = str(all_tasks[method][taskind][len('achievement_'):].replace('_', ' ').title())
        percent = percents[0, seed, taskind]
        df = pd.concat([df, pd.DataFrame.from_dict(dict(seed=[seed], task=[task],
                                                        method=[legend[method]], percent=[percent]))])

  df = df.reset_index(drop=True)
  try:
    collect_iron_baseline = df[(df.isin({'task': ['Collect Iron']}).task) & (df.method == 'Baseline')].percent.to_numpy()
    collect_iron_cr = df[(df.isin({'task': ['Collect Iron']}).task) & (df.method == 'Curious Replay')].percent.to_numpy()
    print(f'Collect Iron. Baseline: {collect_iron_baseline.mean():0.3f}, CR: {collect_iron_cr.mean():0.3f}')
    print(scipy.stats.ttest_ind(collect_iron_baseline, collect_iron_cr, alternative='less'))
    print(scipy.stats.ttest_ind(df[(df.isin({'task': ['Collect Iron']}).task) & (df.method == 'Baseline')].percent,
                          df[(df.isin({'task': ['Collect Iron']}).task) & (df.method == 'Curious Replay')].percent,
                          ))
    print(scipy.stats.ttest_ind(df[(df.isin({'task': ['Collect Iron']}).task) & (df.method == 'Baseline')].percent,
                          df[(df.isin({'task': ['Collect Iron']}).task) & (df.method == 'dg08')].percent,
                          ))
  except:
    pass
  if order_by_difficulty:
    tasks = ['Wake Up', 'Collect Wood', 'Collect Drink', 'Defeat Zombie', 'Eat Cow', 'Defeat Skeleton',
             'Collect Sapling',
             'Place Table', 'Place Plant',
             'Make Wood Pickaxe', 'Make Wood Sword', 'Eat Plant',
             'Collect Stone',
              'Place Stone', 'Collect Coal',
              'Place Furnace', 'Make Stone Sword', 'Make Stone Pickaxe',
             'Collect Iron',
             'Make Iron Pickaxe', 'Make Iron Sword',
             'Collect Diamond',
             ]
    if just_high_difficulty:
      tasks = ['Make Wood Pickaxe', 'Make Wood Sword', 'Eat Plant',
             'Collect Stone',
             'Place Stone', 'Collect Coal',
             'Place Furnace', 'Make Stone Sword', 'Make Stone Pickaxe',
             'Collect Iron',
             'Make Iron Pickaxe', 'Make Iron Sword',
             'Collect Diamond',
             ]
    # tasks = ['Wake Up', 'Collect Wood', 'Collect Drink',
    #          'Collect Sapling',
    #          'Place Table', 'Place Plant',
    #          'Make Wood Pickaxe', 'Make Wood Sword',  'Defeat Zombie', 'Eat Cow', 'Defeat Skeleton',
    #          'Collect Stone', 'Collect Coal',
    #          'Place Furnace', 'Make Stone Pickaxe', 'Make Stone Sword', 'Eat Plant',
    #          'Collect Iron',
    #          'Make Iron Pickaxe', 'Make Iron Sword',
    #          'Collect Diamond',
    #          ]
  else:
    tasks = sorted(df.task.unique())
  if just_high_difficulty:
    fig, ax = plt.subplots(figsize=(5, 3))
  else:
    fig, ax = plt.subplots(figsize=(7, 3))
  if show_error_bars:
    ci = 68
  else:
    ci = None
  g = sns.barplot(ax=ax, data=df, x='task', y='percent', hue='method', order=tasks,
                  # ci='sd',
                  ci=ci, n_boot=100,
                  palette=colors, saturation=1, errcolor = 'black')
  g.legend().remove()
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(
    axis='x', which='both', width=14, length=0.8, direction='inout')
  plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
  plt.xlabel('')
  ax.set_ylabel('Success Rate (%)')
  ax.set_yscale('log')
  ax.set_ylim(0.01, 100)
  ax.set_yticks([0.01, 0.1, 1, 10, 100])
  ax.set_yticklabels('0.01 0.1 1 10 100'.split())
  plt.minorticks_off()
  # plt.ylim([0, 100])
  plt.ylabel('Success Rate (%)')
  # ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')
  plt.legend(loc='upper center', ncol=10, frameon=False, borderpad=0, borderaxespad=0,
             bbox_to_anchor=[.5, 1.15])
  fig.tight_layout()
  plt.savefig(str(outpath) + f'{just_high_difficulty}.pdf')
  plt.savefig(str(outpath)+f'{just_high_difficulty}.png')
  plt.show()



inpaths = [
    # 'scores/crafter_reward-cr-per-key-1M.json',
    'scores/crafter_reward-cr-full-1M.json',
    'scores/crafter_reward-td-full-1M.json',
    'scores/crafter_reward-dv3-baseline-1M.json',
    #'scores/crafter_reward-ppo.json',
    #'scores/crafter_reward-rainbow.json',
]
legend = {
    # 'cr-per-key-1M': 'cr-per-key',
    'cr-full-1M': 'Curious Replay',
    'td-full-1M': 'TD',
    'dv3-baseline-1M': 'DreamerV3 (our 4 runs)',
    #'ppo': 'PPO',
    #'rainbow': 'Rainbow',
}
colors = ['#377eb8', '#5fc35d', '#984ea3']

budget = 1000000

runs = common.load_runs(inpaths, budget)
percents, methods, seeds, tasks = common.compute_success_rates(
    runs, budget, sortby=False and legend and list(legend.keys())[0])

plot_spectrum_ik(percents, tasks, 'plots/spectrum-reward.pdf', legend, colors)