import json
import pathlib
import warnings

import numpy as np


def load_runs(filenames, budget=1e6, verbose=True):
  verbose and print('')
  runs = []
  for filename in filenames:
    loaded = json.loads(pathlib.Path(filename).read_text())
    for run in [loaded] if isinstance(loaded, dict) else loaded:
      message = f'Loading {run["method"]} seed {run["seed"]}'
      verbose and print(message, flush=True)
      if run['xs'][-1] < budget - 1e4:
        verbose and print(f'  Contains only {run["xs"][-1]} steps!')
      runs.append(run)
  verbose and print('')
  return runs


def compute_success_rates(runs, budget=1e6, sortby=None):
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))
  tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))

  if sortby == 'depth':
    """
    [
    'achievement_collect_coal', 
    'achievement_collect_diamond', 
    'achievement_collect_drink', 
    'achievement_collect_iron', 
    'achievement_collect_sapling', 
    'achievement_collect_stone', 
    'achievement_collect_wood', 
    'achievement_defeat_skeleton', 
    'achievement_defeat_zombie', 
    'achievement_eat_cow', 
    'achievement_eat_plant', 
    'achievement_make_iron_pickaxe', 
    'achievement_make_iron_sword', 
    'achievement_make_stone_pickaxe', 
    'achievement_make_stone_sword', 
    'achievement_make_wood_pickaxe', 
    'achievement_make_wood_sword', 
    'achievement_place_furnace', 
    'achievement_place_plant', 
    'achievement_place_stone', 
    'achievement_place_table', 
    'achievement_wake_up']
    
    to
    
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
    """

    tasks = [
      'achievement_wake_up',
        'achievement_collect_wood',
        'achievement_collect_drink',
        'achievement_defeat_zombie',
        'achievement_eat_cow',
        'achievement_defeat_skeleton',
        'achievement_collect_sapling',
        'achievement_place_table',
        'achievement_place_plant',
        'achievement_make_wood_pickaxe',
        'achievement_make_wood_sword',
        'achievement_eat_plant',
        'achievement_collect_stone',
        'achievement_place_stone',
        'achievement_collect_coal',
        'achievement_place_furnace',
        'achievement_make_stone_sword',
        'achievement_make_stone_pickaxe',
        'achievement_collect_iron',
        'achievement_make_iron_pickaxe',
        'achievement_make_iron_sword',
        'achievement_collect_diamond',
    ]

  percents = np.empty((len(methods), len(seeds), len(tasks)))
  percents[:] = np.nan
  counts = np.empty((len(methods), len(seeds), len(tasks)))
  counts[:] = np.nan
  total_episodes = np.empty((len(methods), len(seeds)))
  total_episodes[:] = np.nan
  for run in runs:
    episodes = (np.array(run['xs']) <= budget).sum()
    i = methods.index(run['method'])
    j = seeds.index(run['seed'])
    total_episodes[i, j] = episodes
    for key, values in run.items():
      if key in tasks:
        k = tasks.index(key)
        percent = 100 * (np.array(values[:episodes]) >= 1).mean()
        percents[i, j, k] = percent
        counts[i, j, k] = (np.array(values[:episodes]) >= 1).sum()

  # if isinstance(sortby, (str, int)):
  #   if isinstance(sortby, str):
  #     sortby = methods.index(sortby)
  #   order = np.argsort(-np.nanmean(percents[sortby], 0), -1)
  #   percents = percents[:, :, order]
  #   tasks = np.array(tasks)[order].tolist()
  return percents, methods, seeds, tasks


def compute_scores(percents):
  # Geometric mean with an offset of 1%.
  assert (0 <= percents).all() and (percents <= 100).all()
  if (percents <= 1.0).all():
    print('Warning: The input may not be in the right range.')
  with warnings.catch_warnings():  # Empty seeds become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
  return scores


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  xs, ys = np.array(xs), np.array(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  with warnings.catch_warnings():  # Empty buckets become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    for start, stop in zip(borders[:-1], borders[1:]):
      left = (xs <= start).sum()
      right = (xs <= stop).sum()
      if left < right:
        value = reducer(ys[left:right])
      elif binned:
        value = {'nan': np.nan, 'last': binned[-1]}[fill]
      else:
        value = np.nan
      binned.append(value)
  return borders[1:], np.array(binned)
