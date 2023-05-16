import collections
import json
import pathlib

import numpy as np

import common


def read_stats(indir, outdir, task, method, budget=int(1e6), verbose=False):
  indir = pathlib.Path(indir)
  outdir = pathlib.Path(outdir)
  runs = []
  print(f'Loading {indir.name}...')
  filenames = sorted(list(indir.glob('**/stats.jsonl')))
  for index, filename in enumerate(filenames):
    if not filename.is_file():
      continue
    rewards, lengths, achievements = load_stats(filename, budget)
    if sum(lengths) < budget - 1e4:
      message = f'Skipping incomplete run ({sum(lengths)} < {budget} steps): '
      message += f'{filename.relative_to(indir.parent)}'
      print(f'==> {message}')
      continue
    runs.append(dict(
        task=task,
        method=method,
        seed=str(index),
        xs=np.cumsum(lengths).tolist(),
        reward=rewards,
        length=lengths,
        **achievements,
    ))
  if not runs:
    print('No completed runs.\n')
    return
  print_summary(runs, budget, verbose)
  outdir.mkdir(exist_ok=True, parents=True)
  filename = (outdir / f'{task}-{method}.json')
  filename.write_text(json.dumps(runs))
  print('Wrote', filename)
  print('')


def load_stats(filename, budget):
  steps = 0
  rewards = []
  lengths = []
  achievements = collections.defaultdict(list)
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    steps += episode['length']
    if steps > budget:
      break
    lengths.append(episode['length'])
    for key, value in episode.items():
      if key.startswith('achievement_'):
        achievements[key].append(value)
    unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
    health = -0.9
    rewards.append(unlocks + health)
  return rewards, lengths, achievements


def print_summary(runs, budget, verbose):
  episodes = np.array([len(x['length']) for x in runs])
  rewards = np.array([np.mean(x['reward']) for x in runs])
  lengths = np.array([np.mean(x['length']) for x in runs])
  percents, methods, seeds, tasks = common.compute_success_rates(
      runs, budget, sortby=0)
  scores = np.squeeze(common.compute_scores(percents))
  print(f'Score:        {np.mean(scores):10.2f} ± {np.std(scores):.2f}')
  print([s.round(1) for s in scores])
  print(f'Reward:       {np.mean(rewards):10.2f} ± {np.std(rewards):.2f}')
  print(f'Length:       {np.mean(lengths):10.2f} ± {np.std(lengths):.2f}')
  print(f'Episodes:     {np.mean(episodes):10.2f} ± {np.std(episodes):.2f}')
  print(f'Runs:         {len(episodes):10.0f}')
  if verbose:
    for task, percent in sorted(tasks, np.squeeze(percents).T):
      name = task[len('achievement_'):].replace('_', ' ').title()
      print(f'{name:<20}  {np.mean(percent):6.2f}%')


budget = 1000000

print(f'Analyzing at {budget} steps')

short_budget = f'{budget // 1000}k' if budget < 1e6 else f'{int(budget // 1e6)}M'

# read_stats(
#     '/home/cd/remote-download/crafter_collect_20230314/cr-per-key',
#     '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
#     'crafter_reward',
#     f'cr-per-seq-{short_budget}',
#     budget=int(budget))

read_stats(
    '/home/cd/remote-download/crafter_collect_20230321/cr-full',
    '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
    'crafter_reward',
    f'cr-full-{short_budget}',
    budget=int(budget))


read_stats(
    '/home/cd/remote-download/crafter_collect_20230330_dv3td/td',
    '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
    'crafter_reward',
    f'td-full-{short_budget}',
    budget=int(budget))

read_stats(
    '/home/cd/remote-download/crafter_countbased_20230424/count-based',
    '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
    'crafter_reward',
    f'count-based-{short_budget}',
    budget=int(budget))

read_stats(
    '/home/cd/remote-download/crafter_adversarial_20230419/adversarial',
    '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
    'crafter_reward',
    f'adversarial-{short_budget}',
    budget=int(budget))

# read_stats(
#     '/home/cd/remote-download/crafter_collect_20230312/cr',
#     '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
#     'crafter_reward',
#     'cr-300k',
#     budget=int(300000))

read_stats(
    '/home/cd/remote-download/crafter_collect_20230313/dv3',
    '/home/cd/remote-download/crafter_collect_20230308/crafter/scores',
    'crafter_reward',
    f'dv3-baseline-{short_budget}',
    budget=int(budget))

read_stats(
    '/home/cd/remote-download/crafter_priority_info_20230508/node1',
    '/home/cd/remote-download/crafter_priority_info_20230508/node1',
    'prio_run_crafter_reward',
    f'dv3-baseline_prio_-{short_budget}',
    budget=int(budget))
#

#
# read_stats(
#     'logdir/crafter_reward-ppo',
#     'scores', 'crafter_reward', 'ppo')
#
# read_stats(
#     'logdir/crafter_reward-rainbow',
#     'scores', 'crafter_reward', 'rainbow')
#
# read_stats(
#     'logdir/crafter_noreward-unsup_plan2explore',
#     'scores', 'crafter_noreward', 'unsup_plan2explore')
#
# read_stats(
#     'logdir/crafter_noreward-unsup_rnd',
#     'scores', 'crafter_noreward', 'unsup_rnd')
#
# read_stats(
#     'logdir/crafter_noreward-random',
#     'scores', 'crafter_noreward', 'random')
