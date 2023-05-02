import numpy as np
import analysis.common as common
import pandas as pd


def print_matrix(inpaths, budget=1e6, legend=None, sort=False):
    runs = common.load_runs(inpaths, budget)
    percents, methods, seeds, tasks = common.compute_success_rates(
        runs, budget, sortby=sort and legend and list(legend.keys())[0])

    names = [x[len('achievement_'):].replace('_', ' ').title() for x in tasks]

    df = pd.DataFrame(index=methods, columns=names)

    for index, (method, label) in enumerate(legend.items()):
        df.iloc[index, :] = np.nanmean(percents[methods.index(method)], 0)

    print(df.T.astype(float).round(decimals=3))

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

print_matrix(inpaths, legend=legend)