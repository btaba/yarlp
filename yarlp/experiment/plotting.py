
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_data(data, value, time, run, condition, title='', ax=None):
    """
    Plot time series data using sns.tsplot

    Params
    ----------
    data (pd.DataFrame):
    value (str): value column
    time (str): time column
    condition (str): sns.tsplot condition
    title (str):
    ax (matplotlib axis):
    """
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    plot = sns.tsplot(
        data=data, time=time, value=value, unit=run, condition=condition,
        ax=ax)
    plt.title(title)
    return plot


def normalize_timesteps(data, x, by, steps):
    """
    Normalize timesteps to fit in sns.tsplots
    """
    data = data.copy()
    for b in data[by].unique():
        data.loc[data[by] == b, x] = \
            np.arange(0, data[data[by] == b].shape[0] * steps, steps)

    return data


def normalize_to_seconds(data, time, by, experiment_name):
    """
    Round seconds and group reward by second for each experiment
    """
    data = data.copy()
    bys = data[by].unique()

    return_data = pd.DataFrame()
    for b in bys:
        d = data.loc[data[by] == b]
        d.loc[:, time] = d[time].apply(lambda x: round(x, 0))
        exp_name = d[experiment_name][0]
        d = d.groupby(time).mean().reset_index()
        d.loc[:, by] = b
        d.loc[:, experiment_name] = exp_name
        d.index = d[time]
        d = d.reindex(index=range(1, int(d[time].max())), method='ffill')
        d[time] = d.index
        return_data = return_data.append(d)

    return return_data


def make_plots(data, env):
    """
    Make plots by second, timestep, and episode
    """

    figure, axes = plt.subplots(ncols=3, nrows=1, figsize=(3 * 6, 6))

    # plot episodes
    plot1 = plot_data(
        data, 'avg_total_reward', 'episode', 'run_name', 'name', env, axes[0])

    # plot timesteps
    timestep_data = normalize_timesteps(
        data, 'timesteps_so_far', 'run_name', 1000)
    plot2 = plot_data(
        timestep_data, 'avg_total_reward',
        'timesteps_so_far', 'run_name', 'name',
        env, axes[1])

    # plot by seconds
    ts_data = normalize_to_seconds(
        data, 'time_elapsed', 'run_name', 'name')
    plot3 = plot_data(
        ts_data, 'avg_total_reward', 'time_elapsed',
        'run_name', 'name', env, axes[2])

    figure.add_subplot(plot1)
    figure.add_subplot(plot2)
    figure.add_subplot(plot3)

    plt.tight_layout()

    return figure
