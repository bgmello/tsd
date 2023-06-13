import numpy as np
import os
import concurrent.futures
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from typing import List, Tuple


def get_consistent_ys(line: List[Tuple[float, float]]) -> List[float]:
    """
    Given a single line, find the y values for the determined x ticks.
    """
    x_ticks = np.linspace(0, 100, 201)
    y = []
    for x in x_ticks:
        i = 0
        while i < len(line) and line[i][0] <= x:
            i += 1
        # Append the y value of the closest point that is not greater than x
        y.append(line[i-1][1] if i > 0 else line[i][1])
    return y


def merge_lines(lines: List[List[Tuple[float, float]]]) -> Tuple[List[float], List[float]]:
    """
    Given multiple lines in the form [(x,y), (x2, y2), ...], returns the aggregated line
    in the form ([x, x2, ...], [y, y2, ...])
    """
    consistent_ys = [get_consistent_ys(line) for line in lines]
    x_ticks = np.linspace(0, 100, 201)
    y_avgs = [np.mean([consistent_y[i] for consistent_y in consistent_ys]) for i in range(len(x_ticks))]
    return x_ticks, y_avgs


def get_xs_ys(iteration: dict) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Given a iteration dict, returns three lists with the points (x,y) for each algorithm in the form:
        [(x,y), (x2,y2), ...]
    """
    final_error = min(min(iteration['tsd_inner_objective']), min(iteration['tsd_inner_objective_rand']),
                      min(iteration['rgd_objective']))

    rgd_times = np.array(iteration["rgd_time"]).cumsum()
    max_time = rgd_times[rgd_times < iteration["max_time"]][-1]

    tsd = [(100*time, 100*final_error/obj)
           for time, obj in zip(np.array(iteration['tsd_inner_time']).cumsum()/max_time, iteration['tsd_inner_objective']) if time <= 1]
    tsd_rand = [
        (100 *
         time,
         100 *
         final_error /
         obj) for time,
        obj in zip(
            np.array(
                iteration['tsd_inner_time_rand']).cumsum() /
            max_time,
            iteration['tsd_inner_objective_rand']) if time <= 1]
    rgd = [(100*time, 100*final_error/obj)
           for time, obj in zip(np.array(iteration['rgd_time']).cumsum()/max_time, iteration['rgd_objective']) if time <= 1]

    return tsd, tsd_rand, rgd


def get_single_lines(data: List[dict]) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]], Tuple[List[float], List[float]]]:
    """
    Given the data for a given instance, get the lines that best fits for each algorithm

    """

    tsds, tsds_rand, rgds = [], [], []

    for iteration in data:
        tsd, tsd_rand, rgd = get_xs_ys(iteration)
        tsds.append(tsd)
        tsds_rand.append(tsd_rand)
        rgds.append(rgd)

    return merge_lines(tsds), merge_lines(tsds_rand), merge_lines(rgds)


def plot_single_lines(m, d, r, data: List[dict]) -> plt.Figure:

    fig, axes = plt.subplots(figsize=[10, 6], ncols=1, nrows=1, constrained_layout=True)

    tsd_line, tsd_rand_line, rgd_line = get_single_lines(data)

    x_tsd, y_tsd = tsd_line
    x_tsd_rand, y_tsd_rand = tsd_rand_line
    x_rgd, y_rgd = rgd_line
    sns.lineplot(y=y_tsd, x=x_tsd, ax=axes, linestyle='-.', color='b')
    sns.lineplot(y=y_tsd_rand, x=x_tsd_rand,
                 ax=axes, linestyle=':', color='g')
    sns.lineplot(y=y_rgd, x=x_rgd,
                 ax=axes, color='r')

    axes.set(yscale="log")
    axes.legend(title='Algorithms')
    axes.set_xlabel('% time elapsed')
    axes.set_ylabel('% gap closed')
    axes.set_title(f'(m, d, r)=({m}, {d}, {r})')
    axes.xaxis.set_major_formatter(ScalarFormatter())
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    ]

    # Add the custom legend to the figure with the specified label
    axes.legend(custom_lines, ['det. TSD', 'rand. TSD', 'GD'])

    # Optional: You can also disable the offset in the formatter
    axes.xaxis.get_major_formatter().set_useOffset(False)

    return fig


def plot_multi_lines(m, d, r, data: List[dict]) -> plt.Figure:
    fig, axes = plt.subplots(figsize=[10, 6], ncols=1, nrows=1, constrained_layout=True)

    for iteration in data:

        tsd_line, tsd_rand_line, rgd_line = get_xs_ys(iteration)

        x_tsd, y_tsd = zip(*tsd_line)
        x_tsd_rand, y_tsd_rand = zip(*tsd_rand_line)
        x_rgd, y_rgd = zip(*rgd_line)
        sns.lineplot(y=y_tsd, x=x_tsd, ax=axes, color='b')
        sns.lineplot(y=y_tsd_rand, x=x_tsd_rand,
                     ax=axes, color='g')
        sns.lineplot(y=y_rgd, x=x_rgd,
                     ax=axes, color='r')

    axes.set(yscale="log")
    axes.legend(title='Algorithms')
    axes.set_xlabel('% time elapsed')
    axes.set_ylabel('% gap closed')
    axes.set_title(f'(m, d, r)=({m}, {d}, {r})')
    axes.xaxis.set_major_formatter(ScalarFormatter())
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='green', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    ]

    # Add the custom legend to the figure with the specified label
    axes.legend(custom_lines, ['det. TSD', 'rand. TSD', 'GD'])

    axes.xaxis.get_major_formatter().set_useOffset(False)

    return fig


def plot_instance(m, d, r):
    try:
        if os.path.exists(f"data/plot_{m}_{d}_{r}_single.pdf"):
            return
        data = []

        for seed in range(2):
            tmp = {}
            for algo in ["tsd", "tsd_rand", "rgd"]:
                with open(f"data/wishart_{m}_{d}_{r}_seed_{seed}_algo_{algo}.json", "r") as f:
                    tmp = tmp | json.loads(f.read())
            data.append(tmp)
        plot_single_lines(m, d, r, data).savefig(f"data/plot_{m}_{d}_{r}_single.pdf")
        plot_multi_lines(m, d, r, data).savefig(f"data/plot_{m}_{d}_{r}_multi.pdf")
    except BaseException as e:
        print(e)


if __name__ == "__main__":
    ms = [1000]
    ds = [50]
    rs = [10, 50]
    num_seeds = 10

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for m in ms:
            for d in ds:
                for r in rs:
                    futures.append(executor.submit(plot_instance, m, d, r))

        for future in concurrent.futures.as_completed(futures):
            print(f"Process {future} completed")
