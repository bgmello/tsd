import numpy as np
import os
import ujson as json
import matplotlib.pyplot as plt
from typing import List, Tuple


def get_consistent_ys(line) -> np.array:
    line = np.array(line)
    x_ticks = np.linspace(0, 100, 201)
    max_y_values = []

    for x_tick in x_ticks:
        # Find the y values for x <= x_tick
        y_values = line[line[:, 0] <= x_tick, 1]
        # If there are any y values, find the max, otherwise use NaN (or some other value)
        max_y = np.max(y_values) if len(y_values) > 0 else np.nan
        max_y_values.append(max_y)
    return np.array(max_y_values)


def merge_lines(lines: List[np.ndarray]) -> Tuple[np.array, np.array]:
    consistent_ys = np.array([get_consistent_ys(line) for line in lines])
    x_ticks = np.linspace(0, 100, 201)
    y_avgs = np.mean(consistent_ys, axis=0)
    return x_ticks, y_avgs


def get_xs_ys(iteration: dict) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    rgd_times = np.array(iteration["rgd_time"]).cumsum()
    max_time = rgd_times[rgd_times < iteration["max_time"]][-1]

    tsd_times = np.array(iteration['tsd_inner_time']).cumsum()/max_time
    tsd_rand_times = np.array(iteration['tsd_inner_time_rand']).cumsum()/max_time
    rgd_times = np.array(iteration['rgd_time']).cumsum()/max_time

    tsd_objective_filtered = [obj for time, obj in zip(tsd_times, iteration['tsd_inner_objective']) if time <= 1]
    tsd_objective_rand_filtered = [obj for time, obj in zip(tsd_rand_times, iteration['tsd_inner_objective_rand']) if time <= 1]
    rgd_objective_filtered = [obj for time, obj in zip(rgd_times, iteration['rgd_objective']) if time <= 1]

    final_error = min(min(tsd_objective_filtered, default=float('inf')),
                      min(tsd_objective_rand_filtered, default=float('inf')),
                      min(rgd_objective_filtered, default=float('inf')))

    tsd = [(100*time, 100*final_error/obj)
           for time, obj in zip(tsd_times, iteration['tsd_inner_objective']) if time <= 1]
    tsd_rand = [(100*time, 100*final_error/obj)
                for time, obj in zip(tsd_rand_times, iteration['tsd_inner_objective_rand']) if time <= 1]
    rgd = [(100*time, 100*final_error/obj)
           for time, obj in zip(rgd_times, iteration['rgd_objective']) if time <= 1]

    return tsd, tsd_rand, rgd


def get_single_lines(data: List[dict]) -> Tuple[np.array, np.array, np.array]:
    tsds, tsds_rand, rgds = [], [], []

    for iteration in data:
        tsd, tsd_rand, rgd = get_xs_ys(iteration)
        tsds.append(tsd)
        tsds_rand.append(tsd_rand)
        rgds.append(rgd)

    return merge_lines(tsds), merge_lines(tsds_rand), merge_lines(rgds)


def generate_single_lines(m, d, r, data: List[dict]) -> plt.Figure:

    fig, axes = plt.subplots(figsize=[10, 6], ncols=1, nrows=1, constrained_layout=True)

    tsd_line, tsd_rand_line, rgd_line = get_single_lines(data)

    x_tsd, y_tsd = tsd_line
    x_tsd_rand, y_tsd_rand = tsd_rand_line
    x_rgd, y_rgd = rgd_line

    return {
        "x_tsd": list(x_tsd),
        "y_tsd": list(y_tsd),
        "x_tsd_rand": list(x_tsd_rand),
        "y_tsd_rand": list(y_tsd_rand),
        "x_rgd": list(x_rgd),
        "y_rgd": list(y_rgd)
    }


def plot_instance(m, d, r):
    if os.path.exists(f"data/final_data_{m}_{d}_{r}_single.json"):
        return
    data = []

    for seed in range(10):
        tmp = {}
        for algo in ["tsd", "tsd_rand", "rgd"]:
            with open(f"data/wishart_{m}_{d}_{r}_seed_{seed}_algo_{algo}.json", "r") as f:
                tmp = tmp | json.loads(f.read())
        data.append(tmp)
    with open(f"data/final_data_{m}_{d}_{r}_single.json", "w") as f:
        f.write(json.dumps(generate_single_lines(m, d, r, data)))


if __name__ == "__main__":
    ms = [1000, 10000]
    ds = [50, 100, 500, 1000]
    rs = [10, 50]

    num_seeds = 10

    for m in ms:
        for d in ds:
            for r in rs:
                plot_instance(m, d, r)
