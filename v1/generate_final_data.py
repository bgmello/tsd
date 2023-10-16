import numpy as np
import os
import ujson as json
import matplotlib.pyplot as plt
from typing import List, Tuple


def get_consistent_ys(line) -> np.array:
    line = np.array(line)
    x_ticks = np.linspace(0, 100, 201)
    line_arr = np.array(line)
    idx = np.searchsorted(line_arr[:, 0], x_ticks, side='right') - 1
    idx[idx < 0] = 0
    return line_arr[idx, 1]


def merge_lines(lines: List[np.ndarray]) -> Tuple[np.array, np.array]:
    consistent_ys = np.array([get_consistent_ys(line) for line in lines])
    x_ticks = np.linspace(0, 100, 201)
    y_avgs = np.mean(consistent_ys, axis=0)
    return x_ticks, y_avgs

def get_xs_ys(iteration: dict) -> Tuple[np.array, np.array, np.array]:

    def process_algorithm(times, objectives):
        times_cum = np.array(times).cumsum()
        times_norm = (100 * times_cum / max_time)
        objectives_norm = np.array(objectives)
        mask = times_norm <= 100
        return np.vstack((times_norm[mask], objectives_norm[mask])).T

    rgd_times_cum = np.array(iteration["rgd_time"]).cumsum()
    max_time = rgd_times_cum[rgd_times_cum < iteration["max_time"]][-1]

    tsd = process_algorithm(iteration['tsd_inner_time'], iteration['tsd_inner_objective'])
    tsd_rand = process_algorithm(iteration['tsd_inner_time_rand'], iteration['tsd_inner_objective_rand'])
    rgd = process_algorithm(iteration['rgd_time'], iteration['rgd_objective'])

    # Get the final error based on the final objectives of each algorithm
    final_error = min(tsd[1, -1], tsd_rand[1, -1], rgd[1, -1])

    tsd[1, :] = 100 * final_error / tsd[1, :]
    tsd_rand[1, :] = 100 * final_error / tsd_rand[1, :]
    rgd[1, :] = 100 * final_error / rgd[1, :]

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


def generate_multi_lines(m, d, r, data: List[dict]) -> plt.Figure:
    fig, axes = plt.subplots(figsize=[10, 6], ncols=1, nrows=1, constrained_layout=True)

    final_data = []

    for iteration in data:


        tsd_line, tsd_rand_line, rgd_line = get_xs_ys(iteration)

        y_tsd = get_consistent_ys(tsd_line)
        y_tsd_rand = get_consistent_ys(tsd_rand_line)
        y_rgd = get_consistent_ys(rgd_line)
        x_ticks = np.linspace(0, 100, 201)

        final_data.append({
            "x_tsd": list(x_ticks),
            "y_tsd": list(y_tsd),
            "x_tsd_rand": list(x_ticks),
            "y_tsd_rand": list(y_tsd_rand),
            "x_rgd": list(x_ticks),
            "y_rgd": list(y_rgd)
            })


    return final_data


def plot_instance(m, d, r):
    if os.path.exists(f"data/final_data_{m}_{d}_{r}_multi.json") and os.path.exists(f"data/final_data_{m}_{d}_{r}_single.json"):
        return
    data = []

    for seed in range(10):
        tmp = {}
        for algo in ["tsd", "tsd_rand", "rgd"]:
            with open(f"data/wishart_{m}_{d}_{r}_seed_{seed}_algo_{algo}.json", "r") as f:
                tmp = tmp | json.loads(f.read())
        data.append(tmp)
    if not os.path.exists(f"data/final_data_{m}_{d}_{r}_single.json"):
        with open(f"data/final_data_{m}_{d}_{r}_single.json", "w") as f:
            f.write(json.dumps(generate_single_lines(m, d, r, data)))
    if not os.path.exists(f"data/final_data_{m}_{d}_{r}_multi.json"):
        with open(f"data/final_data_{m}_{d}_{r}_multi.json", "w") as f:
            f.write(json.dumps(generate_multi_lines(m, d, r, data)))


if __name__ == "__main__":
    ms = [1000, 10000]
    ds = [50, 100, 500, 1000]
    rs = [10, 50]

    num_seeds = 10

    for m in ms:
        for d in ds:
            for r in rs:
                plot_instance(m, d, r)
