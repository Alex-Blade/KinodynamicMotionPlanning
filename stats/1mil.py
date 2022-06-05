import logging
import os
from itertools import repeat


from pathplanning.environment import Map, Car
from pathplanning.rrt import RRT, ReedsShepp

import random
import string
import numpy as np
import pandas as pd
import time
from copy import deepcopy


from multiprocessing.pool import Pool


ENV = Map.read_svg("/home/alex/Study/pathplanning/images/example3.svg")
CAR = Car(4.42, 1.7, 5.12)


GRID = {
    "max_edge_len": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "goal_rate": [0.1, 0.3, 0.5]
}


def init(car):
    env = deepcopy(ENV)
    source = random.randint(0, len(env.obstacles) - 1)
    start = env.obstacles.pop(source)
    start = (*start.center,
             3.14 / 2 if start.bounding_box[2] - start.bounding_box[0] < start.bounding_box[3] - start.bounding_box[
                 1] else 0)

    dest = random.randint(0, len(env.obstacles) - 1)
    end = env.obstacles.pop(dest)
    end = (*end.center,
           3.14 / 2 if end.bounding_box[2] - end.bounding_box[0] < end.bounding_box[3] - end.bounding_box[1] else 0)
    env.__post_init__()

    if not env.is_free(x=start[0], y=start[1], car=car, phi=start[2]):
        raise ValueError("FIX ME")

    return env, start, end


def run_testing(args):
    goal_rate, max_edge_len, prefix, num_iter, thread_id = args
    logging.info(f"Started R[{goal_rate}] M[{max_edge_len}] ID[{thread_id}]")
    car = deepcopy(CAR)
    df = pd.DataFrame(columns=["start_x", "start_y",
                               "end_x", "end_y",
                               "euclidian_distance",
                               "max_edge_len", "goal_rate",
                               "finished", "iter_count", "final_distance"])

    local_planner = ReedsShepp(max_edge_len=max_edge_len, step_size=0.5, turning_radius=car.turning_radius)
    logger = logging.getLogger(thread_id)
    for _ in range(num_iter):
        logger.info(f"Iteration {_ + 1}/{num_iter}")
        env, start, end = init(car)

        euclidian_dist = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5

        to_add = [start[0], start[1], end[0], end[1], euclidian_dist, max_edge_len, goal_rate]

        rrt = RRT(env, local_planner=local_planner, car=car, precision=(1, 1, 3.14))
        rrt.set_start(start)
        result = rrt.run(end, num_iterations=2000, goal_rate=goal_rate, show_plots=False)
        if result is not None:
            to_add.extend([True, result, rrt.whole_path_distance])
        else:
            to_add.extend([False, np.NaN, np.NaN])
        to_add = pd.DataFrame(dict(zip(df.columns, to_add)), index=[0])
        df = pd.concat([df, to_add], ignore_index=True, axis=0)

    df.to_csv(f"{prefix}result_{num_iter}_{thread_id}.csv")


def main():
    np.random.seed(42)
    random.seed(42)
    logging.basicConfig(level=logging.INFO)

    num = 6
    attempts = 100
    pool = Pool(num)

    all_args = []

    for rate in GRID["goal_rate"]:
        for max_edge_len in GRID["max_edge_len"]:
            s_time = time.perf_counter()
            prefix = f"results/R{rate}/M{max_edge_len}/"
            os.makedirs(prefix, exist_ok=True)
            arg1 = repeat(rate, num)
            arg2 = repeat(max_edge_len, num)
            arg3 = repeat(prefix, num)
            arg4 = repeat(attempts, num)
            arg5 = (''.join(random.choices(string.ascii_letters, k=num)) for _ in range(num))
            all_args.extend([_ for _ in zip(arg1, arg2, arg3, arg4, arg5)])

    logging.info(f"There will be {len(all_args)} tests")
    pool.map(run_testing, all_args)
    e_time = time.perf_counter()
    logging.info(f"Finished R[{rate}] M[{max_edge_len}]: {e_time - s_time}s")


if __name__ == '__main__':
    main()
