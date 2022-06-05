import logging

import matplotlib.pyplot as plt

from pathplanning.environment import Map, Car
from pathplanning.rrt import RRT, ReedsShepp, Dubbins

import random
import numpy as np


def main():
    logger = logging.getLogger("Example3")

    car = Car(4.42, 1.7, 5.12)
    # car = Car(4.0, 1.3, 4.0)
    env = Map.read_svg("images/example3.svg")
    for i, o in enumerate(env.obstacles):
        if o._meta == "svg_7":
            source = i
            break
    # source = 15 # random.randint(0, len(env.obstacles))
    start = env.obstacles.pop(source)
    start = (*start.center, 3.14 / 2 if start.bounding_box[2] - start.bounding_box[0] < start.bounding_box[3] - start.bounding_box[1] else 0)

    for i, o in enumerate(env.obstacles):
        # if o._meta == "svg_24":
        if o._meta == "svg_50":
            dest = i
            break

    # dest = 25  # random.randint(0, len(env.obstacles))
    end = env.obstacles.pop(dest)
    end = (*end.center, 3.14 / 2 if end.bounding_box[2] - end.bounding_box[0] < end.bounding_box[3] - end.bounding_box[1] else 0)
    env.__post_init__()

    if not env.is_free(x=start[0], y=start[1], car=car, phi=3.14 / 2):
        raise ValueError("FIX ME")

    lp = ReedsShepp(max_edge_len=5.0, step_size=0.5, turning_radius=car.turning_radius)
    lp1 = Dubbins(max_edge_len=1.5, step_size=0.5, turning_radius=car.turning_radius)

    rrt = RRT(env, local_planner=lp, car=car, precision=(1, 1, 3.14))
    logger.info(f"Initialized the map")

    logger.info(f"Got target points")

    # Initialisation of the tree, to have a first edge
    rrt.set_start(start)
    logger.info(f"Set start")
    np.random.seed(42)
    random.seed(42)
    rrt.run(end, num_iterations=10000, goal_rate=0.3, show_plots=False)
    rrt.plot_all()
    logger.info("Finished")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
