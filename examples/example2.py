import logging

import matplotlib.pyplot as plt

from pathplanning.environment import Map, Car
from pathplanning.rrt import RRT, ReedsShepp


def main():
    logger = logging.getLogger("Example2")

    car = Car(4.42, 1.7, 5.12)
    # car = Car(4.0, 1.3, 4.0)
    env = Map.read_svg("images/example2.2.svg")

    lp = ReedsShepp(max_edge_len=1.5, turning_radius=car.turning_radius, step_size=0.5)
    rrt = RRT(env, local_planner=lp, car=car, precision=(1, 1, 3.14))
    logger.info(f"Initialized the map")

    # start = (12.5, 13, 3.14 * 3 / 2)
    # end = (8, 3, 3.14 * 3 / 2)
    start = (12.8, 13, 3.14 * 3 / 2)
    end = (8.3, 3, 3.14 * 3 / 2)
    logger.info(f"Got target points")

    # Initialisation of the tree, to have a first edge
    rrt.set_start(start)
    logger.info(f"Set start")

    rrt.run(end, num_iterations=10000000, goal_rate=0.3, show_plots=False)
    rrt.plot_all(show=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
