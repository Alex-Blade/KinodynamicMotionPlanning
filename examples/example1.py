import logging

import matplotlib.pyplot as plt

from pathplanning.environment import Map, Obstacle
from pathplanning.rrt import RRT


def main():
    logger = logging.getLogger("Example1")

    o0 = Obstacle([[0, 5], [35, 5], [35, 10], [0, 10]])
    o1 = Obstacle([[5, 15], [40, 15], [40, 20], [5, 20]])
    o2 = Obstacle([[0, 25], [35, 25], [35, 30], [0, 30]])
    o3 = Obstacle([[5, 35], [40, 35], [40, 40], [5, 40]])
    env = Map((40, 40), [o0, o1, o2, o3])

    rrt = RRT(env, max_edge_len=5, turning_radius=0.8, precision=(1.5, 1.5, 1))
    logger.info(f"Initialized the map")

    start = env.random_free_point()
    end = env.random_free_point()
    logger.info(f"Got target points")

    # Initialisation of the tree, to have a first edge
    rrt.set_start(start)
    logger.info(f"Set start")

    rrt.run(end, num_iterations=10000000)

    env.plot()
    rrt.plot(include_nodes=True)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
