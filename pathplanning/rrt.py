import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import random
from itertools import chain

import dubins
import reeds_shepp
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import networkx as nx
import numpy as np

from pathplanning.environment import Map, Car
from pathplanning.rrt_types import *

from collections import defaultdict
from time import perf_counter


@dataclass(frozen=True)
class Node:
    """
    Node of the RRT.

    Attrs:
        position: Two-dimensional coordinates tuple
        cost: Cost to reach this node
    """

    position: Position
    parent: Optional[Position]
    cost: Number


@dataclass
class Link:
    """
    Edge of the RRT.

    Attrs:
        source: The "from" node of the Link
        dest: The "to" node of the Link
        path: A list of coordinates to follow
        cost: Cost to follow the path
    """

    source: Node
    dest: Node
    path: List[Position]
    cost: Number


class SteeringFunction:
    def __init__(self, max_edge_len: Number, turning_radius: Number = 1.0, step_size: Number = 0.5):
        self.turning_radius = turning_radius
        self.step_size = step_size
        self.max_edge_len = max_edge_len

    def get_shortest_path(self, source: Position, dest: Position) -> LenPath:
        """
        Get shortest Dubins curve from source to destination.

        Args:
            source: Initial point
            dest: Target point
        Returns:
            Unsampled path
        """
        pass

    def sample_points(self, path: Any) -> List[Position]:
        """
        Sample points from given unsampled path.

        Args:
            path: Output of `get_shortest_path` that will be used for sampling
        Returns:
            A list of points for the robot to follow
        """
        pass


class Dubbins(SteeringFunction):
    def get_shortest_path(self, source: Position, dest: Position) -> LenPath:
        path = dubins.shortest_path(source, dest, self.turning_radius)
        return path.path_length(), path

    def sample_points(self, path) -> List[Position]:
        path_length = path.path_length()
        segment_length = path_length if path_length < self.max_edge_len else self.max_edge_len
        segment = path.extract_subpath(segment_length)
        configurations, _ = segment.sample_many(self.step_size)
        return configurations


class ReedsShepp(SteeringFunction):
    def get_shortest_path(self, source: Position, dest: Position) -> LenPath:
        path = reeds_shepp.PyReedsSheppPath(source, dest, self.turning_radius)
        return path.distance(), path

    def sample_points(self, path) -> List[Position]:
        conf = path.sample(self.step_size)
        return conf[:int((self.max_edge_len // self.step_size) + 1)]


class RRT:
    """
    The RRT solver.

    Attrs:
        nodes: A map that holds instances of Node with coordinates as keys
        edges: nodes: A map that holds instances of Link with two coordinates as keys
        map: An instance of Map
        goal: Coordinates of the target point with angle
        root: Coordinates of the initial point with angle
        precision: Error level that will be ignored
        final_node: Node that was chosen as final withing the precision
        whole_path: A path from `root` to `goal` represented by a list of Node coordinates
    Args:
        _map: An instance of a Map
        car: An instance of a Car
        local_planner: An instance of SteeringFunction
        precision: Error level that will be ignored
    """
    def __init__(self, _map: Map,
                 car: Car,
                 local_planner: SteeringFunction,
                 precision=(5, 5, 1)):
        self.nodes = {}
        self.edges = {}
        self.map = _map
        self.car = car
        self.local_planner = local_planner
        self.goal = (0, 0, 0)
        self.root = (0, 0, 0)
        self.precision = precision
        self.final_node: Optional[Node] = None

    @property
    def whole_path(self) -> Optional[List[Node]]:
        if self.final_node is None:
            return None
        path = []
        node = self.nodes[self.final_node]
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)
        path.reverse()
        return path

    @property
    def whole_path_distance(self):
        path = self.whole_path
        edges = self.edges.copy()
        s = 0
        for inode, jnode in zip(path, path[1:]):
            val = edges.pop((inode, jnode))
            if val.path:
                arr = np.array(val.path)
                points = arr[:, 0:2]
                d = np.diff(points, axis=0)
                s += np.sum(np.hypot(d[:, 0], d[:, 1]))
        return s

    def set_start(self, start: Position):
        """
        Reset the tree and set new start point.

        Args:
            start: Coordinates of the new start point
        """
        self.nodes = {}
        self.edges = {}
        self.nodes[start] = Node(start, None, 0)
        self.root = start
        self.car.position = start
        _ = self.car.obstacle  # Initialize

    def select_options(self, sample: Position, max_options: int) -> List[Tuple[Position, LenPath]]:
        """
        Find `max_options` existing Nodes that are the closest to the `sample`.
        Distance to the Node is calculated on Dubins curves.

        Args:
            sample: Coordinates of the target point
            max_options: Maximum number of Nodes to output
        Returns:
            A list of tuples that contain Nodes' coordinates and unsampled Dubins paths to them.
        """
        options = []
        for node in self.nodes:
            options.extend([(node, self.local_planner.get_shortest_path(node, sample))])
        options.sort(key=lambda x: x[1][0])
        options = options[:max_options]
        return options

    def in_goal_region(self, sample: Position) -> bool:
        """
        Check if `sample` is close to the goal (withing precision)

        Args:
            sample: Coordinates of the target point
        Returns:
            True if it is, False otherwise
        """
        for i, value in enumerate(sample):
            if abs(self.goal[i]-value) > self.precision[i]:
                return False
        return True

    def run_animated(self, path: str, goal: Position, num_iterations=100, goal_rate=.1, fps: int = 5, show_plots: bool = False):
        fig = plt.gcf()
        moviewriter = PillowWriter(fps=fps)
        with moviewriter.saving(fig, path, dpi=100):
            self.run(goal, num_iterations=num_iterations, goal_rate=goal_rate, show_plots=show_plots, moviewriter=moviewriter)
        return

    def run(self, goal, num_iterations=100, goal_rate=.1,
            show_plots: bool = None,
            moviewriter: matplotlib.animation.MovieWriter = None):
        """
        Execute the algorithm with a graph initialized with the start position.

        Args:
            goal: Coordinates and angle of the destination
            num_iterations: Maximum available iterations
            goal_rate: The probability to expand towards the goal

        Note:
            Instead of a result, the algorithm will set `final_node` if the path exists
        """

        if len(goal) != len(self.precision):
            raise ValueError("Dimensionality mismatch in goal and precision")
        self.goal = goal

        for _ in range(num_iterations):
            # Step 1. Sample a point
            if np.random.rand() > 1 - goal_rate:
                sample = goal
            else:
                sample = self.map.random_free_point()

            # Step 2. Find closest points to the sampled
            options = [random.choice(self.select_options(sample, 10))]
            # Step 3. Among closest find one with a better path

            # Note, we do not consider all neighbors, meaning that we could be losing good samples
            for node, (distance, dpath) in options:
                # Step 3.1. Sample Dubins paths and check for intersections
                path = self.local_planner.sample_points(dpath)
                sample = path[-1]
                for i, point in enumerate(path):
                    free = self.map.is_free(point[0], point[1], car=self.car, phi=point[2])
                    if moviewriter:
                        plt.gcf().clear()
                        self.plot_all(show=False, close=False)
                        moviewriter.grab_frame()
                        logging.info(f"Prepared frame #{_}")
                    if not free:
                        break
                else:
                    # Step 3.2. Add the node to the graph if there's a path
                    if show_plots:
                        if _ % 10 == 0:
                            self.plot_all(show=True, close=False)
                    goal_reached = False
                    for idx, point in enumerate(path):
                        if self.in_goal_region(point):
                            sample = point
                            path = path[:idx + 1]
                            goal_reached = True
                            break

                    sample_node = Node(sample, self.nodes[node], self.nodes[node].cost + distance)
                    self.nodes[sample] = sample_node
                    # Adding the Edge
                    self.edges[self.nodes[node], sample_node] = Link(node, sample_node, path, distance)
                    # Step 3.3. Check if the goal is reached
                    if goal_reached:
                        self.final_node = sample
                        return _
                    break

    def plot_all(self, path='', close=False, show=True):
        self.map.plot(display=False)
        self.plot(include_nodes=True, include_edges=True)
        self.car.obstacle.plot()
        if show:
            plt.show()
        if path:
            plt.savefig(path)
        if close:
            plt.close()

    def plot(self, include_nodes=False, include_edges=False):
        nodes = list(self.nodes.keys())
        edges = self.edges.copy()
        if (path := self.whole_path) is not None:
            path_arr = np.array([p.position for p in path])
            plt.scatter(path_arr[:, 0], path_arr[:, 1])
            nodes = np.array(list(filter(lambda n: n not in path, nodes)))
            for inode, jnode in zip(path, path[1:]):
                val = edges.pop((inode, jnode))
                if val.path:
                    path = np.array(val.path)
                    plt.plot(path[:, 0], path[:, 1], 'g')

        if include_nodes and self.nodes:
            if len(nodes) > 1:
                try:
                    plt.scatter(nodes[:, 0], nodes[:, 1], alpha=0.3, c='gray', s=15)
                except TypeError:
                    pass
            plt.scatter(self.root[0], self.root[1], c='g')
            plt.scatter(self.goal[0], self.goal[1], c='r')

        if include_edges:
            for _, val in edges.items():
                if val.path:
                    path = np.array(val.path)
                    plt.plot(path[:, 0], path[:, 1], 'gray', alpha=0.3)

    def car_driving_gif(self, path: str, fps: int):
        fig = plt.gcf()
        moviewriter = PillowWriter(fps=fps)
        with moviewriter.saving(fig, path, dpi=100):
            edges = self.edges.copy()
            path = self.whole_path
            path_arr = np.array([p.position for p in path])
            plt.scatter(path_arr[:, 0], path_arr[:, 1])
            for inode, jnode in zip(path, path[1:]):
                val = edges.pop((inode, jnode))
                if val.path:
                    path = np.array(val.path)
                    for pos in path:
                        self.car.position = pos
                        plt.gcf().clear()
                        self.map.plot(display=False)
                        self.car.obstacle.plot()
                        self.plot(include_edges=False)
                        plt.gca().scatter(self.root[0], self.root[1], c='g')
                        plt.gca().scatter(self.goal[0], self.goal[1], c='r')
                        moviewriter.grab_frame()
