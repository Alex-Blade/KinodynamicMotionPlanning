import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import dubins
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pathplanning.environment import Map
from pathplanning.types import *


@dataclass
class Node:
    """
    Node of the RRT.

    Attrs:
        position: Two-dimensional coordinates tuple
        cost: Cost to reach this node
        destination_list: A list of other nodes that can be reached from the current
    """

    position: Position
    cost: Number
    destination_list: List["Node"] = field(init=False, default_factory=list)


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


class Dubins:
    """
    Args:
        turning_radius:
    """
    def __init__(self, turning_radius: Number = 1.0, step_size: Number = 0.5):
        self.turning_radius = turning_radius
        self.step_size = step_size

    def get_shortest_path(self, source: Position, dest: Position) -> LenPath:
        """
        Get shortest Dubins curve from source to destination.

        Args:
            source: Initial point
            dest: Target point
        Returns:
            Unsampled path (i.e. `dubins._DubinsPath`)
        """
        path = dubins.shortest_path(source, dest, self.turning_radius)
        return path.path_length(), path

    def sample_points(self, path: dubins._DubinsPath) -> List[Position]:
        """
        Sample points from given unsampled path.

        Args:
            path: Output of `get_shortest_path` that will be used for sampling
        Returns:
            A list of points for the robot to follow
        """
        configurations, _ = path.sample_many(self.step_size)
        return configurations


class RRT:
    """
    The RRT solver.

    Attrs:
        nodes: A map that holds instances of Node with coordinates as keys
        edges: nodes: A map that holds instances of Link with two coordinates as keys
        map: An instance of Map
        local_planner: Dubins solver initialized with `turning_radius` and `step_size`
        goal: Coordinates of the target point with angle
        root: Coordinates of the initial point with angle
        precision: Error level that will be ignored
        final_node: Node that was chosen as final withing the precision
        graph: Instance of nx.DiGraph that represents all current Links in adjacency matrix
        whole_path: A path from `root` to `goal` represented by a list of Node coordinates
    Args:
        _map: An instance of a Map
        turning_radius: Forward velocity divided by maximum angular velocity # TODO: make dynamic
        step_size: Discretization that will be used when sampling Dubins path
        precision: Error level that will be ignored
    """
    def __init__(self, _map: Map, turning_radius: Number = 1.0, step_size: Number = 0.5, precision=(5, 5, 1)):
        self.nodes = {}
        self.edges = {}
        self.map = _map
        self.local_planner = Dubins(turning_radius, step_size)
        self.goal = (0, 0, 0)
        self.root = (0, 0, 0)
        self.precision = precision
        self.final_node = None

    @property
    def graph(self):
        graph = nx.DiGraph()
        [graph.add_node(n) for n in self.nodes]
        for inode, jnode in self.edges:
            graph.add_edge(inode, jnode, weight=self.edges[inode, jnode].cost)
        return graph

    @property
    def whole_path(self) -> Optional[List[Position]]:
        if self.final_node is None:
            return None
        return nx.shortest_path(self.graph, self.root, self.final_node, weight='weight')

    def set_start(self, start: Position):
        """
        Reset the tree and set new start point.

        Args:
            start: Coordinates of the new start point
        """
        self.nodes = {}
        self.edges = {}
        self.nodes[start] = Node(start, 0)
        self.root = start

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

    def run(self, goal, num_iterations=100, goal_rate=.1):
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
            if _ % 50000 == 0 and _ != 0:
                self.map.plot()
                self.plot(include_nodes=True)
                plt.show()
            if (_ + 1) % 100 == 0:
                logging.info(f"Iteration {_}/{num_iterations}")
            ##############################################################
            #  Step 1. Sample a point                                    #
            ##############################################################
            if np.random.rand() > 1 - goal_rate:
                sample = goal
            else:
                sample = self.map.random_free_point()

            ##############################################################
            #  Step 2. Find closest points to the sampled                #
            ##############################################################
            options = self.select_options(sample, 10)

            ##############################################################
            # Step 3. Among closest find one with a better path          #
            ##############################################################

            # Note, we do not consider all neighbors, meaning that we could be losing good samples
            for node, (distance, dpath) in options:
                ##############################################################
                # Step 3.1. Sample Dubins paths and check for intersections  #
                ##############################################################
                path = self.local_planner.sample_points(dpath)
                for i, point in enumerate(path):
                    if not self.map.is_free(point[0], point[1]):
                        break
                else:
                    ##############################################################
                    # Step 3.2. Add the node to the graph if there's a path      #
                    ##############################################################
                    self.nodes[sample] = Node(sample, self.nodes[node].cost + distance)
                    self.nodes[node].destination_list.append(sample)
                    # Adding the Edge
                    self.edges[node, sample] = Link(node, sample, path, distance)
                    ##############################################################
                    # Step 3.3. Check if the goal is reached                     #
                    ##############################################################
                    if self.in_goal_region(sample):
                        self.final_node = sample
                        return
                    break

    def plot(self, file_name='', close=False, include_nodes=False):
        nodes = list(self.nodes.keys())
        edges = self.edges.copy()
        if (path := self.whole_path) is not None:
            path_arr = np.array(path)
            plt.scatter(path_arr[:, 0], path_arr[:, 1])
            nodes = np.array(list(filter(lambda n: n not in path, nodes)))
            for inode, jnode in zip(path, path[1:]):
                val = edges.pop((inode, jnode))
                if val.path:
                    path = np.array(val.path)
                    plt.plot(path[:, 0], path[:, 1], 'g')

        if include_nodes and self.nodes:
            if len(nodes) > 1:
                plt.scatter(nodes[:, 0], nodes[:, 1], alpha=0.3, c='gray', s=15)
            plt.scatter(self.root[0], self.root[1], c='g')
            plt.scatter(self.goal[0], self.goal[1], c='r')

        for _, val in edges.items():
            if val.path:
                path = np.array(val.path)
                plt.plot(path[:, 0], path[:, 1], 'gray', alpha=0.3)
        if file_name:
            plt.savefig(file_name)
        if close:
            plt.close()
