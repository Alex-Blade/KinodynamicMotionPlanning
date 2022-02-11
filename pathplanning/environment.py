from dataclasses import dataclass, field
from operator import itemgetter, attrgetter
from typing import Tuple, Union, Type, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.spatial.kdtree import KDTree
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, GeometryCollection
from pathplanning.types import *


@dataclass
class Obstacle:
    """
    An object representing a polygonal obstacle.

    Attrs:
        points: A container with coordinates of all points of the polygon
        bounding_box: Coordinates of a bounding box of the polygon
        center: Coordinates of the center point of the polygon
        polygon: Instance of shapely.geometry.Polygon
        radius: Radius of a circle that would cover the whole Obstacle
    """
    points: List[Position]
    bounding_box: Union[Tuple[Position, Position, Position, Position]] = field(init=False)
    center: Position = field(init=False)
    polygon: Polygon = field(init=False)

    def __post_init__(self):
        o0 = itemgetter(0)
        o1 = itemgetter(1)
        # self.bounding_box = (min(self.points, key=o0)[0],
        #                      min(self.points, key=o1)[1],
        #                      max(self.points, key=o0)[0],
        #                      max(self.points, key=o1)[1])
        self.polygon = Polygon(self.points)
        self.bounding_box = self.polygon.bounds
        self.center = self.polygon.centroid.coords[0]
        self.radius = max([np.linalg.norm(np.asarray(self.center) - np.asarray(p)) for p in self.bounding_box])

    @classmethod
    def generate_random(cls: Type["Obstacle"],
                        map_limits: Tuple[Number, Number],
                        size: Number,
                        num_points: int) -> "Obstacle":
        """
        # TODO: this is painfully broken
        Generate random convex Obstacle.

        Args:
            map_limits: Size of a two-dimensional map
            size: Size of the Obstacle
            num_points: Number of points in the Polygon representing the Obstacle
        Returns:
            An instance of Obstacle
        """
        center = np.array((np.random.rand() * map_limits[0],
                           np.random.rand() * map_limits[1]))
        angles = sorted((np.random.rand() * 2 * np.pi for _ in range(num_points)))
        points = np.array([center, *[np.array([size*np.cos(angle), size*np.sin(angle)]) for angle in angles]])

        return cls(points, center)

    def collides(self, x: Number, y: Number) -> bool:
        """
        Check if the given point is in the obstacle or not.
        
        Args:
            x: X-axis coordinate
            y: Y-axis coordinate
        Returns:
            True if collides, False otherwise
        """
        return self.polygon.contains(Point(x, y))

    def plot(self):
        """
        Add obstacle to the current plot.
        """
        plt.gca().add_patch(pat.Polygon(self.points, color='black', fill=True))


@dataclass
class Map:
    """
    Two-dimensional map representation that holds polygonal obstacles.

    map_limits: Size of a two-dimensional map
    obstacles: A list of Obstacles
    kdtree: KDTree that holds obstacles
    max_radius: Maximum radius that will be queried in KDTree
    """
    map_limits: Tuple[Number, Number]
    obstacles: List[Obstacle] = field(default_factory=list)

    def __post_init__(self):
        self.kdtree = KDTree([o.center for o in self.obstacles])
        self.max_radius = max(self.obstacles, key=attrgetter("radius")).radius + 0.05
        # TODO: rethink this part...
        # When obstacles in the map are big (like in a labyrinth) we are essentially scanning the whole map
        # because max_radius is enormous.

    @classmethod
    def generate_random_map(cls, map_limits: Tuple[Number, Number], num_obstacles: int) -> "Map":
        """
        # TODO: this is painfully broken
        """
        obstacles = [Obstacle.generate_random(map_limits, 0.05*map_limits[0], 5) for _ in range(num_obstacles)]
        return cls(map_limits, obstacles)

    def get_close_obstacles(self, x: Number, y: Number) -> List[Obstacle]:
        """
        Scan KDTree for obstacles that are close to the given point.

        Args:
            x: X-coordinate of the target point
            y: Y-coordinate of the target point
        Returns:
            A list of Obstacles that are within max_radius of the point
        """
        return [self.obstacles[index] for index in self.kdtree.query_ball_point((x, y), r=self.max_radius, workers=-1)]

    def is_free(self, x: Number, y: Number) -> bool:
        """
        Check if point collides with any Obstacle.

        Args:
            x: X-coordinate of the target point
            y: Y-coordinate of the target point
        Returns:
            True if it does, False otherwise
        """
        if not ((0 <= x <= self.map_limits[0]) and (0 <= y <= self.map_limits[1])):
            return False

        for obstacle in self.get_close_obstacles(x, y):
            if obstacle.collides(x, y):
                return False
        return True

    def random_free_point(self) -> Position:
        """
        Get random point that does not collide with anything.

        Returns:
            Coordinates of the point and angle.
        """
        x, y = np.random.rand() * self.map_limits[0], np.random.rand() * self.map_limits[1]
        while not self.is_free(x, y):
            x, y = np.random.rand() * self.map_limits[0], np.random.rand() * self.map_limits[1]
        return x, y, np.random.rand() * np.pi * 2

    def plot(self, close=False, display=True):
        """
        Create a figure and plot the map with obstacles on it.

        Args:
            close: Whether to close the plot or not
            display: Whether to show the plot or not
        """
        plt.ion() if display else plt.ioff()
        for obstacle in self.obstacles:
            obstacle.plot()
        plt.gca().set_xlim(0, self.map_limits[0])
        plt.gca().set_ylim(0, self.map_limits[1])
        if close:
            plt.close()

