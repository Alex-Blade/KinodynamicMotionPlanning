import re
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Tuple, Union, Type, List
import xml.etree.ElementTree as ET

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import rtree
import shapely.ops as ops
import shapely.affinity as aff
import json
from shapely.geometry import Polygon, Point

from pathplanning.rrt_types import *


@dataclass
class Obstacle:
    """
    An object representing a polygonal obstacle.

    Attrs:
        points: A container with coordinates of all points of the polygon
        bounding_box: Coordinates of a bounding box of the polygon
        center: Coordinates of the center point of the polygon
        polygon: Instance of shapely.geometry.Polygon
    """
    points: List[Position]
    bounding_box: Tuple[Position, Position, Position, Position] = field(init=False)
    cut_factor: Number = 0.25
    center: Position = field(init=False)
    polygon: Polygon = field(init=False)
    _meta: str = field(default_factory=str)

    def __post_init__(self):
        self.polygon = Polygon(self.points)
        self.bounding_box = self.polygon.bounds
        self.center = self.polygon.centroid.coords[0]

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

    def boxify(self) -> List[Tuple[Position, Position, Position, Position]]:
        x_min, y_min, x_max, y_max = self.bounding_box
        if y_max - y_min < x_max - x_min:
            while y_min < y_max:
                y_up = min(y_max, y_min + self.cut_factor)
                pl = ops.clip_by_rect(self.polygon, x_min, y_min, x_max, y_up)
                if not pl.bounds:
                    break
                yield pl.bounds
                y_min += self.cut_factor
        else:
            while x_min < x_max:
                x_up = min(x_max, x_min + self.cut_factor)
                pl = ops.clip_by_rect(self.polygon, x_min, y_min, x_up, y_max)
                if not pl.bounds:
                    break
                yield pl.bounds
                x_min += self.cut_factor

    def plot(self):
        """
        Add obstacle to the current plot.
        """
        # plt.gca().add_patch(pat.Polygon(self.points, color='black', fill=True))
        for box in self.boxify():
            xmin, ymin, xmax, ymax = box
            plt.gca().add_patch(pat.Polygon([
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin)
            ], color='red', fill=False))


@dataclass
class Car:
    x_size: Number
    y_size: Number
    turning_radius: Number
    _position: Union[Position, None] = field(init=False)
    _obstacle: Union[Obstacle, None] = field(init=False)

    def __post_init__(self):
        self._position = None
        self._obstacle = None

    @property
    def position(self):
        if self._position is None:
            raise ValueError("No position was set")
        return self._position

    @position.setter
    def position(self, pos: Position):
        self._obstacle = None
        self._position = pos

    @property
    def obstacle(self):
        if self._obstacle:
            return self._obstacle

        polygon = Polygon(((self.position[0] - self.x_size / 2, self._position[1] - self.y_size / 2),
                           (self.position[0] - self.x_size / 2, self._position[1] + self.y_size / 2),
                           (self.position[0] + self.x_size / 2, self._position[1] + self.y_size / 2),
                           (self.position[0] + self.x_size / 2, self._position[1] - self.y_size / 2)))
        polygon: Polygon = aff.rotate(polygon, self._position[2], use_radians=True)
        points = [p for p in polygon.exterior.coords]
        self._obstacle = Obstacle(points, cut_factor=0.25)
        return self._obstacle


@dataclass
class Map:
    """
    Two-dimensional map representation that holds polygonal obstacles.

    Attrs:
        map_limits: Size of a two-dimensional map
        obstacles: A list of Obstacles
        rtree: RTree that holds obstacles
    """
    map_limits: Tuple[Number, Number]
    obstacles: List[Obstacle] = field(default_factory=list)
    cut_factor: Number = 1

    def __post_init__(self):
        self.rtree = rtree.Index()
        for obstacle in self.obstacles:
            obstacle.cut_factor = self.cut_factor
            for box in obstacle.boxify():
                self.rtree.insert(id(obstacle), box)

    @classmethod
    def generate_random_map(cls, map_limits: Tuple[Number, Number], num_obstacles: int) -> "Map":
        """
        # TODO: this is painfully broken
        """
        obstacles = [Obstacle.generate_random(map_limits, 0.05*map_limits[0], 5) for _ in range(num_obstacles)]
        return cls(map_limits, obstacles)

    def is_free(self, x: Number, y: Number, car: Car = None, phi: Number = None) -> bool:
        """
        Check if point collides with any Obstacle.

        Args:
            x: X-coordinate of the target point
            y: Y-coordinate of the target point
            car: Instance of car to check for collisions
            phi: Angle of the car if car was passed
        Returns:
            True if it does, False otherwise
        """
        if car is not None:
            car.position = (x, y, phi)
            for pos in car.obstacle.polygon.exterior.coords:
                if not ((0 <= pos[0] <= self.map_limits[0]) and (0 <= pos[1] <= self.map_limits[1])):
                    return False
            for box in car.obstacle.boxify():
                if self.rtree.count(box):
                    return False
        else:
            if not ((0 <= x <= self.map_limits[0]) and (0 <= y <= self.map_limits[1])):
                return False
            cnt = self.rtree.count((x, y))
            if cnt:
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

    @classmethod
    def read_svg(cls, path: str) -> "Map":
        root = ET.parse(path).getroot()
        obstacles = []
        map_settings = {}
        for child in root:
            if child[0].text == "background":
                map_settings = child[1].attrib
            else:
                child[:] = sorted(child, key=lambda c: c.get("id", ""))
                for c in child:
                    if c.tag.endswith("rect"):
                        o = c.attrib
                        x, y, w, h = float(o['x']) / 100, float(o['y']) / 100, float(o['width']) / 100, float(o['height']) / 100
                        rotation = int(re.match("rotate\((\d+)", o.get("transform", "rotate(0")).group(1))
                        p = Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                        p = aff.rotate(p, rotation, use_radians=False)
                        obstacles.append(Obstacle(p.exterior.coords, _meta=o.get("id")))
        return cls((float(map_settings['height']) / 100, float(map_settings['width']) / 100), obstacles)

    @classmethod
    def read_json(cls, path: str) -> "Map":
        with open(path) as file:
            data = json.load(file)
            obstacles = []
            for i, o in enumerate(data['obstacles']):
                p = Polygon(o['coordinates'])
                if angle := o.get('angle'):
                    p = aff.rotate(p, angle, use_radians=False)
                obstacles.append(Obstacle(p.exterior.coords, _meta=str(i)))
            return cls((float(data['height']), float(data['width'])), obstacles)

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
        if display:
            plt.show()
        if close:
            plt.close()

