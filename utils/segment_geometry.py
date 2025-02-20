"""
Work on segments from geojson file to extract the contour of rooms.

@Version: 0.1
@Project: Capstone Vinci Contour Detection
@Date: 2025-01-25
@Author: Tristan Waddington (GitHub:waddason)
"""

import json
import shapely  # buit with shapely 2.0.6
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from .parse_geojson import load_geojson

####################################################################################################
# Classes
# Implement the ideas from  M. Schäfer, C. Knapp, and S. Chakraborty, “Automatic
# generation of topological indoor maps for real-time map-
# based localization and tracking,” in Proceedings of Indoor
# Positioning and Indoor Navigation (IPIN), 2011 International
# Conference, pp. 1–8, IEEE, Guimaraes, Portugal, September 2011.
#
####################################################################################################
CLOSENESS_TOLERANCE = 1e-2  # 1 cm

####################################################################################################
# Classes
####################################################################################################
# Note: Since version 1.8, Shapely classes are build in C and do not allow easy inheritance.


# ---------------------------------------------------------------------------------------------------
# Pt
# ---------------------------------------------------------------------------------------------------
class Pt:
    """The point class"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Pt({self.x}, {self.y})"

    def __iter__(self):
        # Allow to unpack the point for creation in lines
        yield self.x
        yield self.y

    @property
    def to_shapely(self):
        return shapely.geometry.Point([self.x, self.y])

    def is_approx_eq(self, other, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if two points are close enough.
        Compare Euclidian distance to CLOSENESS_TOLERANCE = 1cm by default.

        Example:
        >>> a = Pt(0, 0)
        >>> b = Pt(0.01, 0)
        >>> a.is_approx_eq(b)
        True
        >>> c = Pt(0.1, 0.1)
        >>> a.is_approx_eq(c)
        False"""
        if not isinstance(other, Pt):
            return False
        return self.distance(other) <= tol

    def distance(self, other):
        return self.to_shapely.distance(other.to_shapely)


# ---------------------------------------------------------------------------------------------------
# Segment: a 2 points line
# ---------------------------------------------------------------------------------------------------
class Segment:
    """The segment class, correspond to the Line of Shäfer et al. 2011"""

    def __init__(self, start: Pt, end: Pt):
        self.start = start
        self.end = end
        self.line = shapely.geometry.LineString(
            [start.to_shapely, end.to_shapely]
        )

    def __str__(self):
        return f"Segment({self.start}, {self.end})"

    def __repr__(self):
        return f"Segment({self.start}, {self.end})"

    def __len__(self):
        return self.length

    @property
    def vector(self):
        """Return the vector of the segment"""
        return np.array([self.end.x - self.start.x, self.end.y - self.start.y])

    @property
    def length(self):
        """Return the length of the segment"""
        return np.linalg.norm(self.vector)

    @property
    def plotable(self):
        """Matplotlib plotable format"""
        return [self.start.x, self.end.x], [self.start.y, self.end.y]

    def is_approx_eq(self, other, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if two segments are close enough to be merged after.
        Compare Euclidian distance to CLOSENESS_TOLERANCE = 1cm by default.

        Example:
        >>> a = Segment(Pt(0, 0), Pt(1, 1))
        >>> b = Segment(Pt(0, 0.01), Pt(1.01, 1))
        >>> a.is_approx_eq(b)
        True
        >>> c = Segment(Pt(0.1, 0.1), Pt(1.1, 1.1))
        >>> a.is_approx_eq(c)
        False"""
        if not isinstance(other, Segment):
            return False
        return (
            self.start.is_approx_eq(other.start, tol)
            and self.end.is_approx_eq(other.end, tol)
        ) or (
            self.start.is_approx_eq(other.end, tol)
            and self.end.is_approx_eq(other.start, tol)
        )

    def is_point_adjacent(self, p: Pt, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if the point is adjacent to the segment"""
        if not isinstance(p, Pt):
            return False
        return self.start.is_approx_eq(p, tol) or self.end.is_approx_eq(p, tol)

    def is_line_adjacent(self, other, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if the line is adjacent to the segment. Meaning
        that only one end of the other line is adjacent to the segment."""
        # TODO: check if include in the other
        if not isinstance(other, Segment):
            return False
        return (
            self.is_point_adjacent(other.start, tol)
            and not self.is_point_adjacent(other.end)
        ) or (
            self.is_point_adjacent(other.end, tol)
            and not self.is_point_adjacent(other.start, tol)
        )

    def is_parallel(self, other, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if the two segments are parallel
        iff the dot product of their vectors almost
        equals the product of their lengths"""
        if not isinstance(other, Segment):
            return False
        return (
            abs(np.dot(self.vector, other.vector) - self.length * other.length)
            < tol
        )

    def is_orthogonal(self, other) -> bool:
        """Check if the two segments are orthogonal"""
        if not isinstance(other, Segment):
            return False
        return np.isclose(np.dot(self.vector, other.vector), 0)

    def contains_point(self, p: Pt, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if segment contains a point but not the extremities"""
        if not isinstance(p, Pt):
            return False
        if self.is_point_adjacent(p, tol):
            return False
        # see as a triangle, own interpretation, differ from paper
        left_segment = Segment(self.start, p)
        right_segment = Segment(p, self.end)
        return np.isclose(
            self.length, left_segment.length + right_segment.length, tol
        )

    def contains(self, geom, tol=CLOSENESS_TOLERANCE) -> bool:
        """Check if the segment contains a point or another segment"""
        if isinstance(geom, Pt):
            return self.contains_point(geom, tol)
        elif isinstance(geom, Segment):
            return self.contains_point(geom.start, tol) and self.contains_point(
                geom.end, tol
            )

    def concatenate(self, other, tol=CLOSENESS_TOLERANCE) -> "Segment":
        """Concatenate two adjacent parallel segments."""
        if (
            not isinstance(other, Segment)
            or not self.is_parallel(other, tol)
            or self.contains(other, tol)
            or not self.is_line_adjacent(other)
        ):
            return None
        # find the two adjacent points
        if self.start.is_approx_eq(other.start, tol):
            return Segment(self.end, other.end)
        elif self.start.is_approx_eq(other.end, tol):
            return Segment(self.end, other.start)
        elif self.end.is_approx_eq(other.start, tol):
            return Segment(self.start, other.end)
        elif self.end.is_approx_eq(other.end, tol):
            return Segment(self.start, other.start)
        else:
            return None


# ---------------------------------------------------------------------------------------------------
# Line: a multipoint line
# ---------------------------------------------------------------------------------------------------
class Line:
    """A multipoints line class"""

    # TODO
    def __init__(self, points: list[list[float]]):
        self.points = points

    def from_list(self, points: list[list[float]]):
        self.points = points


####################################################################################################
# Geojson
####################################################################################################
def filter_segments(geom_col):
    """Filter the geometry collection to keep only the segments."""
    segments = []
    for geom in geom_col:
        if isinstance(geom, shapely.geometry.LineString):
            segments.append(geom)
        elif isinstance(geom, shapely.geometry.MultiLineString):
            for line in geom:
                segments.append(line)
    return segments


def load_reduce_geomoriginal_file(filepath: Path):
    """Load the original geojson file."""
    return load_geojson(filepath)


def print_test():
    print("Test 2")
