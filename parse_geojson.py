"""
Parse the geojson files to extract the segments
"""

import json

from pathlib import Path
from dataclasses import dataclass

####################################################################################################
# Classes
# Implement the ideas from  M. Schäfer, C. Knapp, and S. Chakraborty, “Automatic
# generation of topological indoor maps for real-time map-
# based localization and tracking,” in Proceedings of Indoor
# Positioning and Indoor Navigation (IPIN), 2011 International
# Conference, pp. 1–8, IEEE, Guimaraes, Portugal, September 2011.
#
####################################################################################################


class Point:
    """The point class"""

    closeness_threshold: float = 1e-6

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (
            abs(self.x - other.x) < self.closeness_threshold
            and abs(self.y - other.y) < self.closeness_threshold
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __le__(self, other):
        return self.x < other.x or (self.x == other.x and self.y <= other.y)

    def __gt__(self, other):
        return self.x > other.x or (self.x == other.x and self.y > other.y)

    def __ge__(self, other):
        return self.x > other.x or (self.x == other.x and self.y >= other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, c: float):
        return Point(self.x * c, self.y * c)

    def __truediv__(self, c: float):
        return Point(self.x / c, self.y / c)

    def __neg__(self):
        return Point(-self.x, -self.y)


class Segment:
    """The segment class"""

    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def __str__(self):
        return f"Segment({self.start}, {self.end})"

    def __repr__(self):
        return f"Segment({self.start}, {self.end})"

    def plotable(self):
        """Matplotlib plotable format"""
        return [self.start.x, self.end.x], [self.start.y, self.end.y]


class Line:
    """A multipoints line class"""

    def __init__(self, points: list[list[float]]):
        self.points = points

    def from_list(self, points: list[list[float]]):
        self.points = points


####################################################################################################
# Geojson parser
####################################################################################################
def read_box_from_file(filepath: Path) -> tuple[float, float, float, float]:
    """Read the min and max coordinates from the geojson file.
    Returns:
    --------
    tuple[float, float, float, float]: (min_x, min_y, max_x, max_y)
    """
    with open(filepath) as f:
        data = json.load(f)
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for feature in data["features"]:
        multi_line = feature["geometry"]["coordinates"]
        for line in multi_line:
            for x, y in line:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    print(f"{filepath}: {min_x:.3f}, {min_y:.3f}, {max_x:.3f}, {max_y:.3f}")
    return min_x, min_y, max_x, max_y


def compute_factor_correction(filepath: Path) -> tuple[float, float]:
    """Compute the factor correction for the coordinates.
    Returns:
    --------
    x_shiff: the shift to center x coordiantes
    y_shift: the shift to center y coordiantes
    scale: the factor correction to work in meters
    """
    min_x, min_y, max_x, max_y = read_box_from_file(filepath)

    factor = 1.0
    MAX_BAT_SIZE_IN_M = 1_000  # Buildings are never bigger than 1km
    # Modify the factor to work in meters
    if max_x - min_x > MAX_BAT_SIZE_IN_M or max_y - min_y > MAX_BAT_SIZE_IN_M:
        factor = 1 / 1_000  # Transform back to meters

    # Translate the coordinates
    x_shift = min_x
    y_shift = min_y

    return x_shift, y_shift, factor


def parse_segments_geojson(filepath: Path) -> list[Segment]:
    """Parse the geojson file to extract the segments.
    Returns:
    --------
    list[Segment]: the list of segments
    """
    # First read to check the factor correction
    x_shift, y_shift, factor = compute_factor_correction(filepath)
    shift_point = Point(x_shift, y_shift)

    # Second read to actually extract the segments
    with open(filepath) as f:
        data = json.load(f)

    segments = []
    for feature in data["features"]:
        multi_line = feature["geometry"]["coordinates"]
        for line in multi_line:
            # Center the coordinates
            line_center = [
                (Point(float(x), float(y)) - shift_point) * factor
                for x, y in line
            ]

            # Create the segments
            for start, end in zip(line_center[:-1], line_center[1:]):
                seg_candidate = Segment(start, end)
                # Avoid duplicates
                if seg_candidate not in segments:
                    segments.append(Segment(start, end))

    print(f"{filepath}: {len(segments)} segments")
    return segments


# def parse_segments_geojson_no_center(filepath: Path) -> list[Segment]:
#     """Parse the geojson file to extract the segments.
#     Returns:
#     --------
#     list[Segment]: the list of segments
#     """
#     # Second read to actually extract the segments
#     with open(filepath) as f:
#         data = json.load(f)

#     segments = []
#     for feature in data["features"]:
#         multi_line = feature["geometry"]["coordinates"]
#         for line in multi_line:
#             # Create the segments
#             for start, end in zip(line[:-1], line[1:]):
#                 seg_candidate = Segment(Point(*start), Point(*end))
#                 # Avoid duplicates
#                 if seg_candidate not in segments:
#                     segments.append(seg_candidate)
#         break
#     print(f"{filepath}: {len(segments)} segments")
#     return segments


# [ ] TODO: check factor for Output9 -> but may be not the same scale
# [ ] TODO: use shapely points to use geometrical operations
# [ ] TODO: preserve the inverse transform.

# [ ] TODO: create the geojoon from the polygons
# [ ] TODO: create the polygons from the geojson