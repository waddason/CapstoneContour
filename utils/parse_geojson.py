"""Parse the geojson files to extract the segments.

Manipulate GeometryCollection from shapely
@Version: 0.2
@Project: Capstone Vinci Contour Detection
@Date: 2025-03-07
@Author: Tristan Waddington (GitHub:waddason)
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    Polygon,
    shape,
)

OUTPUT_GEOJSON_INDENT = None
MAX_BAT_SIZE_IN_M = 1_000  # Buildings are never bigger than 1km


###############################################################################
# Input: Geojson to GeometryCollection
###############################################################################
def load_geometrycollection_from_geojson(filepath: Path) -> GeometryCollection:
    """Load the geojson file and return the geometry collection."""
    with Path.open(filepath) as f:
        features = json.load(f)["features"]
    return GeometryCollection(
        [shape(feature["geometry"]) for feature in features],
    )


def load_geojson(
    filepath: Path,
) -> tuple[GeometryCollection, list[float, float, float]]:
    """Load and scale the geojson file.

    Returns the geometry collection scaled to meters and translated to
    start at the origin.
    ==> **Wrapper to use to load base output data.**

    Args:
    ----
    filepath: Path
        the path to the GeoJson file to load.

    Returns:
    -------
    geometry_collection: shapely.GeometryCollection
        the content as Shapely Geometries. Access throug .geoms.
    transform_parameters: list[float, float, float]
        the paramters used to normalized the geometries:
        [shift_x, shif_y, scale]

    """
    geom_col = load_geometrycollection_from_geojson(filepath)
    return _offset_reduce_GeometryCollection(geom_col)


def get_segments(filepath: Path) -> GeometryCollection:
    """Extract the segments from a raw GeoJson."""
    gc = load_geometrycollection_from_geojson(filepath)
    return shapely.GeometryCollection(extract_segments(gc))


def load_segments(filepath: Path) -> tuple[GeometryCollection, tuple[float,]]:
    """Load the clean segments, offset and reduce to meters.

    This file contains only one feature: a Geometry collection
    Also returns the transformation parameters.
    """
    with Path.open(filepath) as f:
        json_dic = json.load(f)
        transform_parameters = json_dic["transform_parameters"]
    return (
        GeometryCollection(
            [
                shape(feature)
                for feature in json_dic["features"][0]["geometries"]
            ]
        ),
        transform_parameters,
    )


###############################################################################
# Processing: GeometryCollection to GeometryCollection
###############################################################################


# -----------------------------------------------------------------------------
# Whole GeometryCollection transformation
# -----------------------------------------------------------------------------
def transform_gc(
    geom_col: GeometryCollection,
    shift_x: float,
    shift_y: float,
    factor: float,
) -> GeometryCollection:
    """Translate and scale the GeometryCollection.

    Args:
    ----
    geom_col: GeometryCollection
        The geometry collection to be transformed.
    shift_x: float
        The translation value for x
    shift_y: float
        The translation value for y
    factor: float
        The scaling factor.

    Returns:
    -------
    GeometryCollection
        The transformed geometry collection.

    """
    return shapely.transform(
        geom_col,
        lambda x: _offset_reduce(
            geom=x,
            shift_x=shift_x,
            shift_y=shift_y,
            factor=factor,
        ),
    )


def inverse_transform_gc(
    geom_col: GeometryCollection,
    shift_x: float,
    shift_y: float,
    factor: float,
) -> GeometryCollection:
    """Translate and scale back the GeometryCollection.

    Args:
    ----
    geom_col: GeometryCollection
        The geometry collection to be transformed.
    shift_x: float
        The **orignial** translation value for x
    shift_y: float
        The **original** translation value for y
    factor: float
        The **orignial** scaling factor.

    Returns:
    -------
    GeometryCollection
        The transformed geometry collection back to orignial coordinates.

    """
    """Inverse the transformation."""
    if factor == 0:
        msg = "inverse_transfor_gc: Factor cannot be zero."
        raise ValueError(msg)
    return shapely.transform(
        geom_col,
        lambda x: _offset_reduce(
            geom=x,
            shift_x=-shift_x * factor,
            shift_y=-shift_y * factor,
            factor=1 / factor,
        ),
    )


# Helper functions
def _offset_reduce(geom, shift_x, shift_y, factor) -> shapely.Geometry:
    """Translate and scale the geometry as numpy array.

    To be used with shapely.transform as lambda function.
    """
    geom[:, 0] = (geom[:, 0] - shift_x) * factor
    geom[:, 1] = (geom[:, 1] - shift_y) * factor
    return geom


def _offset_reduce_GeometryCollection(
    geom_col: GeometryCollection,
) -> tuple[GeometryCollection, list[float, float, float]]:
    """Return the geometry collection
    that is translated to the origin and scaled to meters,
    along with parameters needed for the inverse transform.
    """
    # Compute the factor correction
    min_x, min_y, max_x, max_y = geom_col.bounds

    # Modify the factor to work in meters
    if max_x - min_x > MAX_BAT_SIZE_IN_M or max_y - min_y > MAX_BAT_SIZE_IN_M:
        factor = 1 / 1_000
    else:
        factor = 1

    return (
        transform_gc(geom_col, min_x, min_y, factor),
        [min_x, min_y, factor],
    )


# -----------------------------------------------------------------------------
# Geometry to segments
# -----------------------------------------------------------------------------
def extract_segments(geom_col: GeometryCollection) -> list[LineString]:
    """Extract the segments from the geometry collection.

    Return:
    ------
    segments: list[shapely.LineString]
        The list of all segments as a shapely LineString (pt1, pt2)
        where pt1 < pt2

    """
    segments = []
    # Extract the segments from the geometry collection
    for geom in geom_col.geoms:
        if isinstance(geom, MultiLineString):
            segments.extend(_extract_segments_from_MultiLineString(geom))
        elif isinstance(geom, Polygon):
            segments.extend(_extract_segments_from_Polygon(geom))
        elif isinstance(geom, LineString):
            segments.extend(_extract_segments_from_LineString(geom))
        else:
            msg = f"Geometry type {type(geom)} not supported."
            raise TypeError(msg)

    # Remove duplicates
    return list(set(segments))


def _sort_points(p1, p2):
    """Sort the points for consistency."""
    return (p1, p2) if p1 <= p2 else (p2, p1)


def _extract_segments_from_LineString(ls: LineString) -> list[LineString]:
    """Extract the segments from a LineString."""
    segments = []
    for p1, p2 in zip(ls.coords[:-1], ls.coords[1:]):
        start, end = _sort_points(p1, p2)
        segments.append(LineString([start, end]))
    return segments


def _extract_segments_from_MultiLineString(
    mls: MultiLineString,
) -> list[LineString]:
    """Extract the segments from a MultiLineString."""
    # print(f"Extracting segments from {mls}")
    segments = []
    # Each geometry contains a list of list of Points
    for line_string in mls.geoms:
        segments.extend(
            _extract_segments_from_LineString(LineString(line_string))
        )
    return segments


def _extract_segments_from_Polygon(poly: Polygon) -> list[LineString]:
    """Extract the segments from a Polygon."""
    # print(f"Extracting segments from {poly}")
    segments = []
    linear_ring = poly.exterior
    segments.extend(_extract_segments_from_LineString(linear_ring))
    for interior in poly.interiors:
        segments.extend(_extract_segments_from_LineString(interior))
    return segments


###############################################################################
# Output: GeometryCollection to geojson
###############################################################################
def save_as_geojson(
    geom_col: GeometryCollection,
    filepath: Path,
    transform_parameters: tuple[float, float, float] = None,
) -> None:
    """Save the geometry collection as a geojson file.

    Args:
    ----
    geom_col: GeometryCollection
        The geometry collection to save.
    transform_parameters: tuple[float, float, float] (min_x, min_y, factor)
        The parameters used by _offset_reduce_GeometryCollection to transform
        the geometry for future inverse transform.
        DEPRECATED
    filepath: Path
        The path to save the geojson file.

    """
    # Include the GeometryCollection in a FeatureCollection to mimic the output
    # from AutoCAD
    header = '{"type":"FeatureCollection",'
    # Add the transform parameters if any
    if transform_parameters:
        header += f'"transform_parameters":{transform_parameters},'
    header += '"features": ['
    footer = "]}"
    with Path.open(filepath, "w") as f:
        geojson_data = shapely.to_geojson(geom_col, OUTPUT_GEOJSON_INDENT)
        f.write(header)
        f.write(geojson_data)
        f.write(footer)


def plot_GeometryCollection(
    geom_col: GeometryCollection,
    figsize: tuple[float, float] = (30, 30),
    **kwargs,
):
    """Plot the geometry collection."""
    gs = gpd.GeoSeries(geom_col)
    fig, ax = plt.subplots(figsize=figsize)
    gs.plot(ax=ax, **kwargs)
    plt.show()


def export_to_geojson(geom_col: GeometryCollection, filepath: Path) -> None:
    """Save the GeometryCollection as a GeoJson file.

    Args:
    ----
    geom_col: GeometryCollection
        The geometry collection to save.
    filepath: Path
        The path to save the geojson file.

    """
    # Include the GeometryCollection in a FeatureCollection to mimic the output
    # from AutoCAD
    header = '{"type":"FeatureCollection","features": ['
    footer = "]}"
    geojson_data = ""
    feature_head = '{"type": "Feature", "geometry":'
    feature_tail = ',"properties":{}},'
    for feature in geom_col.geoms:
        geojson_data += (
            feature_head
            + shapely.to_geojson(feature, OUTPUT_GEOJSON_INDENT)
            + feature_tail
        )
    # Write file on disk
    with Path.open(filepath, "w") as f:
        f.write(header)
        f.write(geojson_data[:-1])  # Remove trailing comma
        f.write(footer)


###############################################################################
# Pipes: GeoJson to GeoJson
###############################################################################
def clean_geojson_to_segments_and_save(
    filepath: Path,
    output_filepath: Path,
) -> None:
    """Compute segments from shapes in GeoJson.

    Load the geojson file, offset the map and reduce to meters, then
    extract the segments and save the segments in a new geojson file.
    """
    print(f"Cleaning {filepath} to {output_filepath}")
    geom_col, transform_parameters = load_geojson(filepath)
    segments_list = extract_segments(geom_col)

    save_as_geojson(
        GeometryCollection(segments_list), output_filepath, transform_parameters
    )


###############################################################################
# Spaces and walls files
###############################################################################
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


def segment_to_rectangle(p1, p2, width):
    """Convert a line segment (p1 to p2) into a rectangle of given width.

    Returns:
    -------
    Polygon: A Shapely Polygon representing the rectangle.

    """
    p1, p2 = np.array(p1), np.array(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return None  # Skip zero-length segments

    direction /= length  # Normalize direction vector
    normal = np.array([-direction[1], direction[0]])  # Perpendicular vector
    offset = (width / 2) * normal  # Half-width offset

    # Compute four corners
    corner1, corner2 = p1 + offset, p1 - offset
    corner3, corner4 = p2 - offset, p2 + offset

    return Polygon([corner1, corner2, corner3, corner4, corner1])


def open_spaces_walls_in_folder_and_save_svg(folder_path: Path):
    """Open the tuple of geojson files Walls and Spaces in a single folder."""
    walls_name = "Walls.geojson"
    spaces_name = "Spaces.geojson"
    walls_gdf = gpd.read_file(folder_path / walls_name)
    walls_gdf.geometry = walls_gdf.apply(
        lambda x: segment_to_rectangle(
            np.array(x.geometry.coords[0]),
            np.array(x.geometry.coords[1]),
            x["Width"],
        ),
        axis=1,
    )
    spaces_gdf = gpd.read_file(folder_path / spaces_name)

    ax = walls_gdf.plot(lw=0.01, zorder=3)
    ax.set_aspect("equal")
    spaces_gdf.plot(color="red", ax=ax, zorder=4, alpha=0.8)

    plt.savefig(folder_path / "plot.svg", format="svg")
