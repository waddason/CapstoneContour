"""
Parse the geojson files to extract the segments

@Version: 0.1
@Project: Capstone Vinci Contour Detection
@Date: 2025-01-25
@Author: Tristan Waddington (GitHub:waddason)
"""

import json
import shapely
from shapely.geometry import (
    shape,
    GeometryCollection,
    LineString,
    Polygon,
    MultiLineString,
)
import matplotlib.pyplot as plt
import geopandas as gpd

from pathlib import Path

OUTPUT_GEOJSON_INDENT = None
MAX_BAT_SIZE_IN_M = 1_000  # Buildings are never bigger than 1km


####################################################################################################
# Geojson parser
####################################################################################################
def _offset_reduce(geom, shift_x, shift_y, factor):
    """Translate and scale the geometry as numpy array.
    To be use with shapely.transform as lambda function."""
    geom[:, 0] = (geom[:, 0] - shift_x) * factor
    geom[:, 1] = (geom[:, 1] - shift_y) * factor
    return geom


def _load_GeometryCollection_from_geojson(filepath: Path) -> GeometryCollection:
    """Read a geojson file and return the geometry collection."""
    with open(filepath) as f:
        features = json.load(f)["features"]
    return GeometryCollection(
        [shape(feature["geometry"]) for feature in features]
    )


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

    return shapely.transform(
        geom_col, lambda x: _offset_reduce(x, min_x, min_y, factor)
    ), [min_x, min_y, factor]


def inverse_transform(geom_col, min_x, min_y, factor) -> GeometryCollection:
    """Inverse the transformation"""
    assert factor != 0, "Factor cannot be zero."
    return shapely.transform(
        geom_col, lambda x: _offset_reduce(x, -min_x, -min_y, 1 / factor)
    )


def load_geojson(
    filepath: Path,
) -> tuple[GeometryCollection, list[float, float, float]]:
    """Load the geojson file and return the geometry collection
    scaled to meters and translated to the origin.
    ==> Wrapper to use to load base output data."""
    geom_col = _load_GeometryCollection_from_geojson(filepath)
    return _offset_reduce_GeometryCollection(geom_col)


def load_GeometryCollection_from_geojson(filepath: Path) -> GeometryCollection:
    """Load the geojson file and return the geometry collection."""
    return _load_GeometryCollection_from_geojson(filepath)


def clean_geojson_to_segments_and_save(filepath: Path, output_filepath: Path):
    """Load the geojson file, offset the map and reduce to meters, then
    extract the segments and save the segments in a new geojson file."""
    print(f"Cleaning {filepath} to {output_filepath}")
    geom_col, transform_parameters = load_geojson(filepath)
    segments_list = extract_segments(geom_col)

    save_as_geojson(
        GeometryCollection(segments_list), output_filepath, transform_parameters
    )


def load_segments(filepath: Path) -> tuple[GeometryCollection, tuple[float,]]:
    """Load the clean segments, offset and reduce to meters.
    This file contains only one feature: a Geometry collection
    Also returns the transformation parameters."""
    with open(filepath) as f:
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


####################################################################################################
# Geometry to segments
####################################################################################################
# TODO: Try to vectorize the operations
# TODO: Create a graph from the points


def _extract_segments_from_LineString(ls: LineString) -> list[LineString]:
    """Extract the segments from a LineString."""
    segments = []
    # print(f"Extracting segments from {ls}")
    for start, end in zip(ls.coords[:-1], ls.coords[1:]):
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


def extract_segments(geom_col: GeometryCollection) -> list[LineString]:
    """Extract the segments from the geometry collection."""
    # print(f"Extracting segments from {geom_col}")
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
            raise ValueError(f"Geometry type {type(geom)} not supported.")

    # Remove duplicates
    print(f"\t{len(segments)} segments extracted")
    segments = list(set(segments))
    print(f"\t{len(segments)} unique segments")

    return segments


####################################################################################################
# Output
####################################################################################################
def save_as_geojson(
    geom_col: GeometryCollection,
    filepath: Path,
    transform_parameters: tuple[float, float, float] = None,
):
    """Save the geometry collection as a geojson file.
    Parameters:
    -----------
    geom_col: GeometryCollection
        The geometry collection to save.
    transform_parameters: tuple[float, float, float] (min_x, min_y, factor)
        The parameters used by _offset_reduce_GeometryCollection to transform
        the geometry for future inverse transform.
    filepath: Path
        The path to save the geojson file."""
    # Include the GeometryCollection in a FeatureCollection to mimic the output
    # from AutoCAD
    header = '{"type":"FeatureCollection",'
    # Add the transform parameters if any
    if transform_parameters:
        header += f'"transform_parameters":{transform_parameters},'
    header += '"features": ['
    footer = "]}"
    with open(filepath, "w") as f:
        geojson_data = shapely.to_geojson(geom_col, OUTPUT_GEOJSON_INDENT)
        f.write(header)
        f.write(geojson_data)
        f.write(footer)


def plot_GeometryCollection(
    geom_col: GeometryCollection, figsize=(30, 30), path_to_save=None,**kwargs
):
    """Plot the geometry collection."""
    gs = gpd.GeoSeries(geom_col)
    fig, ax = plt.subplots(figsize=figsize)
    gs.plot(ax=ax, edgecolor="black", **kwargs)
    if path_to_save is not None:
        ax.axis('off')
        fig.savefig("Test_image/outpout0.png", bbox_inches='tight', dpi=30)
        fig.show()
    fig.show()


####################################################################################################
# Main
####################################################################################################
if __name__ == "__main__":
    """Clean the geojson file to extract the segments."""
    data_dir = Path("data", "geojson")
    output_dir = Path("data", "clean")
    for filepath in data_dir.iterdir():
        out_file = output_dir / f"{filepath.stem}_clean.geojson"
        clean_geojson_to_segments_and_save(filepath, out_file)


"""
TODO LIST
- [ ]  check factor for Output9 -> but may be not the same scale
- [X]  use shapely points to use geometrical operations
- [X]  preserve the inverse transform.
- [ ]  try to vectorize the operations
- [ ]  create the geojson from the polygons
- [A] create the polygons from the geojson
"""
