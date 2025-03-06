"""Generate plots."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from matplotlib.lines import Line2D

import parse_geojson as pg


def try_to_polygonize(file: Path, outsize: int = 20) -> plt:
    """Display the actual geometry from the file.

    The results are display in 4 categories:
    - poly: The polygonal valid output
    - cut_edges: edges connected on both ends but not part of polygonal output
    - dangles: edges connected on one end but not part of polygonal output
    - invalid rings: polygons formed but which are not valid
    """
    assert Path.exists(file), "Invalid index for geojson file."

    gc = pg.load_GeometryCollection_from_geojson(file)
    poly, cut_edges, dangles, invalid = shapely.polygonize_full(gc.geoms)

    # Convert each variable to a GeoSeries
    poly_series = gpd.GeoSeries(poly)
    cut_edges_series = gpd.GeoSeries(cut_edges)
    dangles_series = gpd.GeoSeries(dangles)
    invalid_series = gpd.GeoSeries(invalid)

    # Plot each GeoSeries on the same plot with different colors
    fig, ax = plt.subplots(figsize=(outsize, outsize))
    if not poly.is_empty:
        poly_series.boundary.plot(
            ax=ax, color="darkblue", label="Polygon Boundaries"
        )
        poly_series.plot(ax=ax, color="blue", label="Polygons", alpha=0.2)
    if not cut_edges.is_empty:
        cut_edges_series.plot(ax=ax, color="orange", label="Cut Edges")
    if not dangles.is_empty:
        dangles_series.plot(ax=ax, color="green", label="Dangles", alpha=0.5)
    if not invalid.is_empty:
        invalid_series.plot(ax=ax, color="red", label="Invalid")

    # Create custom legend handles

    legend_handles = [
        Line2D([0], [0], color="blue", lw=4, label="Polygons"),
        Line2D([0], [0], color="orange", lw=4, label="Cut Edges"),
        Line2D([0], [0], color="green", lw=4, label="Dangles", alpha=0.5),
        Line2D([0], [0], color="red", lw=4, label="Invalid"),
    ]
    ax.legend(handles=legend_handles)
    ax.set_title(file.parent)
    plt.show()
    print(f"{poly_series.shape[0]} polygons detected")
