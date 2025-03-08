"""Generate plots."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from matplotlib.lines import Line2D

import utils.parse_geojson as pg


def try_to_polygonize(file: Path, outsize: int = 20) -> plt:
    """Display the actual geometry from the file.

    The results are display in 4 categories:
    - poly: The polygonal valid output
    - cut_edges: edges connected on both ends but not part of polygonal output
    - dangles: edges connected on one end but not part of polygonal output
    - invalid rings: polygons formed but which are not valid
    """
    assert Path.exists(file), "Invalid index for geojson file."

    gc = pg.load_geometrycollection_from_geojson(file)
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


def plot_score(model: callable, sample_folder: Path, metric: callable) -> plt:
    """Plot the score on a specific sample.

    Args:
    ----
    model: a instance of a geometry collection prediction model.
        The model that will make the prediction on the given files.
    sample_folder: Path
        the folder containing Spaces and Walls GeoJson files.
    metric: callable

    """
    fig, axs = plt.subplots(1, 3, layout="constrained")

    # Walls
    x = pg.load_geometrycollection_from_geojson(
        sample_folder / "Walls.geojson",
    )
    x_df = gpd.GeoDataFrame(
        x.geoms,
        columns=["geometry"],
    ).reset_index()
    x_df.plot(ax=axs[0], alpha=0.5, column="index", edgecolor="black")
    axs[0].set_title("Walls")

    # Predictions
    y_pred = model(x)

    y_pred_df = gpd.GeoDataFrame(
        y_pred.geoms,
        columns=["geometry"],
    ).reset_index()
    y_pred_df.plot(ax=axs[1], alpha=0.5, column="index", edgecolor="black")
    axs[1].set_title("Prediction")

    # Ground Truth
    axs[2].set_title("Ground truth")
    geoms_true = pg.load_geometrycollection_from_geojson(
        sample_folder / "Spaces.geojson",
    )
    gpd.GeoSeries(geoms_true).plot(ax=axs[2], alpha=0.5)

    # Ensure the same bounds
    axs[1].set_xlim(axs[2].get_xlim())
    axs[1].set_ylim(axs[2].get_ylim())
    axs[0].set_xlim(axs[2].get_xlim())
    axs[0].set_ylim(axs[2].get_ylim())
    # Display the score in the title
    score = metric(geoms_true.geoms, y_pred.geoms)
    plt.suptitle(
        f"Prediction score: {score:.3f}, {len(y_pred.geoms)} rooms found."
    )
    return plt
