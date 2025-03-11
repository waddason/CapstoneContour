"""Definition of model metrics.

This module contains the definitions of the metrics that will be used in
models comparisons.
"""

import time
from itertools import product
from pathlib import Path

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment

import utils.parse_geojson as pg


###############################################################################
# vector geometry
# Intersection over Union IoU
def intersection_over_union(
    geom1: shapely.Polygon, geom2: shapely.Polygon
) -> float:
    """Calculate the IoU of 2 shapely objects."""
    # Catch GEOSException: TopologyException:
    try:
        intersection = geom1.intersection(geom2)
        union_area = geom1.union(geom2).area
    except:
        return 0.0
    return intersection.area / union_area if union_area else 0.0


def average_iou(
    geoms_true: shapely.GeometryCollection.geoms,
    geoms_pred: shapely.GeometryCollection.geoms,
    *,
    with_unmatched_penality: bool = False,
) -> float:
    """Calculate the average IoU for all pairs of polygons."""
    ious = [
        intersection_over_union(gt, pred)
        for gt, pred in product(geoms_true, geoms_pred)
    ]
    if with_unmatched_penality:
        # Include penalties for unmatched ground truth and predicted polygons
        matched = len([iou for iou in ious if iou > 0])
        unmatched_true = len(geoms_true) - matched
        unmatched_pred = len(geoms_pred) - matched
        total_loss = 1 - np.nanmean(ious) + unmatched_true + unmatched_pred
    else:
        total_loss = np.nanmean(ious)
    return total_loss


# -----------------------------------------------------------------------------
# Loss function
def jaccard_loss(
    geom1: shapely.Polygon,
    geom2: shapely.Polygon,
    smooth: float = 1e-6,
) -> float:
    """Calculate loss using the IoU of 2 shapely objects."""
    # Catch GEOSException: TopologyException:
    try:
        intersection = geom1.intersection(geom2)
        union_area = geom1.union(geom2).area
    except:
        return None

    return 1 - (intersection.area + smooth) / (union_area + smooth)


# -----------------------------------------------------------------------------
# Matched Intersection ovec Union
def compute_iou_matrix(
    geoms_true: shapely.GeometryCollection.geoms,
    geoms_pred: shapely.GeometryCollection.geoms,
) -> float:
    """Compute the IoU matrix for ground truth and predicted polygons."""
    iou_matrix = np.zeros((len(geoms_true), len(geoms_pred)))
    for i, gt in enumerate(geoms_true):
        for j, pred in enumerate(geoms_pred):
            iou_matrix[i, j] = intersection_over_union(gt, pred)
    return iou_matrix


def match_polygons(
    geoms_true: shapely.GeometryCollection.geoms,
    geoms_pred: shapely.GeometryCollection.geoms,
) -> float:
    """Match ground truth to predicted polygons using the Hungarian algo."""
    iou_matrix = compute_iou_matrix(geoms_true, geoms_pred)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU
    return row_ind, col_ind, iou_matrix


def matched_iou(
    geoms_true: shapely.GeometryCollection.geoms,
    geoms_pred: shapely.GeometryCollection.geoms,
) -> np.ndarray[float]:
    """Calculate IoU per ground truth on matched predicted poly.

    Return:
    ------
    score per ground truth polygon.

    """
    row_ind, col_ind, iou_matrix = match_polygons(geoms_true, geoms_pred)
    return iou_matrix[row_ind, col_ind]


def average_matched_iou(
    geoms_true: shapely.GeometryCollection.geoms,
    geoms_pred: shapely.GeometryCollection.geoms,
) -> float:
    """Calculate average IoU for matched ground truth and predicted poly."""
    # row_ind, col_ind, iou_matrix = match_polygons(geoms_true, geoms_pred)
    # matched_ious = iou_matrix[row_ind, col_ind]
    # return np.nanmean(matched_ious)
    return np.nanmean(matched_iou(geoms_true, geoms_pred))


###############################################################################
# Computer vision
# mAP mean Average Precision
def average_precision():
    # need a prediction
    pass


def mean_ap() -> float:
    """Compute mean Average Precision score (mAP)."""
    return 0.0


###############################################################################
# Wrapper scorer
###############################################################################
def score_model(model: callable, data_folder: Path, metric: callable) -> float:
    """Comptute the model score on a dataset.

    Act as a dataloader
    model returns a geometry collection
    datafolder contains a Walls and Spaces geojson file per subfolder
    metric is the name of metric to use from ["average_iou",
        "average_matched_iou"] default to "average_matched_iou"
    """
    x_file = "Walls.geojson"
    ground_truth_file = "Spaces.geojson"
    start_time = time.time()
    scores = []
    for subfolder in data_folder.glob("**/"):
        if (
            subfolder.is_dir()
            and (subfolder / x_file).exists()
            and (subfolder / ground_truth_file).exists()
        ):
            # Load the geometries
            x = pg.load_geometrycollection_from_geojson(subfolder / x_file)
            geoms_true = pg.load_geometrycollection_from_geojson(
                subfolder / ground_truth_file,
            )
            # Compute prediction
            try:
                geoms_pred = model(x)
            except ValueError as e:
                print(f"ValueError in folder {subfolder}: {e}")
                continue

            # Compute the metric
            if geoms_pred:
                scores.append(metric(geoms_true.geoms, geoms_pred.geoms))

    total_score = np.nanmean(scores)
    # Display the result
    print(
        f"Score: {total_score:.3f} for {model.__name__} on folder "
        f"{data_folder} in {time.time() - start_time:.2f}s with "
        f"{metric.__name__}",
    )
    return total_score


def test_model_on_sample(
    model: callable, sample_folder: Path, metric: callable
) -> float:
    """Comptute the model score on a sample.

    model returns a geometry collection
    **sample contains a Walls and Spaces geojson file per subfolder**
    metric is the name of metric to use from ["average_iou",
        "average_matched_iou"] default to "average_matched_iou"
    """
    x_file = "Walls.geojson"
    ground_truth_file = "Spaces.geojson"
    if (
        not (sample_folder / x_file).exists
        or not (sample_folder / ground_truth_file).exists
    ):
        msg = (
            f"Sample_folder {sample_folder} should contain a Walls.geojson"
            f" and a Spaces.geojson files."
        )
        raise ValueError(msg)

    # Load the geometries
    x = pg.load_geometrycollection_from_geojson(sample_folder / x_file)
    geoms_true = pg.load_geometrycollection_from_geojson(
        sample_folder / ground_truth_file,
    )
    # Compute prediction
    try:
        geoms_pred = model(x)
    except ValueError as e:
        print(f"ValueError in folder {sample_folder}: {e}")

    # Compute the metric
    return metric(geoms_true.geoms, geoms_pred.geoms) if geoms_pred else None
