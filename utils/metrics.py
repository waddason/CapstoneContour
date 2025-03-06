"""Definition of model metrics.

This module contains the definitions of the metrics that will be used in
models comparisons.
"""

from itertools import product

import numpy as np
import shapely


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
    geoms_true: shapely.GeometryCollection,
    geoms_pred: shapely.GeometryCollection,
) -> float:
    """Calculate the average IoU for all pairs of polygons."""
    ious = [
        intersection_over_union(gt, pred)
        for gt, pred in product(geoms_true, geoms_pred)
    ]
    return np.nanmean(ious)
