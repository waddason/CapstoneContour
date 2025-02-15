from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from floor_plan_class import FloorPlan

import utils.segment_geometry as sg
import utils.parse_geojson as pg


# Define the woring directory and filename
data_dir = Path("data", "clean")
filename = "Output0_clean.geojson"
segments, transform_parameters = pg.load_segments(data_dir / filename)
pg.plot_GeometryCollection(segments, figsize=(100, 100), path_to_save="Test_image/output0.png")
