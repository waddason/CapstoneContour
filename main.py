import json
from pathlib import Path
from importlib import reload
import shapely

# own imports
import utils.segment_geometry as sg
import utils.parse_geojson as pg


data_dir = Path("data", "clean")

for i in range(9):
    filename = f"Output{i}_clean.geojson"
    segments0, transform_parameters0 = pg.load_segments(data_dir / filename)
    path_to_save = f"test_images/samples/Output{i}_clean.png"
    pg.plot_GeometryCollection(
        segments0, color="k", linewidth=1.5, alpha=0.7, path_to_save=path_to_save
    )