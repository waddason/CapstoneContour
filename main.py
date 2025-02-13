import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiLineString
from pathlib import Path

path = Path("data/clean", "Output2.geojson")

gdf = gpd.GeoDataFrame.from_file(path)

gdf.plot(aspect=1)
# gdf.boundary.plot(aspect=1)
# gdf.boundary.buffer(100).plot(aspect=1)
