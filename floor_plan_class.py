import geopandas
import pandas as pd
import matplotlib.pyplot as plt

from utils.parse_geojson import plot_GeometryCollection

class FloorPlan:
    def __init__(self, path, name=None):
        self.path = path
        self.floor_plan = geopandas.read_file(path)
        self.name = path.split("/")[-1].split(".")[0] if name is None else name
        self.segments = None
        self.transform_parameters = None

    def plan_preprocessing(self):
        return None

    def plot(self):
        fig, ax = plt.subplots()
        self.floor_plan.plot(ax, aspect=1)
        plt.show()

    def clean_plot(self):
        plot_GeometryCollection(self.segments, figsize=(100, 100))
        plt.show()