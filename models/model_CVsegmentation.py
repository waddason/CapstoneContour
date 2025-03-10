"""Parse the geojson function to detect the contours of the pieces.

@Version: 0.2
@Project: Capstone Vinci Contour Detection
@Date: 2025-03-06
@Author: Fabien Lagnieu, Tristan Waddington
"""

import os
from pathlib import Path

import cv2
import numpy as np
import shapely
from PIL import Image

# own imports
import utils.parse_geojson as pg
from utils.preprocess_segments import complete_preprocessing


###############################################################################
# @Fabien : Genrate binary image from segments
###############################################################################
class CVSegmentation:
    """Find rooms in GeometryCollection using CV segmentation."""

    def __init__(
        self,
        dpi: int = 50,
        thickness: int = 7,
        dilatation_method: str = "gaussian",
        surf_min: int = 1,
        surf_max: int = 5_000,
        clean_segements: bool = False,
    ) -> "CVSegmentation":
        """Find rooms in GeometryCollection using CV segmentation.

        Args:
        ----
        dpi: int
            number of pixel to represent 1m in image. Default to 50.
        thickness: int
            number of pixels to draw walls and compute dilatation. Will be
            rounded to the next odd integer.
            Default to 7.
        dilatation_method: str
            name of the dilatation method to use. Choose from ["gaussian",
            "ellipse", "cross"]. Default to "gaussian", as more efficient on
            curves.
        surf_min: int
            The minimum square meter of the detected rooms. Deflaut to 1m^2.
        surf_max: int
            The maximum square meter of the detected rooms. Deflaut to 5000m^2.

        """
        # Paramètres pour la génération des images
        self.dpi = dpi  # Changer la résolution ici (ex: 30, 50, 100...)
        # Épaisseur des lignes en pixels (impair de préférence)
        self.thickness = thickness if thickness % 2 else thickness + 1
        self.dilation_method = dilatation_method
        # Transform the surfaces into number of pixels
        self.surf_min = surf_min * self.dpi * self.dpi
        self.surf_max = surf_max * self.dpi * self.dpi
        self.clean_segments = clean_segements

        # Class constants
        self.__name__ = (
            f"CVSegmentation(dpi:{self.dpi}, thickness:"
            f"{self.thickness}, dilatation:{self.dilation_method})"
        )
        self.BLACK: int = 0  # cv2 color
        self.kernels_dic = {
            "ellipse": cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (thickness, thickness),
            ),
            "cross": cv2.getStructuringElement(
                cv2.MORPH_CROSS,
                (thickness, thickness),
            ),
        }

    def predict(
        self,
        geometry_collection: shapely.GeometryCollection,
        draw_image: bool = False,
    ) -> shapely.GeometryCollection:
        """Find rooms from geometry collection."""
        # Get the segments from the geometry collection and normalize
        geometry_collection, transform_parameters = (
            pg._offset_reduce_GeometryCollection(geometry_collection)
        )
        self.segments: list[shapely.LineString] = pg.extract_segments(
            geometry_collection,
        )
        if self.clean_segments:
            self.segments = complete_preprocessing(self.segments)

        # Create binary image
        self.binary: np.ndarray = self.generate_binary_image()

        # Contour detection
        self.contours: list[np.ndarray] = self.find_contours()

        # Draw on image
        self.rooms: np.ndarray = self.draw_contours() if draw_image else None

        # create the polygons
        self.polygons: shapely.GeometryCollection = (
            self.polygons_from_contours()
        )
        # Scale back the polygons
        # Back from the dpi scaling
        self.polygons = pg.inverse_transform_gc(
            self.polygons,
            *self.scale_parameters,
        )
        # Back from the loading process
        self.polygons = pg.inverse_transform_gc(
            self.polygons,
            *transform_parameters,
        )

        return self.polygons

    def __call__(
        self,
        x: shapely.GeometryCollection,
    ) -> shapely.GeometryCollection:
        """Call predict."""
        if type(x) is not shapely.GeometryCollection:
            msg = "This model handles only shapely.GeometryCollection."
            raise ValueError(msg)
        return self.predict(x, draw_image=False)

    def generate_binary_image(self) -> None:
        """Create binary image from segments."""
        if not self.segments:
            msg = "No segments found to generate binary image."
            raise ValueError(msg)
        # Bounds of image
        minx, miny, maxx, maxy = shapely.geometrycollections(
            self.segments,
        ).bounds
        width = int((maxx - minx) * self.dpi) + 1
        height = int((maxy - miny) * self.dpi) + 1

        # Scale the segments at the desired scale
        self.scale_parameters = [minx, miny, self.dpi]
        scale_segments = pg.transform_gc(self.segments, *self.scale_parameters)
        # White background image
        img = np.ones((height, width), dtype=np.uint8) * 255
        # Trace segments on image
        all_cords = shapely.get_coordinates(scale_segments).astype(int)
        for pt1, pt2 in zip(all_cords[:-1:2], all_cords[1::2]):
            cv2.line(img, pt1, pt2, self.BLACK, self.thickness)

        # Rise the thickness
        if self.thickness > 1:
            if self.dilation_method == "gaussian":
                img = cv2.GaussianBlur(img, (self.thickness, self.thickness), 0)
                # Confirm threshold 128 or 200 ?
                _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            else:
                kernel = self.kernels_dic.get(
                    self.dilation_method,
                    np.ones((self.thickness, self.thickness), np.uint8),
                )
                img = cv2.dilate(img, kernel, iterations=1)
        # Return binary image
        return img

    def find_contours(self) -> list:
        """Find and create polygons."""
        # image preprocessing
        # 1. morphology closing
        kernel = np.ones((5, 5), np.uint8)
        closed_morph = cv2.morphologyEx(
            self.binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        # TODO: brenchmark
        # 2. contour detections
        # retrievalModes
        # -  cv.RETR_EXTERNAL: retrieves only the extreme outer contours.
        #   It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
        # - cv.RETR_LIST: retrieves all of the contours without establishing
        #   any hierarchical relationships.
        # - cv.RETR_CCOMP: retrieves all of the contours and organizes them
        #   into a two-level hierarchy. At the top level, there are external
        #   boundaries of the components. At the second level, there are
        #   boundaries of the holes. If there is another contour inside a hole
        #   of a connected component, it is still put at the top level.
        # - cv.RETR_TREE: retrieves all of the contours and reconstructs a full
        #   hierarchy of nested contours.
        # - cv.RETR_FLOODFILL

        # ContourApproximationModes
        # - cv.CHAIN_APPROX_NONE: stores absolutely all the contour points.
        #   That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour
        #   will be either horizontal, vertical or diagonal neighbors, that
        #   is max(abs(x1-x2),abs(y2-y1))==1.
        # - cv.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and
        #   diagonal segments and leaves only their end points. For example, an
        #   up-right rectangular contour is encoded with 4 points.
        # - cv.CHAIN_APPROX_TC89_L1: applies one of the flavors of the Teh-Chin
        #   chain approximation algorithm [209]
        # - cv.CHAIN_APPROX_TC89_KCOS: applies one of the flavors of the
        #   Teh-Chin chain approximation algorithm [209]
        contours_hierarchy_morph = list(
            cv2.findContours(
                closed_morph,
                mode=cv2.RETR_CCOMP,
                # mode=cv2.RETR_LIST,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )[0],
        )
        # filter out polygons
        return self.filter_out_polygons(contours_hierarchy_morph)

    def filter_out_polygons(
        self,
        contours_list: list[np.array],
    ) -> list[np.array]:
        """Filter out unwanted polygons with expert rules."""
        # 1. Remove the too small or too big
        # 3. Remove the ones touching borders of images
        keep_contours = []
        for contour in contours_list:
            area = cv2.contourArea(contour)
            touch_border = self.is_touch_border(contour)
            if self.surf_min <= area <= self.surf_max and not touch_border:
                keep_contours.append(contour)

        return keep_contours

    def draw_contours(self) -> np.ndarray:
        """Draw the contours on a color image."""
        if not self.contours:
            msg = "No contour to draw on plan."
            raise ValueError(msg)
        # Init a color image from the walls
        color_img_area = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)
        # Draw contours
        rng = np.random.default_rng(seed=1)
        for contour in self.contours:
            # Random color
            color = rng.integers(0, 256, 3).tolist()
            cv2.drawContours(
                color_img_area,
                [contour],
                contourIdx=-1,
                color=color,
                thickness=cv2.FILLED,
            )
        print(f"{len(self.contours)} rooms drawn.")
        # switch the y-axis
        return np.flipud(color_img_area)

    def is_touch_border(self, contour: np.ndarray) -> bool:
        """Compute if the polygon touches the border of the image."""
        height, width = self.binary.shape
        for point in contour:
            x, y = point[0]
            if x <= 0 or y <= 0 or x >= width - 1 or y >= height - 1:
                return True
        return False

    def polygons_from_contours(self) -> shapely.GeometryCollection:
        """Compute the coordinates of polygons."""
        if not self.contours:
            msg = "No contour to draw on plan."
            raise ValueError(msg)

        # Reshape to list of coordinnates and convert to polygons
        polygon_list = [
            shapely.Polygon(poly.reshape(-1, 2)) for poly in self.contours
        ]
        failed_conversions = len(self.contours) - len(polygon_list)
        if failed_conversions:
            msg = f"{failed_conversions} failed Polygons conversions."
            raise ValueError(msg)

        # Merge into Geometry collection
        gc = None
        try:
            gc = shapely.GeometryCollection(polygon_list)
        except Exception as err:
            msg = "Invalid geometry"
            raise ValueError(msg) from err
        return gc

    def plot_prediction(self) -> Image:
        """Display the colored polygons on the original plan."""
        if not self.rooms:
            self.rooms = self.draw_contours()
        return Image.fromarray(self.rooms)
