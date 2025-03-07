"""Parse the geojson function to detect the contours of the pieces.

@Version: 0.1
@Project: Capstone Vinci Contour Detection
@Date: 2025-01-25
@Author: Fabien Lagnieu, Tristan Waddington
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import shapely
from PIL import Image

# own imports
import utils.parse_geojson as pg


####################################################################################################
# @Fabien : Genrate binary image from segments
####################################################################################################
class CVSegmentation:
    """Find rooms in GeometryCollection using CV segmentation."""

    def __init__(
        self, dpi: int = 50, thickness: int = 7, dilatation_method="gaussian"
    ) -> "CVSegmentation":
        """Find rooms in GeometryCollection."""
        # Paramètres pour la génération des images
        self.dpi = dpi  # Changer la résolution ici (ex: 30, 50, 100...)
        # Épaisseur des lignes en pixels (impair de préférence)
        self.thickness = thickness if thickness % 2 else thickness + 1
        # 1m = dpi_choice pixels
        # Méthode de dilatation ('ellipse', 'cross', 'gaussian')
        # -> gaussien est plus efficace pour l'épaississement des traits
        # obliques et courbes)/
        self.dilation_method = dilatation_method

        # Class constants
        self.__name__ = f"CVSegmentation(dpi:{self.dpi}, thickness:{self.thickness}, dilatation:{self.dilation_method}"
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

    def __call__(self, x):
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
        # print(f"Create {width}x{height} blank image.")
        # Scale the segments at the desired scale
        # scale_segments = shapely.transform(
        #     self.segments,
        #     lambda x: pg._offset_reduce(x, minx, miny, self.scale),
        # )
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
        # return Image.fromarray(img)

    def find_contours(self, surf_min: int = 5000) -> list:
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
                method=cv2.CHAIN_APPROX_SIMPLE,
            )[0],
        )
        # print("Debug, contour hierarchy morph")
        # print(contours_hierarchy_morph[:2])
        # 3. Filter out too small polygons
        contours_hierarchy_morph = filtrer_par_surface(
            contours_hierarchy_morph,
            surf_min,
        )
        # 4. Suppress the outpolygon (biggest one)
        contours_hierarchy_morph = supprimer_polygone_le_plus_long(
            contours_hierarchy_morph,
        )
        return contours_hierarchy_morph

    def draw_contours(self):
        """Draw the contours on a color image."""
        if not self.contours:
            msg = "No contour to draw on plan."
            raise ValueError(msg)
        # Init a color image
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


####################################################################################################
# Détection des contours des pièces - impression pour visualisation
# - création du fichier GeoJSON à l'échelle et aux coordonnées initiales
####################################################################################################


# Charger les paramètres de transformation à partir du fichier JSON
def charger_parametres_transformation(json_path):
    with open(json_path, "r") as f:
        transform_data = json.load(f)
    transform_parameters = transform_data["transform_parameters"]
    dpi_scale = transform_data["dpi_scale"]  # 1m = 50 pixels (ou autre valeur)
    return transform_parameters, dpi_scale


# Fonction pour convertir les coordonnées pixels en coordonnées réelles
def pixel_to_real(x, y, x_offset, y_offset, scale_factor, hauteur_totale):
    x_real = x / scale_factor + x_offset
    y_real = (hauteur_totale - y) / scale_factor + y_offset  # Inversion de Y
    return [x_real, y_real]


# Fonction pour calculer la longueur de chaque contour
def calculer_longueur_contours(contours):
    longueurs = [cv2.arcLength(contour, True) for contour in contours]
    return longueurs


# Fonction pour supprimer les polygones inférieurs à une surface minimale
def filtrer_par_surface(contours, surface_minimale):
    return [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= surface_minimale
    ]


# Fonction pour supprimer le polygone ayant le périmètre le plus long -> murs extérieurs
def supprimer_polygone_le_plus_long(contours):
    longueurs = calculer_longueur_contours(contours)
    index_max = np.argmax(longueurs)
    # print(f"Suppression du polygone le plus long (murs extérieurs) avec une longueur de : {longueurs[index_max]} pixels")
    contours.pop(index_max)
    return contours


# Fonction pour convertir les contours en format GeoJSON avec échelle et transformation
def contours_to_geojson(
    contours,
    image_name,
    hauteur_totale,
    surface_minimale,
    x_offset,
    y_offset,
    scale_factor,
    dpi_scale,
    rooms_contours_geojson_dir,
):
    # Structure GeoJSON
    geojson = {"type": "FeatureCollection", "features": []}

    # Parcours des contours pour les convertir en polygones GeoJSON
    for i, contour in enumerate(contours):
        # Calcul de la surface de chaque contour
        surface = cv2.contourArea(contour)

        # Filtrer par surface minimale
        if surface >= surface_minimale:
            # Extraction des points du contour
            coords = contour.squeeze().tolist()

            # Conversion des coordonnées pixels en coordonnées réelles
            real_coords = [
                pixel_to_real(
                    x, y, x_offset, y_offset, scale_factor, hauteur_totale
                )
                for x, y in coords
            ]

            # Fermeture du polygone en répétant le premier point à la fin
            real_coords.append(real_coords[0])

            # Construction du polygone GeoJSON
            polygon = {
                "type": "Feature",
                "properties": {
                    "name": f"room_{i + 1} / {len(contours)}",
                    "surface_px": surface,
                    "surface_m2": surface / (dpi_scale**2),  # Conversion en m²
                    "image": image_name,
                },
                "geometry": {"type": "Polygon", "coordinates": [real_coords]},
            }
            geojson["features"].append(polygon)

    # Sauvegarde du GeoJSON dans un fichier
    output_dir = rooms_contours_geojson_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{image_name}_rooms_contours.geojson"
    )
    with open(output_path, "w") as geojson_file:
        json.dump(geojson, geojson_file, indent=4)

    print(f"GeoJSON avec échelle enregistré : {output_path}\n")


# Fonction principale pour générer les pièces et exporter en GeoJSON
def generer_pieces_image_et_geojson(
    file_name,
    binary_images_dir,
    metadatas_dir,
    contours_images_dir=Path("04_contours_images"),
    rooms_contours_geojson_dir=Path("05_rooms_contours_geojson"),
    surface_minimale=5000,
):
    # Charger les paramètres de transformation
    image_path = binary_images_dir / f"{file_name}_binary_image.png"
    metadata_path = metadatas_dir / f"{file_name}_metadata.json"
    transform_parameters, dpi_scale = charger_parametres_transformation(
        metadata_path
    )
    x_offset = transform_parameters[0]
    y_offset = transform_parameters[1]
    scale_factor = transform_parameters[2] * dpi_scale

    # Charger l'image en niveau de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hauteur_totale = img.shape[0]  # Hauteur totale de l'image en pixels

    # Appliquer un seuillage pour binariser l'image
    _, binary_improved = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Application d'une fermeture morphologique pour mieux détecter les contours internes
    kernel = np.ones((5, 5), np.uint8)
    closed_morph = cv2.morphologyEx(
        binary_improved, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # Détection des contours en hiérarchie avec l'image traitée
    contours_hierarchy_morph = list(
        cv2.findContours(closed_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]
    )
    contours_hierarchy_morph = filtrer_par_surface(
        contours_hierarchy_morph, surface_minimale
    )

    # Suppression du polygone avec la longueur maximale (murs extérieurs)
    contours_hierarchy_morph = supprimer_polygone_le_plus_long(
        contours_hierarchy_morph
    )

    # Réinitialisation de l'image couleur pour la coloration
    color_img_area = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Initialisation du compteur de pièces
    compteur_pieces = 0

    # Parcours des contours hiérarchiques pour colorer uniquement les pièces assez grandes
    for contour in contours_hierarchy_morph:
        # Calcul de la surface de chaque contour
        surface = cv2.contourArea(contour)

        # Appliquer le filtre de surface minimale
        if surface >= surface_minimale:
            # Choix d'une couleur aléatoire pour chaque pièce
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(
                color_img_area, [contour], -1, color, thickness=cv2.FILLED
            )

            # Incrémenter le compteur de pièces
            compteur_pieces += 1

    # Affichage du nombre de pièces détectées
    print(
        f"Nombre de pièces détectées pour le fichier '{file_name}' : {compteur_pieces}"
    )

    # Sauvegarde de l'image colorée en PNG
    contours_image_dir = contours_images_dir
    output_image_path = os.path.join(
        contours_image_dir, f"{file_name}_contours_image.png"
    )
    cv2.imwrite(str(output_image_path), color_img_area)
    print(f"Image enregistrée : {output_image_path}")

    # Export des contours en GeoJSON avec échelle appliquée
    contours_to_geojson(
        contours_hierarchy_morph,
        file_name,
        hauteur_totale,
        surface_minimale,
        x_offset,
        y_offset,
        scale_factor,
        dpi_scale,
        rooms_contours_geojson_dir,
    )


####################################################################################################
# Main
####################################################################################################
if __name__ == "__main__":
    ### ETAPE 0 : Création des fichiers de stockage des données ###
    input_dir = Path("00_input_geojson")
    processed_dir = Path("01_processed_geojson")
    binary_images_dir = Path("02_binary_images")
    metadatas_dir = Path("03_metadatas")
    contours_images_dir = Path("04_contours_images")
    rooms_contours_geojson_dir = Path("05_rooms_contours_geojson")

    for directory in [
        processed_dir,
        binary_images_dir,
        metadatas_dir,
        contours_images_dir,
        rooms_contours_geojson_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    ### ETAPE 1 : Nettoyage et uniformisation des données d'entrée (geojson) en segments + Chargement des segments ###
    files_names = []
    for filepath in sorted(input_dir.iterdir()):
        files_names.append(filepath.stem)
        out_file = processed_dir / f"{filepath.stem}_clean.geojson"
        pg.clean_geojson_to_segments_and_save(filepath, out_file)

    paths = []
    for file_name in files_names:
        paths.append(processed_dir / f"{file_name}_clean.geojson")

    # Load the segments
    segments = []
    transform_parameters = []
    for i, path in enumerate(paths):
        segment, transform_parameter = pg.load_segments(path)
        segments.append(segment)
        transform_parameters.append(transform_parameter)

    ### ETAPE 2 : Génération des images binaires à partir des segments ###

    # Paramètres pour la génération des images
    dpi_choice = 50  # Changer la résolution ici (ex: 30, 50, 100...)
    thickness_choice = (
        3  # Épaisseur des lignes en pixels (impair de préférence)
    )
    scale = dpi_choice  # 1m = dpi_choice pixels

    # Méthode de dilatation ('ellipse', 'cross', 'gaussian'
    # -> gaussien est plus efficace pour l'épaississement des traits obliques et courbes)
    method_choice = "gaussian"

    # Génération des images avec le DPI et épaisseur des traits choisis
    for i, segment, file_name in zip(
        range(len(segments)), segments, files_names
    ):
        generate_binary_image(
            segment,
            transform_parameters[i],
            file_name,
            binary_images_dir,
            metadatas_dir,
            scale=scale,
            thickness=thickness_choice,
            method=method_choice,
        )

    ### ETAPE 3 : Détection des contours des pièces et export de l'image colorée + GeoJSON ###

    for file_name in files_names:
        generer_pieces_image_et_geojson(
            file_name,
            binary_images_dir,
            metadatas_dir,
            contours_images_dir,
            rooms_contours_geojson_dir,
            surface_minimale=4000,
        )
