import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import cv2
import shapely
from shapely.geometry import GeometryCollection, Polygon

# own imports
import utils.parse_geojson as pg


class CapstoneVisionSegmentation:
    """
    Modèle de détection de pièces basé sur OpenCV avec filtrage des murs, export GeoJSON, etc.
    """

    def __init__(self,
                 dpi=50,
                 thickness=3,
                 scale=None,
                 method='gaussian',
                 surface_min_m2=0.5,
                 epaisseur_min_m=0.25,
                 data_dir=Path("data_segmentation")):
        """
        Initialisation du modèle.

        Args:
            dpi: résolution en px/m
            thickness: épaisseur des traits
            scale: facteur de conversion mètres → pixels (par défaut dpi)
            method: méthode de dilatation
            surface_min_m2: surface minimale à conserver (m²)
            epaisseur_min_m: seuil minimum pour détecter un mur (m)
            data_dir: dossier de stockage des données intermédiaires
        """

        # Paramètres principaux
        self.dpi = dpi
        self.thickness = thickness if thickness % 2 else thickness + 1
        self.scale = scale or dpi
        self.method = method
        self.surface_min_m2 = surface_min_m2
        self.surface_min_px = surface_min_m2 * (dpi ** 2)
        self.epaisseur_min_m = epaisseur_min_m
        self.epaisseur_min_px = epaisseur_min_m * dpi

        # Dossiers de sortie
        self.data_dir = Path(data_dir)
        self.input_dir = self.data_dir / "00_input_geojson"
        self.processed_dir = self.data_dir / "01_processed_geojson"
        self.binary_images_dir = self.data_dir / "02_binary_images"
        self.metadatas_dir = self.data_dir / "03_metadatas"
        self.contours_images_dir = self.data_dir / "04_contours_images"
        self.rooms_geojson_dir = self.data_dir / "05_rooms_contours_geojson"

        for directory in [self.input_dir, self.processed_dir, self.binary_images_dir,
                          self.metadatas_dir, self.contours_images_dir, self.rooms_geojson_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.__name__ = (
            f"CapstoneSegmentation(dpi:{self.dpi}, thickness:{self.thickness}, "
            f"method_dilatation:{self.method}, surface_min_m2:{self.surface_min_m2}, "
            f"surface_min_px:{self.surface_min_px})"
        )

    def __call__(self, gc_raw):
        """Appel direct avec predict"""
        return self.predict(gc_raw)

    def predict(self, gc_raw):
        """
        Pipeline complet pour transformer une GeometryCollection en rooms GeometryCollection.
        """
        # 1️⃣ Sauvegarde du geojson brut
        file_name = "input_gc"
        input_geojson_path = self.input_dir / f"{file_name}.geojson"
        pg.save_geometry_collection_to_geojson(gc_raw, input_geojson_path)

        # 2️⃣ Nettoyage + extraction des segments
        processed_geojson_path = self.processed_dir / f"{file_name}_clean.geojson"
        pg.clean_geojson_to_segments_and_save(input_geojson_path, processed_geojson_path)

        # Charger les segments
        segments, transform_parameters = pg.load_segments(processed_geojson_path)

        # 3️⃣ Génération de l'image binaire
        self.generate_binary_image(
            segments,
            transform_parameters,
            file_name
        )

        # 4️⃣ Détection des contours et export des pièces
        self.generer_pieces_image_et_geojson(
            file_name=file_name,
            surface_minimale=self.surface_min_px,
            dpi=self.dpi,
            epaisseur_min_m=self.epaisseur_min_m
        )

        # 5️⃣ Chargement des rooms détectés
        output_geojson_path = self.rooms_geojson_dir / f"{file_name}_rooms_contours.geojson"
        y_pred = pg.load_geojson_as_geometry_collection(output_geojson_path)

        return y_pred

    def generate_binary_image(self, segment, transform_parameter, file_name):
        """Génère une image binaire à partir des segments."""
        if len(segment.geoms) == 0:
            print("❗ Aucun segment détecté.")
            return

        minx, miny, maxx, maxy = segment.bounds
        width = int((maxx - minx) * self.scale) + 1
        height = int((maxy - miny) * self.scale) + 1

        binary_image_path = self.binary_images_dir / f"{file_name}_binary_image.png"
        metadata_path = self.metadatas_dir / f"{file_name}_metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "transform_parameters": transform_parameter,
                "dpi_scale": self.scale
            }, f, indent=4)

        img = np.ones((height, width), dtype=np.uint8) * 255  # blanc

        # ➡️ Tracer les segments
        for line in segment.geoms:
            x_vals, y_vals = line.xy
            x_pixels = ((np.array(x_vals) - minx) * self.scale).astype(int)
            y_pixels = height - ((np.array(y_vals) - miny) * self.scale).astype(int)

            # Clamp pour éviter les débordements
            x_pixels = np.clip(x_pixels, 0, width - 1)
            y_pixels = np.clip(y_pixels, 0, height - 1)
            for j in range(len(x_pixels) - 1):
                pt1 = (x_pixels[j], y_pixels[j])
                pt2 = (x_pixels[j + 1], y_pixels[j + 1])
                if 0 <= pt1[0] < width and 0 <= pt1[1] < height and 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                    cv2.line(img, pt1, pt2, 0, self.thickness)

        # ➡️ Dilatation
        if self.thickness > 1:
            if self.method == 'ellipse':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.thickness, self.thickness))
                img = cv2.dilate(img, kernel, iterations=1)
            elif self.method == 'cross':
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self.thickness, self.thickness))
                img = cv2.dilate(img, kernel, iterations=1)
            elif self.method == 'gaussian':
                img = cv2.GaussianBlur(img, (self.thickness, self.thickness), 0)
                _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        # ➡️ Sauvegarde
        binary_image = Image.fromarray(img)
        binary_image.save(binary_image_path)

        #print(f"✅ Binary image saved: {binary_image_path}")
        #print(f"✅ Metadata saved: {metadata_path}")

    def generer_pieces_image_et_geojson(self, file_name, surface_minimale, dpi, epaisseur_min_m):
        """Détecte les contours et exporte en GeoJSON."""
        image_path = self.binary_images_dir / f"{file_name}_binary_image.png"
        metadata_path = self.metadatas_dir / f"{file_name}_metadata.json"

        with open(metadata_path, 'r') as f:
            transform_data = json.load(f)
        transform_parameters = transform_data["transform_parameters"]
        dpi_scale = transform_data["dpi_scale"]

        x_offset = transform_parameters[0]
        y_offset = transform_parameters[1]
        scale_factor = transform_parameters[2] * dpi_scale

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        hauteur_totale = img.shape[0]

        _, binary_improved = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        closed_morph = cv2.morphologyEx(binary_improved, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, hierarchy = cv2.findContours(closed_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            print(f"⚠️ Aucun contour trouvé pour {file_name}")
            return

        # ➡️ Filtrage murs
        contours_filtered, surfaces = self.detecter_et_filtrer_murs(contours, hierarchy, dpi, epaisseur_min_m)

        # ➡️ Affichage
        color_img_area = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for contour in contours_filtered:
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(color_img_area, [contour], -1, color, thickness=cv2.FILLED)

        output_image_path = self.contours_images_dir / f"{file_name}_contours_image.png"
        cv2.imwrite(str(output_image_path), color_img_area)

        #print(f"✅ Contours image saved: {output_image_path}")

        # ➡️ Export GeoJSON
        self.contours_to_geojson(contours_filtered, file_name, hauteur_totale,
                                 surface_minimale, x_offset, y_offset,
                                 scale_factor, dpi_scale)

    def detecter_et_filtrer_murs(self, contours, hierarchy, dpi, epaisseur_min_m):
        """Filtre les contours en supprimant les murs."""
        seuil_epaisseur_pixels = epaisseur_min_m * dpi
        hierarchy = hierarchy[0]

        indices_a_garder = []
        surfaces = []

        for idx, contour in enumerate(contours):
            surface = self.calculer_surface_avec_trous(contours, hierarchy, idx)
            perimetre = cv2.arcLength(contour, True)

            if perimetre == 0 or surface == 0:
                continue

            epaisseur_moyenne = surface / perimetre

            if epaisseur_moyenne >= seuil_epaisseur_pixels:
                indices_a_garder.append(idx)
                surfaces.append(surface)

        if not indices_a_garder:
            print("❗ Aucun contour retenu après filtrage.")
            return [], []

        tri = sorted(zip(surfaces, indices_a_garder), key=lambda x: x[0], reverse=True)
        sorted_indices = [i for _, i in tri]
        contours_resultats = [contours[i] for i in sorted_indices]

        return contours_resultats, surfaces

    def calculer_surface_avec_trous(self, contours, hierarchy, idx_parent):
        """Calcule la surface nette (surface - trous)."""
        surface = cv2.contourArea(contours[idx_parent])
        idx_child = hierarchy[idx_parent][2]

        while idx_child != -1:
            surface -= cv2.contourArea(contours[idx_child])
            idx_child = hierarchy[idx_child][0]

        return abs(surface)

    def contours_to_geojson(self, contours, image_name, hauteur_totale,
                            surface_minimale, x_offset, y_offset,
                            scale_factor, dpi_scale):
        """Exporte les contours en GeoJSON."""
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for i, contour in enumerate(contours):
            surface = cv2.contourArea(contour)
            if surface >= surface_minimale:
                coords = contour.squeeze().tolist()
                real_coords = [
                    [x / scale_factor + x_offset,
                     (hauteur_totale - y) / scale_factor + y_offset]
                    for x, y in coords
                ]
                real_coords.append(real_coords[0])

                polygon = {
                    "type": "Feature",
                    "properties": {
                        "name": f"room_{i + 1} / {len(contours)}",
                        "surface_px": surface,
                        "surface_m2": surface / (dpi_scale ** 2),
                        "image": image_name
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [real_coords]
                    }
                }
                geojson["features"].append(polygon)

        output_geojson_path = self.rooms_geojson_dir / f"{image_name}_rooms_contours.geojson"
        with open(output_geojson_path, 'w') as f:
            json.dump(geojson, f, indent=4)

        #print(f"✅ GeoJSON saved: {output_geojson_path}")