
"""
Parse the geojson function to detect the contours of the pieces

@Version: 0.1
@Project: Capstone Vinci Contour Detection
@Date: 2025-01-25
@Author: Fabien Lagnieu
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from importlib import reload
import shapely
from PIL import Image
import cv2
import cv2.ximgproc
import os
# own imports
import parse_geojson as pg
import contours_functions as cf

####################################################################################################
# @Fabien : Genrate binary image from segments
####################################################################################################


def generate_binary_image(segment, transform_parameter, file_name, binary_images_dir, metadatas_dir,
                          scale=100, thickness=1, method='ellipse'):
    """
    Génère une image binaire à partir d'un segment et sauvegarde 
    les métadonnées.
    `thickness` permet de définir l'épaisseur des lignes en pixels.
    `method` définit la méthode de dilatation : 'ellipse', 'cross', 'gaussian'.
    """
    if len(segment.geoms) == 0:
        print("Skipping: No geometries found.")
        return
    
    # Déterminer les bornes
    minx, miny, maxx, maxy = segment.bounds
    width = int((maxx - minx) * scale) + 1
    height = int((maxy - miny) * scale) + 1
    
    # Sauvegarder les paramètres de transformation
    binary_image_path = binary_images_dir  / f"{file_name}_binary_image.png"
    metadata_path = metadatas_dir / f"{file_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"transform_parameters": transform_parameter, "dpi_scale": scale}, f, indent=4)
    
    # Création de l'image binaire
    img = np.ones((height, width), dtype=np.uint8) * 255  # Fond blanc
    
    # Tracer les segments avec OpenCV
    for line in segment.geoms:
        x_vals, y_vals = line.xy
        x_pixels = ((np.array(x_vals) - minx) * scale).astype(int)
        y_pixels = height - ((np.array(y_vals) - miny) * scale).astype(int)
        
        for j in range(len(x_pixels) - 1):
            pt1 = (x_pixels[j], y_pixels[j])
            pt2 = (x_pixels[j + 1], y_pixels[j + 1])
            if 0 <= pt1[0] < width and 0 <= pt1[1] < height and 0 <= pt2[0] < width and 0 <= pt2[1] < height:
                cv2.line(img, pt1, pt2, 0, thickness)  # Épaisseur personnalisée
    
    # Épaississement avec méthode choisie
    if thickness > 1:
        if method == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        elif method == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (thickness, thickness))
        elif method == 'gaussian':
            img = cv2.GaussianBlur(img, (thickness, thickness), 0)
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        else:  # Méthode par défaut (ellipse)
            kernel = np.ones((thickness, thickness), np.uint8)
        if method != 'gaussian':
            img = cv2.dilate(img, kernel, iterations=1)
    
    # Sauvegarde de l'image binaire propre
    binary_image = Image.fromarray(img)
    binary_image.save(binary_image_path)
    
    print(f"Saved binary map: {binary_image_path}")
    print(f"Saved transform metadata: {metadata_path}")

    
####################################################################################################
# Détection des contours des pièces - impression pour visualisation 
# - création du fichier GeoJSON à l'échelle et aux coordonnées initiales
####################################################################################################

# Charger les paramètres de transformation à partir du fichier JSON
def charger_parametres_transformation(json_path):
    with open(json_path, 'r') as f:
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
    return [contour for contour in contours if cv2.contourArea(contour) >= surface_minimale]

# Fonction pour supprimer le polygone ayant le périmètre le plus long -> murs extérieurs
def supprimer_polygone_le_plus_long(contours):
    longueurs = calculer_longueur_contours(contours)
    index_max = np.argmax(longueurs)
    print(f"Suppression du polygone le plus long (murs extérieurs) avec une longueur de : {longueurs[index_max]} pixels")
    contours.pop(index_max)
    return contours

# Fonction pour convertir les contours en format GeoJSON avec échelle et transformation
def contours_to_geojson(contours, image_name, hauteur_totale, surface_minimale, x_offset, y_offset, scale_factor, dpi_scale, rooms_contours_geojson_dir):   
    # Structure GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Parcours des contours pour les convertir en polygones GeoJSON
    for contour in contours:
        # Calcul de la surface de chaque contour
        surface = cv2.contourArea(contour)

        # Filtrer par surface minimale
        if surface >= surface_minimale:
            # Extraction des points du contour
            coords = contour.squeeze().tolist()

            # Conversion des coordonnées pixels en coordonnées réelles
            real_coords = [pixel_to_real(x, y, x_offset, y_offset, scale_factor, hauteur_totale) for x, y in coords]

            # Fermeture du polygone en répétant le premier point à la fin
            real_coords.append(real_coords[0])
            
            # Construction du polygone GeoJSON
            polygon = {
                "type": "Feature",
                "properties": {
                    "surface_px": surface,
                    "surface_m2": surface / (dpi_scale**2),  # Conversion en m²
                    "image": image_name
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [real_coords]
                }
            }
            geojson["features"].append(polygon)

    # Sauvegarde du GeoJSON dans un fichier
    output_dir = rooms_contours_geojson_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{image_name}_rooms_contours.geojson")
    with open(output_path, 'w') as geojson_file:
        json.dump(geojson, geojson_file, indent=4)
    
    print(f"GeoJSON avec échelle enregistré : {output_path}")

# Fonction principale pour générer les pièces et exporter en GeoJSON
def generer_et_afficher_pieces(file_name, binary_images_dir, metadatas_dir, contours_images_dir, rooms_contours_geojson_dir, surface_minimale=5000):
    # Charger les paramètres de transformation
    image_path = binary_images_dir / f"{file_name}_binary_image.png"
    metadata_path = metadatas_dir / f"{file_name}_metadata.json"
    transform_parameters, dpi_scale = charger_parametres_transformation(metadata_path)
    x_offset = transform_parameters[0]
    y_offset = transform_parameters[1]
    scale_factor = transform_parameters[2]*dpi_scale

    # Charger l'image en niveau de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hauteur_totale = img.shape[0]  # Hauteur totale de l'image en pixels

    # Appliquer un seuillage pour binariser l'image
    _, binary_improved = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Application d'une fermeture morphologique pour mieux détecter les contours internes
    kernel = np.ones((5, 5), np.uint8)
    closed_morph = cv2.morphologyEx(binary_improved, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Détection des contours en hiérarchie avec l'image traitée
    contours_hierarchy_morph = list(cv2.findContours(closed_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours_hierarchy_morph = filtrer_par_surface(contours_hierarchy_morph, surface_minimale)

    # Suppression du polygone avec la longueur maximale (murs extérieurs)
    contours_hierarchy_morph = supprimer_polygone_le_plus_long(contours_hierarchy_morph)

    # Réinitialisation de l'image couleur pour la coloration
    color_img_area = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Initialisation du compteur de pièces
    compteur_pieces = 0

    # Parcours des contours hiérarchiques pour colorer uniquement les pièces assez grandes
    for i, contour in enumerate(contours_hierarchy_morph):
        # Calcul de la surface de chaque contour
        surface = cv2.contourArea(contour)

        # Appliquer le filtre de surface minimale
        if surface >= surface_minimale:
            # Choix d'une couleur aléatoire pour chaque pièce
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(color_img_area, [contour], -1, color, thickness=cv2.FILLED)
            
            # Incrémenter le compteur de pièces
            compteur_pieces += 1

    # Affichage du nombre de pièces détectées
    print(f"Nombre de pièces détectées pour {image_path} : {compteur_pieces}")

    # Sauvegarde de l'image colorée en PNG
    contours_image_dir = contours_images_dir
    output_image_path = os.path.join(contours_image_dir, f"{file_name}_contours_image.png")
    cv2.imwrite(str(output_image_path), color_img_area)
    print(f"Image enregistrée : {output_image_path}")

    # Affichage du résultat
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(color_img_area, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Export des contours en GeoJSON avec échelle appliquée
    contours_to_geojson(contours_hierarchy_morph, file_name, hauteur_totale, surface_minimale, x_offset, y_offset, scale_factor, dpi_scale, rooms_contours_geojson_dir)