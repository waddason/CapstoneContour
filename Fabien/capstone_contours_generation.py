
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

####################################################################################################
# @Fabien : Genrate binary image from segments
####################################################################################################


def generate_binary_image(segment, transform_parameter, file_name, 
                          binary_images_dir=Path("02_binary_images"), 
                          metadatas_dir=Path("03_metadatas"),
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
    print(f"Saved transform metadata: {metadata_path}\n")

    
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

# Fonction pour détecter et supprimer le polygone sucetibles d'être des murs
def detecter_et_filtrer_murs(contours, img, dpi=50, epaisseur_min_m=0.5, epsilon_ratio=0.01):
    """
    Détection et filtrage des contours de murs :
    
    Étapes :
    1️⃣ Supprimer le contour avec le périmètre le plus long
    2️⃣ Supprimer les contours dont le périmètre dépasse le périmètre global de l'image
    3️⃣ Simplifier les contours restants avec approxPolyDP
    4️⃣ Calculer les épaisseurs et supprimer ceux en dessous du seuil
    5️⃣ Retourner les contours ORIGINAUX filtrés
    
    Paramètres :
    - dpi : résolution en pixels par mètre
    - epaisseur_min_m : épaisseur minimale en mètres
    - epsilon_ratio : pourcentage de tolérance pour approxPolyDP
    """
    
    if len(contours) == 0:
        print("❗ Aucun contour fourni.")
        return []

    contours_filtres = contours.copy()

    # ➡️ 1. Supprimer le contour le plus long
    longueurs = [cv2.arcLength(c, True) for c in contours_filtres]
    index_max = np.argmax(longueurs)
    perim_max = longueurs[index_max]
    print(f"➡️ Suppression du contour le plus long : index {index_max}, périmètre {perim_max:.2f}px")
    
    contours_filtres.pop(index_max)

    # # ➡️ 2. Calcul du périmètre global de l'image
    # h, w = img.shape
    # perimetre_image = 2 * (w + h)

    contours_apres_perimetre = []
    for idx, contour in enumerate(contours_filtres):
        perimetre = cv2.arcLength(contour, True)

        # if perimetre > perimetre_image:
        #     print(f"❌ Contour {idx} supprimé : périmètre {perimetre:.2f}px > périmètre image {perimetre_image:.2f}px")
        #     continue
        
        contours_apres_perimetre.append(contour)

    print(f"✅ Restants après suppression périmètre : {len(contours_apres_perimetre)} / {len(contours)}")

    # ➡️ 3. Approximation des contours pour simplification géométrique
    contours_approximated = []
    for idx, contour in enumerate(contours_apres_perimetre):
        epsilon = epsilon_ratio * cv2.contourArea(contour)/cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approximated.append(approx)

    # ➡️ 4. Vérifier l'épaisseur moyenne sur les contours approximés
    seuil_epaisseur_pixels = epaisseur_min_m * dpi
    print(f"ℹ️ Seuil d'épaisseur en pixels : {seuil_epaisseur_pixels:.2f}px (équivalent à {epaisseur_min_m}m)")

    contours_resultats = []
    murs_supprimes = 0

    for idx, (contour_orig, contour_approx) in enumerate(zip(contours_apres_perimetre, contours_approximated)):
        surface = cv2.contourArea(contour_approx)
        perimetre = cv2.arcLength(contour_approx, True)

        if perimetre == 0 or surface == 0:
            print(f"❌ Contour {idx} ignoré : surface ou périmètre nul")
            continue

        epaisseur_moyenne = surface / perimetre

        if epaisseur_moyenne < seuil_epaisseur_pixels:
            murs_supprimes += 1
            print(f"❌ Contour {idx} supprimé : épaisseur {epaisseur_moyenne:.2f}px < seuil {seuil_epaisseur_pixels:.2f}px")
            continue

        # ➡️ Ajouter le contour original à la liste des résultats
        contours_resultats.append(contour_orig)

    print(f"\n✅ Murs supprimés par épaisseur : {murs_supprimes}")
    print(f"✅ Contours retenus : {len(contours_resultats)} / {len(contours)} initiaux")

    return contours_resultats


# Fonction pour convertir les contours en format GeoJSON avec échelle et transformation
def contours_to_geojson(contours, image_name, hauteur_totale, surface_minimale, x_offset, y_offset, scale_factor, dpi_scale, rooms_contours_geojson_dir):   
    # Structure GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Parcours des contours pour les convertir en polygones GeoJSON
    for i, contour in enumerate(contours):
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
                    "name" : f"room_{i+1} / {len(contours)}",
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
    
    print(f"GeoJSON avec échelle enregistré : {output_path}\n")

# Fonction principale pour générer les pièces et exporter en GeoJSON
def generer_pieces_image_et_geojson(file_name, binary_images_dir, metadatas_dir, contours_images_dir=Path("04_contours_images"), rooms_contours_geojson_dir=Path("05_rooms_contours_geojson"), surface_minimale=5000,                              
                                    dpi=50, epaisseur_min_m=0.5, epsilon_ratio=0.01):
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

    # Suppression des murs via analyse d'épaisseur et connexité
    contours_hierarchy_morph = detecter_et_filtrer_murs(contours_hierarchy_morph, img, dpi=dpi, epaisseur_min_m=epaisseur_min_m, epsilon_ratio=epsilon_ratio)

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
            cv2.drawContours(color_img_area, [contour], -1, color, thickness=cv2.FILLED)
            
            # Incrémenter le compteur de pièces
            compteur_pieces += 1

    # Affichage du nombre de pièces détectées
    print(f"Nombre de pièces détectées pour le fichier '{file_name}' : {compteur_pieces}")

    # Sauvegarde de l'image colorée en PNG
    contours_image_dir = contours_images_dir
    output_image_path = os.path.join(contours_image_dir, f"{file_name}_contours_image.png")
    cv2.imwrite(str(output_image_path), color_img_area)
    print(f"Image enregistrée : {output_image_path}")


    # Export des contours en GeoJSON avec échelle appliquée
    contours_to_geojson(contours_hierarchy_morph, file_name, hauteur_totale, surface_minimale, x_offset, y_offset, scale_factor, dpi_scale, rooms_contours_geojson_dir)

    # retourner le nombre de pièces détectées
    return compteur_pieces

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

    for directory in [processed_dir, binary_images_dir, metadatas_dir, contours_images_dir, rooms_contours_geojson_dir]:
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
    thickness_choice = 3  # Épaisseur des lignes en pixels (impair de préférence)
    scale = dpi_choice  # 1m = dpi_choice pixels

    # Méthode de dilatation ('ellipse', 'cross', 'gaussian' 
    # -> gaussien est plus efficace pour l'épaississement des traits obliques et courbes)
    method_choice = 'gaussian'

    # Génération des images avec le DPI et épaisseur des traits choisis
    for i, segment, file_name in zip(range(len(segments)), segments, files_names):
        generate_binary_image(
            segment,
            transform_parameters[i],
            file_name,
            binary_images_dir,
            metadatas_dir,
            scale=scale,
            thickness=thickness_choice,
            method=method_choice
        )

    ### ETAPE 3 : Détection des contours des pièces et export de l'image colorée + GeoJSON ###

    surface_minimale=1 # surface minimale en m²
    surface_minimale_pixels = surface_minimale * (dpi_choice**2)  # Conversion en pixels
    epaisseur_min_m=0.25
    epsilon_ratio=0.05

    for file_name in files_names:
        generer_pieces_image_et_geojson(
                file_name,
                binary_images_dir,
                metadatas_dir,
                contours_images_dir,
                rooms_contours_geojson_dir,
                surface_minimale=surface_minimale_pixels,
                dpi=dpi_choice,
                epaisseur_min_m=epaisseur_min_m, 
                epsilon_ratio=epsilon_ratio
        )

