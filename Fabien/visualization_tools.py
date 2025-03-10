import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import parse_geojson as pg
import capstone_contours_generation as ccg


def create_ui_pipeline():
    """
    Interface interactive pour le pipeline de traitement GeoJSON.
    Tous les paramètres sont accessibles, avec des explications claires.
    """

    ### Répertoires de travail ###
    input_dir = Path("00_input_geojson")
    processed_dir = Path("01_processed_geojson")
    binary_images_dir = Path("02_binary_images")
    metadatas_dir = Path("03_metadatas")
    contours_images_dir = Path("04_contours_images")
    rooms_contours_geojson_dir = Path("05_rooms_contours_geojson")

    for directory in [input_dir, processed_dir, binary_images_dir, metadatas_dir, contours_images_dir, rooms_contours_geojson_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    ### Widgets ###
    file_name_input = widgets.Text(
        value="",
        placeholder="Nom du fichier sans extension (ex : Output0)",
        description="📁 Fichier :",
        style={'description_width': '150px'},
        layout=widgets.Layout(width='50%')
    )

    # -------------------------
    # Bloc 1 : Image Binaire
    # -------------------------
    dpi_label = widgets.HTML(
        "<b>📏 DPI (px/m)</b><br>Définit l’échelle de conversion des distances en pixels. Plus il est élevé, plus l’image est détaillée. <br><i>Valeur conseillée : 50</i>"
    )
    dpi_choice_slider = widgets.IntSlider(
        value=50, min=10, max=150, step=5,
        layout=widgets.Layout(width='50%')
    )

    thickness_label = widgets.HTML(
        "<b>✏️ Épaisseur des contours (px)</b><br>Contrôle l’épaisseur des traits sur l’image binaire générée.<br><i>Valeur conseillée : 3 à 9 px</i>"
    )
    thickness_choice_slider = widgets.IntSlider(
        value=3, min=1, max=15, step=2,
        layout=widgets.Layout(width='50%')
    )

    # -------------------------
    # Bloc 2 : Filtrage Contours
    # -------------------------
    surface_label = widgets.HTML(
        "<b>🏠 Surface minimale pièce (m²)</b><br>Exclut les petites zones parasites (non pièces).<br><i>Valeur conseillée : 1.0 m²</i>"
    )
    surface_min_piece_m2_slider = widgets.FloatSlider(
        value=1.0, min=0, max=10.0, step=0.1,
        layout=widgets.Layout(width='50%')
    )

    # -------------------------
    # Bloc 3 : Suppression des Murs
    # -------------------------
    epaisseur_mur_label = widgets.HTML(
        "<b>🧱 Épaisseur minimale mur (m)</b><br>Filtre les objets trop fins pour être considérés comme des pièces.<br><i>Valeur conseillée : 0.25 m</i>"
    )
    epaisseur_min_murale_m_slider = widgets.FloatSlider(
        value=0.25, min=0, max=2.0, step=0.05,
        layout=widgets.Layout(width='50%')
    )

    epsilon_label = widgets.HTML(
        "<b>📐 Tolérance approximation (epsilon)</b><br>Contrôle la simplification géométrique (Douglas-Peucker). <br>Plus la valeur est grande, plus le contour est simplifié.<br><i>Valeur conseillée : 0.01 à 0.1</i>"
    )
    epsilon_ratio_slider = widgets.FloatSlider(
        value=0.05, min=0.001, max=0.2, step=0.005,
        layout=widgets.Layout(width='50%')
    )

    # Bouton de lancement
    button = widgets.Button(
        description="🚀 Lancer l'analyse",
        button_style='success',
        layout=widgets.Layout(width='50%')
    )

    output = widgets.Output()

    # -------------------------
    # Interface complète
    # -------------------------
    ui = widgets.VBox([
        widgets.HTML("<h2>📋 Traitement complet des plans GeoJSON</h2>"),

        file_name_input,

        widgets.HTML("<h4>1️⃣ Paramètres de génération de l’image binaire</h4>"),
        dpi_label, dpi_choice_slider,
        thickness_label, thickness_choice_slider,

        widgets.HTML("<h4>2️⃣ Filtrage des contours détectés</h4>"),
        surface_label, surface_min_piece_m2_slider,

        widgets.HTML("<h4>3️⃣ Suppression automatique des murs</h4>"),
        epaisseur_mur_label, epaisseur_min_murale_m_slider,
        epsilon_label, epsilon_ratio_slider,

        button,
        output
    ])

    ### Fonction exécutée au clic du bouton ###
    def on_button_clicked(b):
        with output:
            clear_output(wait=True)
            file_name = file_name_input.value.strip()

            if not file_name:
                print("⚠️ Veuillez entrer un nom de fichier.")
                return

            # Paramètres récupérés depuis les sliders
            dpi_choice = dpi_choice_slider.value
            thickness_choice = thickness_choice_slider.value
            surface_min_piece_m2 = surface_min_piece_m2_slider.value
            epaisseur_min_murale_m = epaisseur_min_murale_m_slider.value
            epsilon_ratio = epsilon_ratio_slider.value

            # Conversion surface m² vers pixels
            surface_min_piece_pixels = surface_min_piece_m2 * (dpi_choice ** 2)

            # Affichage des paramètres choisis
            print(f"🎛️ Paramètres sélectionnés :")
            print(f"📏 DPI                    : {dpi_choice} px/m")
            print(f"✏️ Épaisseur Traits       : {thickness_choice} px")
            print(f"🏠 Surface min pièce      : {surface_min_piece_m2} m² ({surface_min_piece_pixels:.0f} px²)")
            print(f"🧱 Épaisseur min mur      : {epaisseur_min_murale_m} m")
            print(f"📐 Epsilon approximation  : {epsilon_ratio}\n")

            # Vérification GeoJSON existant ou nettoyage
            processed_file = processed_dir / f"{file_name}_clean.geojson"
            if not processed_file.exists():
                input_geojson_file = input_dir / f"{file_name}.geojson"
                if not input_geojson_file.exists():
                    print(f"❌ Le fichier '{file_name}.geojson' est introuvable dans {input_dir}.")
                    return

                print("🧹 Nettoyage du GeoJSON...")
                pg.clean_geojson_to_segments_and_save(input_geojson_file, processed_file)

            # Chargement des segments
            segment, transform_parameters = pg.load_segments(processed_file)

            # ➡️ Étape 1 : Génération de l’image binaire
            print("🖼️ Génération de l’image binaire...")
            ccg.generate_binary_image(
                segment,
                transform_parameters,
                file_name,
                binary_images_dir,
                metadatas_dir,
                scale=dpi_choice,
                thickness=thickness_choice,
                method="gaussian"
            )

            # ➡️ Étape 2 : Détection des pièces et suppression des murs
            print("🏠 Détection des pièces et suppression des murs...")
            nb_pieces = ccg.generer_pieces_image_et_geojson(
                file_name,
                binary_images_dir,
                metadatas_dir,
                contours_images_dir,
                rooms_contours_geojson_dir,
                surface_minimale=surface_min_piece_pixels,
                dpi=dpi_choice,
                epaisseur_min_m=epaisseur_min_murale_m,
                epsilon_ratio=epsilon_ratio
            )

            # ➡️ Affichage de l’image résultat
            image_path = contours_images_dir / f"{file_name}_contours_image.png"
            if image_path.exists():
                print("✅ Analyse terminée avec succès.")
                binary_image = Image.open(image_path)
                plt.figure(figsize=(10, 10))
                plt.imshow(binary_image)
                plt.axis("off")
                plt.title(f"Nombre de pièces détectées pour {file_name} : {nb_pieces}")
                plt.show()
            else:
                print("❌ Image de contours non générée.")

    # Action du bouton
    button.on_click(on_button_clicked)

    # Affichage dans le notebook
    display(ui)