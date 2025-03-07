"""Models predicting polygons from GeoJson."""

from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from utils.parse_geojson import extract_segments


# Baseline model
class predict_poly:
    """Find rooms in geojson file using basic geometric rules."""

    def __call__(
        self,
        x: shapely.GeometryCollection,
    ) -> shapely.GeometryCollection:
        return self.predict_poly(x)

    def predict_poly(
        x: shapely.GeometryCollection,
    ) -> shapely.GeometryCollection:
        """Return the well defined polygons of the geometry collection."""
        poly, cut_edges, dangles, invalid = shapely.polygonize_full(x.geoms)

        return poly


# Made by Maha, from the segments


class SegmentBasedClustering:
    """Find rooms in geojson file using geometric rules."""

    def __init__(self, min_room_area=1.0, max_room_area=1000.0):
        self.min_room_area = min_room_area
        self.max_room_area = max_room_area
        self.segments = None
        self.rooms = None
        self.__name__ = "SegmentBasedClustering"

    def find_closed_paths(self):
        """Identifie les chemins fermés formés par les segments."""
        lines = unary_union(self.segments)
        potential_rooms = list(polygonize(lines))

        # Convertir les MultiPolygon en liste de Polygon
        processed_rooms = []
        for room in potential_rooms:
            if isinstance(room, MultiPolygon):
                processed_rooms.extend(list(room.geoms))
            else:
                processed_rooms.append(room)

        return processed_rooms

    def filter_rooms(self, min_area=1.0, max_area=1000.0):
        """Filtre les polygones selon des critères de taille."""
        valid_rooms = []
        for room in self.rooms:
            area = room.area
            if min_area <= area <= max_area:
                valid_rooms.append(room)
        return valid_rooms

    def find_adjacent_rooms(self):
        """Identifie les pièces adjacentes."""
        n_rooms = len(self.rooms)
        adjacency_matrix = np.zeros((n_rooms, n_rooms), dtype=bool)

        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                if self.rooms[i].touches(self.rooms[j]):
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = True

        return adjacency_matrix

    def merge_small_rooms(self, min_area=5.0):
        """Fusionne les petites pièces avec leurs voisines."""
        modified = True
        while modified:
            modified = False

            # Identification des petites pièces
            small_rooms = []
            for i, room in enumerate(self.rooms):
                if room.area < min_area:
                    small_rooms.append(i)

            if not small_rooms:
                break

            adjacency = self.find_adjacent_rooms()

            # Traitement de chaque petite pièce
            for small_room_idx in small_rooms:
                if small_room_idx >= len(self.rooms):
                    continue

                neighbors = [
                    i
                    for i in range(len(self.rooms))
                    if i != small_room_idx and adjacency[small_room_idx][i]
                ]

                if not neighbors:
                    continue

                best_neighbor = max(
                    neighbors,
                    key=lambda x: self.rooms[x].area
                    if x < len(self.rooms)
                    else 0,
                )

                if best_neighbor >= len(self.rooms):
                    continue

                # Fusionne les pièces
                merged = unary_union(
                    [self.rooms[best_neighbor], self.rooms[small_room_idx]]
                )

                # Traite le cas où la fusion produit un MultiPolygon
                if isinstance(merged, MultiPolygon):
                    largest_poly = max(merged.geoms, key=lambda p: p.area)
                    merged = largest_poly

                # Met à jour la liste des pièces
                new_rooms = []
                for i in range(len(self.rooms)):
                    if i == best_neighbor:
                        new_rooms.append(merged)
                    elif i != small_room_idx:
                        new_rooms.append(self.rooms[i])

                self.rooms = new_rooms
                modified = True
                break

    def fit(self, X, y):
        """Exécute l'algorithme complet de détection des pièces."""
        self.segments = X
        self.rooms = self.find_closed_paths()

        if not self.rooms:
            print("Aucune pièce fermée n'a été trouvée")
            return []

        self.rooms = self.filter_rooms(self.min_room_area, self.max_room_area)

        if not self.rooms:
            print("Aucune pièce ne correspond aux critères de taille")
            return []

        self.merge_small_rooms()
        return self.rooms

    def predict(
        self, geometry_collection: shapely.GeometryCollection
    ) -> shapely.GeometryCollection:
        """Exécute l'algorithme complet de détection des pièces."""
        self.segments = extract_segments(geometry_collection)
        self.rooms = self.find_closed_paths()

        if not self.rooms:
            print("SBC: Aucune pièce fermée n'a été trouvée")
            return None

        self.rooms = self.filter_rooms(self.min_room_area, self.max_room_area)

        if not self.rooms:
            print("SBC: Aucune pièce ne correspond aux critères de taille")
            return None

        self.merge_small_rooms()
        return shapely.GeometryCollection(self.rooms)

    def __call__(
        self,
        geometry_collection: shapely.GeometryCollection,
    ) -> shapely.GeometryCollection:
        """Predict polygons from the geometry collection."""
        return self.predict(geometry_collection)
