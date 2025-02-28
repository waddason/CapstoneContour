# CapstoneContour

**Git usage**: 🚧 Work on your onw branch ! 🚧 Push to main when done. ✅
```shell
git checkout -b maBranche
```
</div>


## Feuille de route 📝
1. **Nettoyage / Segmentisation**
    - [X] centrer / réduire
    - [X] normaliser le type de dessin
    - [ ] nettoyer les traits
        - [ ] Identifier / fermer les portes
        - [ ] murs communs
        - [ ] création de classe spécificque
2. **Prédire le contour**
    - [ ] clustering unsupervised
    - [X] colorier l'image, (mais retransformer en json)
    - [ ] quid des poteaux, vides sanitaires, exclusions ?
3. **Export GeoJson**
    - [X] inverse transform
    - [ ] Estimation des coordonnées depuis une image
    - [ ] bonus: label de la pièce avec ML


## Bilan des actions 📆
- **Jeudi 16 janvier** : lancement 🚀
  - présentation du projet en visio par Stéphane Maviel
  - présentation des fichiers par Jérôme Dessouter

- **Jeudi 23 janvier** : chez Diane 🏬
  - rencontre avec l'équipe de Diane sur le site de LEONARD
  - prise en main des formats de fichiers geojson et des librairies
  - Création de la **feuille de route**:

- **Jeudi 30 janvier** : 🦺
  - @Tristan Travail sur les segments : Implémentation du code du papier [Automatic Generation of Topological Indoor Maps for Real-Time Map-Based Localization and Tracking, by Martin Schäfer, Christian Knapp and Samarjit Chakraborty, 2011]
  - @Abdoul usupervised clustering-> besoin de compter le nombre de pièces
  - @Tristan Transformer le Json en graph
  - @Fabien procédure pour détecter les portes/fermer les couloirs de liaison

- **Jeudi 6 février** : conférence IA 🤖

- **Jeudi 13 février** : 👨‍💻
    - visio avec Diane: annulée
    - @Abdoul fermer les portes 🚪
    - @Fabien conserver les coordonnées après traitement d'image 🗺️
    - @Maha approfondir le clustering ፨
    - @Tristan transformer le Json en graph ⿻
    - @Tristan envoyer le 'mid-term document', cf mail Anna Korba du 26 janvier   
 
- **Jeudi 20 février** : 👨‍💻
    - 13h30: Point de situation avec Vinci -> doit nous fournir des plans complémentaires
        - @Abdoul: Segmentation à améliorer
        - @Maha: Plus de contexte pour le clustering
        - @Fabien: Faire ressembler les fichiers Vinci au dataset
        - @Tristan: Json en graph
    - 16h30: Mid term discussion avec Charles-Albert Lehalle, professeur référent sur Zoom
  
- **Jeudi 27 février**:
    - 13h30: Point de situation avec Vinci
        - @Vinci: nous transmettre des données lablelisées
        - @Fabien, essayer GPTo 👀
        - @Maha/@Tristan, poursuivre le traitement vectoriel ⿻ et préparer nouveau format
        - @Abdoul, poursuivre l'entrainement segmentation
    - GPT4 fonctionne bien avec cv.contourArea

- **Jeudi 6 mars**: 📍
    - 13h30: Point de situation avec Vinci
    - réception des donnés labellisées

- **Jeudi 13 mars**

- **Jeudi 20 mars** 📑
   -  Rapport 5 pages + annexes
-  **fin mars** : 🎤
    - Soutnance à l'école Polytechnique + zoom pour mentors
