# Brain Tumor Detector — BraTS MRI Classifier

Application web de détection de tumeurs cérébrales par deep learning, basée sur des IRM au format NIfTI (`.nii` / `.nii.gz`). Le modèle, entraîné sur le dataset **BraTS 2021** (Kaggle), est un **ResNet50 fine-tuné** avec des techniques anti-surconfiance (Focal Loss, Label Smoothing, Z-Score dataset).

---

## Technologies utilisées

| Couche | Outil |
|---|---|
| Framework web | Django 4.2 |
| Deep Learning | TensorFlow / Keras 2.12+ |
| Architecture modèle | ResNet50 (transfer learning) |
| Lecture IRM | NiBabel 5 |
| Traitement image | OpenCV 4.8 |
| Calcul numérique | NumPy 1.24+ |
| Entraînement | Script Python sur Kaggle |
| Langue | Python 3.11 |

---

## Fonctionnalités principales

- **Upload IRM** — Accepte les fichiers `.nii` et `.nii.gz` (jusqu'à 50 Mo)
- **Extraction automatique** — Sélectionne la coupe axiale centrale (slice 75 ou milieu du volume)
- **Prétraitement adaptatif** — Z-Score dataset + CLAHE + redimensionnement selon le format d'entrée du modèle (128×128 ou 224×224)
- **Prédiction calibrée** — Seuil de décision optimal calculé via l'indice de Youden (sauvegardé dans `dataset_stats.json`)
- **Affichage visuel** — Jauge de probabilité colorée (rouge = tumeur, vert = sain, orange = zone incertaine)
- **Gestion des zones atypiques** — Signalement explicite quand la probabilité tumorale est élevée mais sous le seuil (80–95 %)
- **Interface bilingue** — Interface en français, code commenté en français/anglais

---

## Installation et lancement

### Prérequis

- Python 3.11
- pip

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/brain-tumor-detector.git
cd brain-tumor-detector
```

### 2. Créer et activer l'environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Ajouter le modèle entraîné

Placez les fichiers suivants à la racine du projet :

| Fichier | Rôle |
|---|---|
| `brats_final.keras` | Poids du modèle ResNet50 entraîné |
| `dataset_stats.json` | Statistiques Z-Score + seuil de décision |

> Ces fichiers ne sont pas versionnés (trop lourds pour Git). Téléchargez-les depuis la [release](https://github.com/<votre-username>/brain-tumor-detector/releases) ou régénérez-les avec `brats_classifier.py`.

### 5. Lancer le serveur de développement

```bash
python manage.py runserver
```

L'application est accessible sur `http://127.0.0.1:8000/`.

### 6. Ré-entraîner le modèle (optionnel)

```bash
# Configurez TUMOR_DIR et HEALTHY_DIR dans brats_classifier.py
python brats_classifier.py
# → génère brats_final.keras + dataset_stats.json
```

---

## Structure du projet

```
brain-tumor-detector/
│
├── brain_tumor_project/        # Configuration Django
│   ├── settings.py             # Paramètres (MODEL_PATH, STATS_PATH, upload max)
│   ├── urls.py                 # Routage principal
│   └── wsgi.py
│
├── detector/                   # Application Django principale
│   ├── views.py                # Logique de prédiction (extraction, prétraitement, inférence)
│   ├── urls.py                 # Routes /  et /predict/
│   └── templates/
│       └── detector/
│           ├── index.html      # Page d'accueil — formulaire d'upload
│           └── result.html     # Page de résultats — jauge + image IRM
│
├── static/                     # Fichiers statiques (CSS, JS, images)
├── media/                      # Coupes IRM générées à la volée (non versionnées)
│
├── brats_classifier.py         # Script d'entraînement du modèle (Kaggle)
├── evaluation_kaggle.py        # Script d'évaluation (métriques, courbe ROC)
├── dataset_stats.json          # Statistiques Z-Score + seuil optimal (non versionné)
├── requirements.txt            # Dépendances Python
└── manage.py                   # Point d'entrée Django
```

---

## Modèle — détails techniques

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 + couche Dense(1, sigmoid) |
| Entrée | 224 × 224 × 3 (niveaux de gris dupliqués en RGB) |
| Fonction de perte | Focal Loss (γ=2, α=0.25) + Label Smoothing (ε=0.1) |
| Optimiseur | Adam (lr=1e-4) + ReduceLROnPlateau |
| Normalisation | Z-Score dataset (mean/std du train) + CLAHE |
| Seuil de décision | Calculé via indice de Youden sur la validation |
| Dataset | BraTS 2021 (Kaggle) |

---

## Notes importantes

- `SECRET_KEY` dans `settings.py` doit être changée avant tout déploiement en production.
- `DEBUG = True` est actif par défaut — à désactiver en production.
- Les fichiers `.keras` et `.h5` sont exclus du dépôt (`.gitignore`) en raison de leur taille.

---

## Licence

Ce projet est réalisé dans un cadre académique (projet de fin d'études). Utilisation à des fins éducatives uniquement.
