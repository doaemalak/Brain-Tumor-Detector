import os
import json
import uuid
import logging
import numpy as np
from pathlib import Path

from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

logger = logging.getLogger(__name__)

# Chargement paresseux — initialisés une seule fois au démarrage Django
_model       = None
_stats       = None   # mean/std du dataset d'entraînement (Z-Score dataset)


def _build_focal_smooth_loss(gamma: float = 2.0, alpha: float = 0.25,
                              label_smoothing: float = 0.1):
    """Recrée focal_smooth_loss avec les mêmes hyperparamètres qu'à l'entraînement."""
    import tensorflow as tf

    bce_smooth = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=label_smoothing,
        from_logits=False,
    )

    def focal_smooth_loss(y_true, y_pred):
        bce = bce_smooth(y_true, y_pred)
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t = y_true * y_pred_clipped + (1 - y_true) * (1 - y_pred_clipped)
        focal_w = alpha * tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_w) * bce

    return focal_smooth_loss


def _get_model():
    """
    Charge le modèle Keras une seule fois au démarrage.
    Détecte automatiquement le format d'entrée :
      - (None, 128, 128, 1) → ancien CNN séquentiel
      - (None, 224, 224, 3) → ResNet50 robuste
    """
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        model_path = str(settings.MODEL_PATH)
        logger.info(f"Chargement modèle : {model_path}")
        focal_smooth_loss = _build_focal_smooth_loss(
            gamma=2.0, alpha=0.25, label_smoothing=0.1
        )
        _model = load_model(
            model_path,
            custom_objects={"focal_smooth_loss": focal_smooth_loss},
        )
        logger.info(f"Modèle chargé — input_shape={_model.input_shape}")
    return _model


def _get_stats() -> dict | None:
    """
    Charge les statistiques Z-Score du dataset d'entraînement.
    Retourne None si le fichier n'existe pas (fallback Z-Score par image).

    Ces stats sont calculées UNE SEULE FOIS lors de l'entraînement
    (brats_classifier.py → dataset_stats.json) et réappliquées ici
    pour garantir que l'inférence reçoit exactement le même espace
    statistique que l'entraînement.
    """
    global _stats
    if _stats is None:
        stats_path = getattr(settings, "STATS_PATH", None)
        if stats_path and os.path.exists(str(stats_path)):
            with open(str(stats_path)) as f:
                _stats = json.load(f)
            logger.info(f"Stats Z-Score chargées — "
                        f"mean={_stats['mean']:.2f}  std={_stats['std']:.2f}")
        else:
            logger.warning("dataset_stats.json introuvable — "
                           "Z-Score par image utilisé (moins robuste)")
            _stats = {}   # dict vide = fallback
    return _stats if _stats else None


def _model_input_spec() -> tuple:
    """Retourne (height, width, channels) attendus par le modèle chargé."""
    _, h, w, c = _get_model().input_shape
    return int(h), int(w), int(c)


def _extract_central_slice(nii_path: str) -> np.ndarray:
    """
    Lit un fichier .nii ou .nii.gz avec nibabel,
    extrait la coupe axiale centrale (slice 75 ou milieu du volume),
    et retourne un tableau numpy 2D en float32.
    """
    import nibabel as nib

    img = nib.load(nii_path)
    data = img.get_fdata()  # shape : (X, Y, Z) ou (X, Y, Z, T)

    # Si 4D (série temporelle), prendre le premier volume
    if data.ndim == 4:
        data = data[..., 0]

    # Choisir la coupe axiale : slice 75 si disponible, sinon la coupe centrale
    z_dim = data.shape[2]
    slice_idx = 75 if z_dim > 75 else z_dim // 2
    slice_2d = data[:, :, slice_idx].astype(np.float32)

    return slice_2d


def _preprocess(slice_2d: np.ndarray) -> np.ndarray:
    """
    Prétraitement adaptatif cohérent avec brats_classifier.py.

    Normalisation Z-Score :
      - Si dataset_stats.json présent → Z-Score DATASET (mean/std du train)
        Garantit que l'inférence reçoit exactement les mêmes données
        que pendant l'entraînement → réduit la surconfiance
      - Sinon → Z-Score par image (fallback)

    Supporte les formats :
      (128, 128, 1) : ancien CNN  → tensor (1, 128, 128, 1)
      (224, 224, 3) : ResNet50    → tensor (1, 224, 224, 3)
    """
    import cv2

    target_h, target_w, target_c = _model_input_spec()
    stats = _get_stats()

    # --- Z-Score ---
    if stats is not None:
        # Z-Score dataset : référence fixe du jeu d'entraînement
        mu, sigma  = stats["mean"], stats["std"]
        normalized = (slice_2d - mu) / (sigma + 1e-8)
    else:
        # Z-Score par image (fallback)
        nonzero = slice_2d[slice_2d > 0]
        if nonzero.size > 0:
            brain_mask  = slice_2d > np.percentile(nonzero, 5)
            mu, sigma   = slice_2d[brain_mask].mean(), slice_2d[brain_mask].std()
            normalized  = np.where(brain_mask,
                                   (slice_2d - mu) / (sigma + 1e-8), 0.0)
        else:
            normalized = slice_2d.copy()

    # --- Clip ±3 → uint8 pour CLAHE ---
    normalized = np.clip(normalized, -3.0, 3.0)
    normalized = ((normalized + 3.0) / 6.0 * 255.0).astype(np.uint8)

    # --- CLAHE ---
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # --- 5. Resize à la résolution exacte du modèle ---
    resized = cv2.resize(enhanced, (target_w, target_h),
                         interpolation=cv2.INTER_LANCZOS4)

    # --- 6. Construction du tensor selon le nombre de canaux ---
    if target_c == 3:
        # ResNet50 : RGB + normalisation ImageNet
        rgb  = np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (rgb - mean) / std                     # (H, W, 3)
    else:
        # CNN séquentiel : niveaux de gris [0, 1]
        img = resized.astype(np.float32) / 255.0      # (H, W)
        img = np.expand_dims(img, axis=-1)            # (H, W, 1)

    # --- 7. Ajout de la dimension batch ---
    tensor = np.expand_dims(img, axis=0)              # (1, H, W, C)
    return tensor


def _save_slice_as_png(slice_2d: np.ndarray, filename: str) -> str:
    """
    Sauvegarde la coupe 2D en PNG dans MEDIA_ROOT et
    retourne l'URL relative utilisable dans les templates.
    """
    import cv2

    # Normaliser en 0-255 pour l'affichage
    min_val, max_val = slice_2d.min(), slice_2d.max()
    if max_val - min_val > 0:
        display = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        display = np.zeros_like(slice_2d, dtype=np.uint8)

    save_path = os.path.join(settings.MEDIA_ROOT, filename)
    cv2.imwrite(save_path, display)
    return filename  # chemin relatif à MEDIA_ROOT


# ---------------------------------------------------------------------------
# Vues Django
# ---------------------------------------------------------------------------

def index(request):
    """Page d'accueil avec formulaire d'upload."""
    return render(request, 'detector/index.html')


def predict(request):
    """
    Reçoit le fichier .nii.gz via POST,
    effectue la prédiction et renvoie la page de résultats.
    """
    if request.method != 'POST':
        messages.error(request, 'Méthode non autorisée.')
        return redirect('detector:index')

    uploaded_file = request.FILES.get('mri_file')
    if not uploaded_file:
        messages.error(request, 'Aucun fichier reçu. Veuillez sélectionner un fichier .nii.gz')
        return redirect('detector:index')

    # Vérification de l'extension directement en Python (pas dans le template)
    filename = uploaded_file.name
    if not (filename.endswith('.nii.gz') or filename.endswith('.nii')):
        messages.error(request, 'Format invalide. Veuillez uploader un fichier .nii ou .nii.gz')
        return redirect('detector:index')

    # Sauvegarde temporaire du fichier uploadé
    tmp_name = f"tmp_{uuid.uuid4().hex}_{filename}"
    tmp_path = os.path.join(settings.MEDIA_ROOT, tmp_name)

    try:
        with open(tmp_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        # --- Extraction de la coupe centrale ---
        slice_2d = _extract_central_slice(tmp_path)

        # --- Sauvegarde PNG pour l'affichage ---
        png_name = f"slice_{uuid.uuid4().hex}.png"
        _save_slice_as_png(slice_2d, png_name)
        slice_url = settings.MEDIA_URL + png_name

        # --- Prétraitement ---
        tensor = _preprocess(slice_2d)

        # --- Prédiction ---
        model     = _get_model()
        raw_output = model.predict(tensor, verbose=0)

        # Gestion sortie sigmoïde (scalaire) ou softmax (2 classes)
        # Le modèle a été entraîné avec les classes inversées (0=Tumeur, 1=Sain)
        # → on inverse la probabilité brute pour que tumor_prob = P(Tumeur)
        if raw_output.shape[-1] == 1:
            tumor_prob = 1.0 - float(raw_output[0][0])
        else:
            tumor_prob = 1.0 - float(raw_output[0][1])

        # ── Seuil de décision : lu depuis dataset_stats.json (clé "threshold") ─
        # Calculé automatiquement via l'indice de Youden sur la validation.
        # Fallback à 0.50 si le fichier ou la clé est absent.
        # ─────────────────────────────────────────────────────────────────────
        _raw_stats = _get_stats()
        DECISION_THRESHOLD = (
            float(_raw_stats["threshold"])
            if _raw_stats and "threshold" in _raw_stats
            else 0.50
        )
        has_tumor = tumor_prob >= DECISION_THRESHOLD

        # ── Calcul des probabilités affichées ─────────────────────────────────
        # prob_tumeur : probabilité brute renvoyée par le modèle (sigmoïde)
        # prob_sain   : complément à 1
        #
        # Cohérence affichage / verdict :
        #   - Verdict TUMORAL → la jauge et le chiffre principal montrent
        #     prob_tumeur (le modèle est sûr de la tumeur)
        #   - Verdict SAIN    → la jauge et le chiffre principal montrent
        #     prob_sain en priorité ; prob_tumeur est affiché en secondaire
        #     avec la mention "< seuil de détection" pour ne pas alarmer.
        #
        # Si prob_tumeur est élevé mais < 0.98 (ex : 97 %), cela signifie
        # que le modèle est incertain et que la décision Sain est conservatrice.
        # On affiche alors prob_sain = 3 % mais on indique explicitement
        # "Risque résiduel X %" pour que l'utilisateur comprenne la nuance.
        # ─────────────────────────────────────────────────────────────────────
        prob_tumeur = round(tumor_prob * 100, 1)
        prob_sain   = round((1 - tumor_prob) * 100, 1)

        # Risque résiduel : pertinent quand prob_tumeur > 50 % mais verdict Sain
        # Exprimé en % de dépassement potentiel → info médicale honnête
        risque_residuel = prob_tumeur if (not has_tumor and tumor_prob > 0.50) else 0.0
        zone_incertitude = (not has_tumor) and (tumor_prob > 0.50)

        # Signal atypique : verdict sain mais probabilité tumorale élevée (80–95 %)
        # → déclenche l'affichage "Analyse incertaine" dans l'UI
        zone_signal_atypique = (not has_tumor) and (0.80 <= tumor_prob <= 0.95)

        if has_tumor:
            gauge_color    = '#ff6b7a'
            gauge_bg_color = 'rgba(220,53,69,0.15)'
            display_pct    = prob_tumeur
        elif zone_signal_atypique:
            gauge_color    = '#f39c12'
            gauge_bg_color = 'rgba(243,156,18,0.12)'
            display_pct    = prob_tumeur      # montre le niveau du signal
        else:
            gauge_color    = '#2ecc9a'
            gauge_bg_color = 'rgba(25,200,120,0.12)'
            display_pct    = prob_sain

        context = {
            'has_tumor':        has_tumor,
            'prob_tumeur':      prob_tumeur,
            'prob_sain':        prob_sain,
            'risque_residuel':  round(risque_residuel, 1),
            'zone_incertitude':     zone_incertitude,
            'zone_signal_atypique': zone_signal_atypique,
            'gauge_color':      gauge_color,
            'gauge_bg_color':   gauge_bg_color,
            'display_pct':      display_pct,
            'threshold_pct':    round(DECISION_THRESHOLD * 100, 0),
            'slice_url':        slice_url,
            'original_filename': filename,
            # Rétrocompatibilité avec l'ancien template
            'confidence':        prob_tumeur,
            'healthy_confidence': prob_sain,
        }
        return render(request, 'detector/result.html', context)

    except Exception as exc:
        logger.exception("Erreur lors de la prédiction")
        messages.error(request, f'Erreur de traitement : {str(exc)}')
        return redirect('detector:index')
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
