"""
BraTS Brain Tumor Classifier — Script Final Kaggle
====================================================
Corrections anti-surconfiance :
  1. Focal Loss  (gamma=2, alpha=0.25)
  2. Label Smoothing  (epsilon=0.1)  → empêche 0.97/0.99 extrêmes
  3. ReduceLROnPlateau  → affine les poids sans tout écraser
  4. Z-Score DATASET  → moyenne/std calculées sur l'ensemble train,
                        sauvegardées et réappliquées à val/test/inférence

Usage Kaggle :
  1. Ajustez TUMOR_DIR, HEALTHY_DIR
  2. python brats_classifier.py
  3. Récupérez brats_final.keras + dataset_stats.json
  4. Mettez à jour MODEL_PATH dans settings.py
"""

import os
import json
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE   = 224
CHANNELS   = 3
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-4
THRESHOLD  = 0.50   # remplacé en fin d'entraînement par le seuil optimal sur la validation
L2         = 1e-3

DATA_ROOT    = "./data"
TUMOR_DIR    = os.path.join(DATA_ROOT, "tumor")
HEALTHY_DIR  = os.path.join(DATA_ROOT, "healthy")
MODEL_SAVE   = "brats_final.keras"
STATS_FILE   = "dataset_stats.json"   # mean/std du dataset d'entraînement

# Kaggle → décommenter
# TUMOR_DIR   = "/kaggle/input/brats2021/tumor"
# HEALTHY_DIR = "/kaggle/input/healthy-brains/healthy"
# MODEL_SAVE  = "/kaggle/working/brats_final.keras"
# STATS_FILE  = "/kaggle/working/dataset_stats.json"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NORMALISATION Z-SCORE DATASET (la plus robuste)
# ══════════════════════════════════════════════════════════════════════════════

def extract_raw_slice(nii_path: str, slice_idx: int) -> np.ndarray:
    """Charge une coupe 2D brute float32 depuis un fichier .nii/.nii.gz."""
    vol = nib.load(nii_path).get_fdata().astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    z = min(slice_idx, vol.shape[2] - 1)
    return vol[:, :, z]


def compute_dataset_stats(paths: np.ndarray, slices: np.ndarray,
                           save_path: str = STATS_FILE) -> dict:
    """
    Calcule la moyenne et l'écart-type sur TOUTES les coupes du dataset
    d'entraînement.

    Pourquoi cette approche est supérieure au Z-score par image :
      - Z-score par image : chaque image est normalisée selon SES propres
        statistiques → une image avec peu de tissu ou beaucoup de bruit
        peut être artificiellement amplifiée.
      - Z-score dataset : toutes les images partagent la même référence
        → le modèle reçoit toujours des données dans le même espace
        statistique, ce qui réduit la surconfiance sur des images atypiques.

    Les statistiques sont sauvegardées dans un fichier JSON pour être
    réappliquées identiquement sur val, test et inférence en production.
    """
    print("Calcul des statistiques du dataset (peut prendre quelques minutes)...")
    all_pixels = []

    for i, (path, z) in enumerate(zip(paths, slices)):
        s = extract_raw_slice(path, int(z))
        # On ne prend que les pixels cerveau (> 0) pour ne pas biaiser
        brain_px = s[s > 0]
        if brain_px.size > 0:
            all_pixels.append(brain_px)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)} fichiers traités...")

    all_pixels = np.concatenate(all_pixels)
    stats = {
        "mean": float(np.mean(all_pixels)),
        "std":  float(np.std(all_pixels)),
        "n_images": len(paths),
    }

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats dataset → mean={stats['mean']:.2f}  std={stats['std']:.2f}")
    print(f"Sauvegardées : {save_path}")
    return stats


def load_dataset_stats(stats_path: str = STATS_FILE) -> dict:
    """Charge les statistiques pré-calculées depuis le fichier JSON."""
    with open(stats_path) as f:
        return json.load(f)


def preprocess_slice(nii_path: str, slice_idx: int = 75,
                     stats: dict = None) -> np.ndarray:
    """
    Pipeline de prétraitement → float32 (224, 224, 3) normalisé ImageNet.

    Étapes :
      1. Extraction de la coupe brute
      2. Z-Score :
           - Si stats fourni  → Z-Score DATASET (mean/std du train)
           - Sinon            → Z-Score IMAGE   (fallback par image)
      3. Clip ±3 → rescale [0, 255]
      4. CLAHE (clipLimit=2, tileGridSize=8×8)
      5. Resize 224×224 (Lanczos)
      6. Grayscale → RGB
      7. Normalisation ImageNet

    En production (views.py), stats est chargé UNE SEULE FOIS au démarrage
    depuis dataset_stats.json pour garantir la cohérence train/inférence.
    """
    s = extract_raw_slice(nii_path, slice_idx)

    # ── 2. Z-Score ────────────────────────────────────────────────────────────
    if stats is not None:
        # Z-Score dataset : référence fixe issue du jeu d'entraînement
        mu    = stats["mean"]
        sigma = stats["std"]
    else:
        # Z-Score image : fallback si pas de stats (ne pas utiliser en prod)
        nonzero = s[s > 0]
        mu      = nonzero.mean() if nonzero.size > 0 else 0.0
        sigma   = nonzero.std()  if nonzero.size > 0 else 1.0

    normed = (s - mu) / (sigma + 1e-8)

    # ── 3. Clip ±3 → uint8 ───────────────────────────────────────────────────
    normed = np.clip(normed, -3.0, 3.0)
    normed = ((normed + 3.0) / 6.0 * 255.0).astype(np.uint8)

    # ── 4. CLAHE ──────────────────────────────────────────────────────────────
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normed = clahe.apply(normed)

    # ── 5. Resize 224×224 ─────────────────────────────────────────────────────
    resized = cv2.resize(normed, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)

    # ── 6-7. Grayscale → RGB + normalisation ImageNet ─────────────────────────
    rgb  = np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb - mean) / std


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GÉNÉRATEUR
# ══════════════════════════════════════════════════════════════════════════════

class BraTSGenerator(tf.keras.utils.Sequence):
    """
    Génère des batchs (X, y) pour l'entraînement.
    Accepte les statistiques dataset pour la normalisation cohérente.
    """

    def __init__(self, paths, labels, slices,
                 stats=None, batch_size=BATCH_SIZE, training=True):
        self.paths      = paths
        self.labels     = labels
        self.slices     = slices
        self.stats      = stats       # dict mean/std du dataset train
        self.batch_size = batch_size
        self.training   = training
        self.indices    = np.arange(len(paths))
        if self.training:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        b = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(b), IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.float32)
        y = np.zeros(len(b), dtype=np.float32)
        for i, j in enumerate(b):
            X[i] = preprocess_slice(self.paths[j], int(self.slices[j]),
                                    stats=self.stats)
            y[i] = self.labels[j]
        return X, y

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOSS : FOCAL + LABEL SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def build_loss(gamma: float = 2.0, alpha: float = 0.25,
               label_smoothing: float = 0.1):
    """
    Construit la fonction de perte combinée :

    Focal Loss (gamma=2, alpha=0.25) :
      - Réduit le poids des exemples FACILES (cerveau clairement sain
        ou tumeur évidente) → le gradient se concentre sur les cas
        difficiles (coupes ambiguës)
      - Résultat : le modèle cesse de sur-apprendre les cas triviaux

    Label Smoothing (epsilon=0.1) :
      - Transforme les labels binaires {0, 1} en {0.05, 0.95}
      - Empêche le modèle de viser des probabilités extrêmes
        comme 0.97 ou 0.003
      - Effet direct : les sorties softmax/sigmoïde restent dans
        une plage plus raisonnable → moins de surconfiance
      - Sans label smoothing : le modèle peut atteindre 0.97 sur du sain
      - Avec label smoothing : la cible max est 0.95, donc 0.97 est pénalisé
    """
    bce_smooth = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=label_smoothing,
        from_logits=False
    )

    def focal_smooth_loss(y_true, y_pred):
        # Applique le label smoothing via BCE
        bce = bce_smooth(y_true, y_pred)

        # Pondération focal : (1 - p_t)^gamma
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t     = y_true * y_pred_clipped + (1 - y_true) * (1 - y_pred_clipped)
        focal_w = alpha * tf.pow(1.0 - p_t, gamma)

        return tf.reduce_mean(focal_w) * bce

    return focal_smooth_loss


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)) -> Model:
    """
    ResNet50 (ImageNet) avec :
      - Augmentation intégrée (RandomFlip, RandomRotation)
      - GlobalAveragePooling2D
      - Dense + L2 + BatchNorm + Dropout(0.5)
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)

    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=input_shape)
    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers[-20:]:
        layer.trainable = True

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)

    output = layers.Dense(1, activation="sigmoid", name="tumor_prob")(x)
    return Model(inputs=inputs, outputs=output, name="ResNet50_FocalSmooth")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train(model: Model, train_gen, val_gen, y_train: np.ndarray) -> dict:
    """
    Entraîne avec :
      - Focal Loss + Label Smoothing
      - Class weights (plus d'attention sur la classe minoritaire)
      - EarlyStopping patience=5 sur val_loss
      - ReduceLROnPlateau : divise LR par 2 si val_loss stagne 3 epochs
    """
    # Poids asymétrique : pénalise 2× plus les faux négatifs (tumeurs ratées)
    # Préférable à "balanced" pour maximiser la sensibilité (Recall tumoral)
    class_weights = {0: 1.0, 1: 2.0}
    print(f"\nClass weights → Sain: {class_weights[0]:.1f}  Tumoral: {class_weights[1]:.1f}")

    loss_fn = build_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        # ReduceLROnPlateau : affine les poids sans tout écraser
        # Si val_loss ne s'améliore pas pendant 3 epochs → LR × 0.5
        # Permet de sortir des plateaux sans diverger
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    return history.history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model: Model, test_gen, threshold: float = None,
             save_path="evaluation.png"):
    y_true, y_probs = [], []
    for Xb, yb in test_gen:
        y_probs.extend(model.predict(Xb, verbose=0).flatten())
        y_true.extend(yb)

    y_true = np.array(y_true)
    t      = threshold if threshold is not None else THRESHOLD
    y_pred = (np.array(y_probs) >= t).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*44}")
    print(f"  Seuil utilise : {t:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  Specificite  : {tn/(tn+fp+1e-8):.3f}")
    print(f"  Sensibilite  : {tp/(tp+fn+1e-8):.3f}")
    print(f"{'='*44}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Sain", "Tumoral"]))

    sain_p  = np.array(y_probs)[y_true == 0]
    tumor_p = np.array(y_probs)[y_true == 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(cm, cmap=plt.cm.Blues)
    axes[0].set_title(f"Matrice de Confusion (seuil={t:.2f})")
    lbl = [["TN","FP"],["FN","TP"]]
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{lbl[i][j]}\n{cm[i,j]}",
                         ha="center", va="center", fontsize=12,
                         color="white" if cm[i,j] > cm.max()/2 else "black")
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(["Predit Sain","Predit Tumoral"])
    axes[0].set_yticklabels(["Reel Sain","Reel Tumoral"])

    axes[1].hist(sain_p,  bins=30, alpha=0.65, color="#2ecc9a", label="Sain")
    axes[1].hist(tumor_p, bins=30, alpha=0.65, color="#ff6b7a", label="Tumoral")
    axes[1].axvline(t, color="black", linestyle="--",
                    linewidth=2, label=f"Seuil={t:.2f}")
    axes[1].axvspan(0.5, t, alpha=0.07, color="orange", label="Zone grise")
    axes[1].set_xlabel("P(Tumoral)")
    axes[1].set_title("Distribution — un bon modele evite les pics extremes")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Evaluation sauvegardee : {save_path}")
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def plot_history(history: dict, save_path="training.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"],     label="Train Loss", color="#00b4d8")
    axes[0].plot(history["val_loss"], label="Val Loss",   color="#ff6b7a")
    axes[0].set_title("Loss (Focal + Label Smoothing)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["auc"],     label="Train AUC", color="#00b4d8")
    axes[1].plot(history["val_auc"], label="Val AUC",   color="#ff6b7a")
    axes[1].set_title("AUC ROC")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Courbes sauvegardees : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6b — SEUIL OPTIMAL (Youden / F1)
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(model: Model, val_gen,
                            method: str = "youden",
                            stats_path: str = STATS_FILE) -> float:
    """
    Calcule le seuil de décision optimal sur le set de validation.

    method='youden'  → argmax(sensitivity + specificity - 1)
      Maximise la détection des vrais positifs tout en limitant les
      faux positifs — idéal pour les applications médicales.

    method='f1'      → argmax(F1-score tumoral)
      Favorise le rappel tumoral si le dataset est très déséquilibré.

    Le seuil est clamped dans [0.25, 0.75] pour rester cliniquement
    raisonnable, puis sauvegardé dans dataset_stats.json.
    """
    y_true, y_probs = [], []
    for Xb, yb in val_gen:
        y_probs.extend(model.predict(Xb, verbose=0).flatten())
        y_true.extend(yb)
    y_true  = np.array(y_true)
    y_probs = np.array(y_probs)

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        youden    = tpr - fpr
        best_idx  = int(np.argmax(youden))
        best_t    = float(thresholds[best_idx])
        best_score = float(youden[best_idx])
        print(f"\n[Youden] Seuil optimal = {best_t:.4f}  (J = {best_score:.4f})")
    else:
        candidates = np.linspace(0.10, 0.95, 300)
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            score = f1_score(y_true, (y_probs >= t).astype(int),
                             zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, float(t)
        print(f"\n[F1]     Seuil optimal = {best_t:.4f}  (F1 = {best_f1:.4f})")

    # Clamp : évite les seuils extrêmes qui rendraient le modèle inutilisable
    best_t = float(np.clip(best_t, 0.25, 0.75))
    print(f"         Seuil retenu (clamped [0.25, 0.75]) = {best_t:.4f}")

    # Persistance dans dataset_stats.json (lue par views.py à chaque démarrage)
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    stats["threshold"] = best_t
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"         Sauvegarde : {stats_path}  (cle 'threshold' = {best_t:.4f})")

    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — COLLECTE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples_by_slice(brats_dir: str,
                              slice_range: range = range(50, 130)):
    """
    Mode BraTS complet : labellise CHAQUE COUPE via le masque expert.
    Label 1 = coupe avec pixels tumoraux, Label 0 = coupe vide (saine).
    """
    paths, labels, slices = [], [], []
    for case in sorted(os.listdir(brats_dir)):
        case_dir = os.path.join(brats_dir, case)
        if not os.path.isdir(case_dir):
            continue
        flair = os.path.join(case_dir, f"{case}_flair.nii.gz")
        seg   = os.path.join(case_dir, f"{case}_seg.nii.gz")
        if not (os.path.exists(flair) and os.path.exists(seg)):
            continue
        seg_vol = nib.load(seg).get_fdata()
        for z in slice_range:
            z_  = min(z, seg_vol.shape[2] - 1)
            lbl = 1 if seg_vol[:, :, z_].sum() > 0 else 0
            paths.append(flair)
            labels.append(lbl)
            slices.append(z)
    return np.array(paths), np.array(labels, dtype=np.float32), np.array(slices)


def collect_samples_simple(tumor_dir: str, healthy_dir: str,
                            slice_idx: int = 75):
    """Mode simple : un fichier = une coupe fixe."""
    paths, labels, slices = [], [], []
    for f in sorted(os.listdir(tumor_dir)):
        if f.endswith(".nii.gz") or f.endswith(".nii"):
            paths.append(os.path.join(tumor_dir, f))
            labels.append(1); slices.append(slice_idx)
    for f in sorted(os.listdir(healthy_dir)):
        if f.endswith(".nii.gz") or f.endswith(".nii"):
            paths.append(os.path.join(healthy_dir, f))
            labels.append(0); slices.append(slice_idx)
    return np.array(paths), np.array(labels, dtype=np.float32), np.array(slices)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    # ── 1. Collecte ──────────────────────────────────────────────────────────
    # Mode A (BraTS, labellisation par masque) :
    #   paths, labels, slices = collect_samples_by_slice(
    #       "/kaggle/input/brats2021/training", slice_range=range(50, 130)
    #   )
    # Mode B (simple) :
    paths, labels, slices = collect_samples_simple(
        TUMOR_DIR, HEALTHY_DIR, slice_idx=75
    )

    n_sain    = int((labels == 0).sum())
    n_tumoral = int((labels == 1).sum())
    print(f"\nDataset : {len(paths)} echantillons  "
          f"(Sain={n_sain}, Tumoral={n_tumoral})")

    if len(paths) < 10:
        print("ERREUR : moins de 10 fichiers trouves.")
        return

    # ── 2. Split 70 / 15 / 15 ────────────────────────────────────────────────
    idx  = np.random.permutation(len(paths))
    n_tr = int(0.70 * len(paths))
    n_va = int(0.15 * len(paths))

    i_tr = idx[:n_tr]
    i_va = idx[n_tr:n_tr + n_va]
    i_te = idx[n_tr + n_va:]

    # ── 3. Z-Score dataset sur le TRAIN uniquement ───────────────────────────
    # Les stats sont calculées sur le train puis appliquées à val/test/inférence
    # → garantit que val et test voient exactement le même espace statistique
    if os.path.exists(STATS_FILE):
        print(f"\nStats trouvees : {STATS_FILE} (rechargement)")
        stats = load_dataset_stats(STATS_FILE)
    else:
        stats = compute_dataset_stats(paths[i_tr], slices[i_tr],
                                      save_path=STATS_FILE)

    print(f"Z-Score dataset → mean={stats['mean']:.2f}  std={stats['std']:.2f}")

    # ── 4. Générateurs ───────────────────────────────────────────────────────
    train_gen = BraTSGenerator(paths[i_tr], labels[i_tr], slices[i_tr],
                                stats=stats, training=True)
    val_gen   = BraTSGenerator(paths[i_va], labels[i_va], slices[i_va],
                                stats=stats, training=False)
    test_gen  = BraTSGenerator(paths[i_te], labels[i_te], slices[i_te],
                                stats=stats, training=False)

    # ── 5. Modèle + entraînement ──────────────────────────────────────────────
    model = build_model()
    model.summary()

    history = train(model, train_gen, val_gen, labels[i_tr])
    plot_history(history)

    # ── 6. Seuil optimal (Youden sur la validation) ───────────────────────────
    print("\nRecherche du seuil optimal sur le set de validation...")
    best_threshold = find_optimal_threshold(
        model, val_gen, method="youden", stats_path=STATS_FILE
    )

    # ── 7. Évaluation avec le seuil optimal ───────────────────────────────────
    print("\nEvaluation Test Set :")
    evaluate(model, test_gen, threshold=best_threshold)

    # ── 7. Instructions post-entraînement ─────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"Fichiers a recuperer depuis Kaggle :")
    print(f"  {MODEL_SAVE}   <- modele Keras")
    print(f"  {STATS_FILE}   <- stats Z-Score du dataset")
    print(f"\nDans settings.py :")
    print(f"  MODEL_PATH  = BASE_DIR / '{MODEL_SAVE}'")
    print(f"  STATS_PATH  = BASE_DIR / '{STATS_FILE}'")
    print(f"\nDans views.py (deja configure) :")
    print(f"  stats = load_dataset_stats(settings.STATS_PATH)")
    print(f"  tensor = preprocess_slice(path, z, stats=stats)")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
