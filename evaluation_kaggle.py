"""
evaluation_kaggle.py — BrainScan AI
=====================================
Script d'évaluation finale pour la soutenance PFE.
Modèle : ResNet50 — Dataset : BraTS 2021 — Loss : Focal + Label Smoothing

─── Utilisation ───────────────────────────────────────────────────────────
Coller ces lignes APRÈS model.fit() dans votre notebook Kaggle :

    from evaluation_kaggle import (
        plot_learning_curves,
        evaluate_model,
        plot_confusion_matrix,
    )

    plot_learning_curves(history)

    y_true, y_probs = evaluate_model(model, test_gen)

    plot_confusion_matrix(y_true, y_probs)

─── Fichiers générés ──────────────────────────────────────────────────────
    courbes_apprentissage.png   →  Slide 5 / rapport Overleaf
    matrice_confusion.png       →  Slide 5 / rapport Overleaf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # pas d'affichage GUI sur Kaggle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

# ──────────────────────────────────────────────────────────────
#  CONSTANTES — palette BrainScan AI
# ──────────────────────────────────────────────────────────────
TRAIN_COLOR  = "#00b4d8"   # bleu cyan  — courbes d'entraînement
VAL_COLOR    = "#ff6b7a"   # corail     — courbes de validation
BEST_COLOR   = "#2ecc71"   # vert       — meilleure époque (★)
ANNOT_COLORS = {           # annotations de la matrice de confusion
    "VP": "#1e8449",       # vert foncé  — Vrai Positif
    "VN": "#1a5276",       # bleu foncé  — Vrai Négatif
    "FP": "#922b21",       # rouge       — Faux Positif
    "FN": "#784212",       # marron      — Faux Négatif
}
BG_COLOR     = "#fafbfc"   # fond clair
GRAY         = "#4b555f"   # texte secondaire

THRESHOLD    = 0.98        # seuil de décision médical


# ══════════════════════════════════════════════════════════════
#  1. COURBES D'APPRENTISSAGE
# ══════════════════════════════════════════════════════════════

def plot_learning_curves(
    history,
    save_path: str = "courbes_apprentissage.png",
    dpi: int = 300,
) -> None:
    """
    Trace les courbes d'apprentissage (2 panneaux).

    Panneau gauche  →  Métrique de performance (AUC ou Accuracy)
    Panneau droit   →  Perte (Focal Loss)

    Paramètres
    ----------
    history   : objet Keras History  OU  dict history.history
    save_path : nom du fichier PNG   (défaut : courbes_apprentissage.png)
    dpi       : résolution d'export  (défaut : 300 — requis pour Overleaf)
    """
    # ── Normalisation : accepte les deux formes ────────────
    h = history.history if hasattr(history, "history") else history

    # ── Garde : historique valide ──────────────────────────
    if not isinstance(h, dict) or not h:
        raise ValueError(
            "[plot_learning_curves] Historique vide ou invalide.\n"
            "Vérifiez que model.fit() a bien retourné un objet History."
        )
    if "loss" not in h:
        raise ValueError(
            f"[plot_learning_curves] Clé 'loss' absente.\n"
            f"Clés disponibles : {list(h.keys())}"
        )

    # ── Détection automatique de la métrique principale ────
    acc_key, metric_label = _detect_perf_key(h)

    epochs   = range(1, len(h["loss"]) + 1)
    loss     = np.array(h["loss"])
    val_loss = np.array(h["val_loss"]) if "val_loss" in h else None

    # Meilleure époque = val_loss minimale (ou loss si pas de val)
    ref      = val_loss if val_loss is not None else loss
    best_idx = int(np.argmin(ref))

    # ── Affichage console ──────────────────────────────────
    _console_history(h, best_idx, metric_label)

    # ── Figure ────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_perf, ax_loss) = plt.subplots(
        1, 2, figsize=(14, 5), facecolor=BG_COLOR
    )

    # Panneau 1 — Performance
    if acc_key:
        acc     = np.array(h[acc_key])
        val_acc = np.array(h[f"val_{acc_key}"]) \
                  if f"val_{acc_key}" in h else None
        _draw_panel(ax_perf, epochs, acc, val_acc,
                    best_idx=best_idx,
                    title=f"Performance du Modèle ({metric_label})",
                    ylabel="Score")
    else:
        ax_perf.text(0.5, 0.5, "Métrique non détectée",
                     ha="center", va="center",
                     transform=ax_perf.transAxes,
                     fontsize=12, color=GRAY)
        ax_perf.set_title("Performance", fontsize=14,
                           fontweight="bold", color=GRAY)

    # Panneau 2 — Loss
    _draw_panel(ax_loss, epochs, loss, val_loss,
                best_idx=best_idx,
                title="Perte (Focal Loss + Label Smoothing)",
                ylabel="Valeur de Perte")

    fig.suptitle(
        "BrainScan AI — Courbes d'Apprentissage (ResNet50 / BraTS 2021)",
        fontsize=14, fontweight="bold", color=GRAY, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close()
    print(f"\n[✓] Courbes sauvegardées  →  {save_path}  ({dpi} DPI)")


# ── Helpers internes ──────────────────────────────────────────

def _detect_perf_key(h: dict) -> tuple:
    """
    Retourne (clé_train, label_affichage) pour la métrique principale.
    Priorité : AUC > Accuracy > première clé non-loss.
    """
    keys = list(h.keys())

    # Noms possibles générés par Keras selon la version
    for candidate in ["auc", "auc_1", "auc_roc"]:
        if candidate in keys:
            return candidate, "AUC-ROC"

    for candidate in ["accuracy", "acc", "binary_accuracy",
                       "categorical_accuracy"]:
        if candidate in keys:
            return candidate, "Accuracy"

    # Fallback : première clé non-loss, non-val
    fallback = next(
        (k for k in keys if "loss" not in k and not k.startswith("val_")),
        None,
    )
    return (fallback, fallback.upper()) if fallback else (None, None)


def _draw_panel(ax, epochs, train, val, best_idx,
                title: str, ylabel: str) -> None:
    """Trace un panneau (train + val + marqueur meilleure époque)."""
    ax.set_facecolor(BG_COLOR)

    ax.plot(epochs, train, color=TRAIN_COLOR, linewidth=2,
            label="Entraînement", zorder=3)

    if val is not None:
        ax.plot(epochs, val, color=VAL_COLOR, linewidth=2,
                linestyle="--", label="Validation", zorder=3)
        ax.fill_between(epochs, train, val,
                        alpha=0.07, color=VAL_COLOR)

        # ── Marqueur ★ meilleure époque ────────────────────
        best_epoch = best_idx + 1
        best_val   = val[best_idx]

        ax.axvline(best_epoch, color=BEST_COLOR, linewidth=1.4,
                   linestyle=":", alpha=0.85, zorder=2)
        ax.scatter(best_epoch, best_val, color=BEST_COLOR,
                   s=130, marker="*", zorder=6,
                   label=f"Meilleure époque {best_epoch} "
                         f"({best_val:.4f})")
        ax.annotate(
            f" ★ é.{best_epoch}\n  {best_val:.4f}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 0.6, best_val),
            fontsize=8, color=BEST_COLOR,
            arrowprops=dict(arrowstyle="-", color=BEST_COLOR,
                            lw=0.7, alpha=0.5),
        )

    # Style
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=GRAY, pad=8)
    ax.set_xlabel("Époques", fontsize=11, color=GRAY)
    ax.set_ylabel(ylabel,   fontsize=11, color=GRAY)
    ax.tick_params(labelsize=9, colors=GRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.legend(fontsize=9, frameon=True, framealpha=0.9,
              loc="best", edgecolor="#dddddd")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _console_history(h: dict, best_idx: int,
                      metric_label: str) -> None:
    """Affiche proprement dans la console les métriques de l'historique."""
    sep = "─" * 54
    print(f"\n{'═'*54}")
    print(f"   HISTORIQUE D'ENTRAÎNEMENT — BrainScan AI")
    print(f"{'═'*54}")
    print(f"  Époques totales   : {len(h['loss'])}")
    print(f"  Meilleure époque  : {best_idx + 1}  (val_loss minimale)")
    print(f"  Métrique          : {metric_label}")
    print(f"\n  {'Métrique':<22}  {'Finale':>8}  {'Meilleure':>10}")
    print(f"  {sep}")

    groups = [
        (["loss"],                              "Loss"),
        (["auc", "auc_1"],                      "AUC-ROC"),
        (["accuracy", "acc", "binary_accuracy"],"Accuracy"),
        (["precision", "precision_1"],          "Précision"),
        (["recall", "recall_1"],                "Recall"),
        (["f1_score", "f1"],                    "F1-Score"),
        (["lr"],                                "Learning Rate"),
    ]
    keys = list(h.keys())

    for candidates, label in groups:
        tk = next((c for c in candidates if c in keys), None)
        if tk is None:
            continue
        vk = f"val_{tk}"
        t_vals = np.array(h[tk])
        v_vals = np.array(h[vk]) if vk in h else None

        print(f"\n  {label}")
        print(f"    {'Train':<20}  {t_vals[-1]:>8.4f}  "
              f"{t_vals[best_idx]:>10.4f}")
        if v_vals is not None:
            print(f"    {'Validation':<20}  {v_vals[-1]:>8.4f}  "
                  f"{v_vals[best_idx]:>10.4f}")

    print(f"\n  {sep}\n")


# ══════════════════════════════════════════════════════════════
#  2. ÉVALUATION — MÉTRIQUES + PRÉDICTIONS
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    test_gen,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Évalue le modèle sur le générateur de test et affiche
    toutes les métriques dans la console.

    Utilise for i in range(len(test_gen)) pour éviter les
    erreurs d'itérateur liées à next() sur les générateurs Keras.

    Paramètres
    ----------
    model     : modèle Keras chargé (avec custom_objects si besoin)
    test_gen  : générateur BraTSGenerator (tf.keras.utils.Sequence)
    threshold : seuil de décision binaire (défaut : 0.98)

    Retour
    ------
    y_true  : np.ndarray int   — labels réels
    y_probs : np.ndarray float — probabilités tumorales corrigées
    """
    y_true, y_probs = [], []

    print(f"\n[...] Inférence sur {len(test_gen)} batches "
          f"(seuil={threshold})...")

    # ── Boucle sûre — évite les erreurs next() ─────────────
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]

        raw = model.predict(X_batch, verbose=0)

        # Sortie sigmoïde scalaire  →  (batch, 1)
        # Sortie softmax 2 classes  →  (batch, 2)
        probs = raw[:, 0] if raw.shape[-1] == 1 else raw[:, 1]

        # Correction inversion des classes
        # (entraîné avec 0=Tumeur, 1=Sain)
        probs = 1.0 - probs

        y_probs.extend(probs.tolist())
        y_true.extend(np.array(y_batch).flatten().tolist())

    y_true  = np.array(y_true,  dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    y_pred  = (y_probs >= threshold).astype(int)

    # ── Métriques ──────────────────────────────────────────
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec  = tn / (tn + fp + 1e-9)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = float("nan")

    # ── Affichage console ──────────────────────────────────
    sep = "─" * 44
    print(f"\n{'═'*44}")
    print(f"   RÉSULTATS D'ÉVALUATION — BrainScan AI")
    print(f"{'═'*44}")
    print(f"  Seuil de décision  : {threshold}")
    print(f"  Échantillons test  : {len(y_true)}")
    print(f"\n  {sep}")
    print(f"  {'Métrique':<24}  {'Valeur':>8}")
    print(f"  {sep}")
    _metric_row("Accuracy",             acc)
    _metric_row("Précision (Tumoral)",  prec)
    _metric_row("Recall / Sensibilité", rec)
    _metric_row("Spécificité",          spec)
    _metric_row("F1-Score",             f1)
    _metric_row("AUC-ROC",              auc)
    print(f"  {sep}")
    print(f"\n  VP={tp}   FP={fp}   FN={fn}   VN={tn}")
    print(f"{'═'*44}\n")

    print(classification_report(
        y_true, y_pred,
        target_names=["Sain", "Tumoral"],
        digits=4,
    ))

    # Valeurs prêtes pour le tableau LaTeX
    print(f"{'─'*44}")
    print("  COPIER-COLLER → tableau LaTeX (Slide 5) :")
    print(f"{'─'*44}")
    print(f"  VP = {tp}   FP = {fp}")
    print(f"  FN = {fn}   VN = {tn}")
    print(f"{'─'*44}\n")

    return y_true, y_probs


def _metric_row(label: str, value: float) -> None:
    print(f"  {label:<24}  {value:>8.4f}")


# ══════════════════════════════════════════════════════════════
#  3. MATRICE DE CONFUSION
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = THRESHOLD,
    save_path: str = "matrice_confusion.png",
    dpi: int = 300,
    class_names: list = None,
) -> None:
    """
    Génère et sauvegarde une heatmap Seaborn annotée.

    Chaque cellule affiche :
      - Le nom (VP / VN / FP / FN)
      - Le nombre d'exemples
      - Le pourcentage ligne (taux de bonne/mauvaise classification)

    Paramètres
    ----------
    y_true      : labels réels          (sortie de evaluate_model)
    y_probs     : probabilités corrigées (sortie de evaluate_model)
    threshold   : seuil de décision      (défaut : 0.98)
    save_path   : chemin PNG             (défaut : matrice_confusion.png)
    dpi         : résolution d'export    (défaut : 300)
    class_names : labels affiché         (défaut : ["Sain", "Tumoral"])
    """
    if class_names is None:
        class_names = ["Sain", "Tumoral"]

    y_pred = (y_probs >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── Annotations : nom + effectif + % ligne ─────────────
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_pct   = np.where(row_sums > 0, cm / (row_sums + 1e-9), 0.0)

    cell_meta = [
        [("VN", tn, cm_pct[0, 0]),  ("FP", fp, cm_pct[0, 1])],
        [("FN", fn, cm_pct[1, 0]),  ("VP", tp, cm_pct[1, 1])],
    ]
    annots = np.array([
        [f"{meta[0]}\n{meta[1]}\n({meta[2]:.1%})"
         for meta in row]
        for row in cell_meta
    ])

    # ── Figure ─────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Heatmap de base (valeurs brutes pour la couleur)
    sns.heatmap(
        cm,
        annot=annots,
        fmt="",
        cmap=sns.light_palette("#00b4d8", as_cmap=True),
        linewidths=2.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.70, "label": "Nombre de cas"},
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    # ── Couleurs des annotations (VP/VN vert, FP/FN rouge) ─
    texts = ax.texts
    colors_map = {
        "VN": ANNOT_COLORS["VN"],
        "FP": ANNOT_COLORS["FP"],
        "FN": ANNOT_COLORS["FN"],
        "VP": ANNOT_COLORS["VP"],
    }
    for text in texts:
        first_word = text.get_text().split("\n")[0].strip()
        if first_word in colors_map:
            text.set_color(colors_map[first_word])

    # ── Contours colorés par cellule ───────────────────────
    contour_colors = [
        [ANNOT_COLORS["VN"], ANNOT_COLORS["FP"]],
        [ANNOT_COLORS["FN"], ANNOT_COLORS["VP"]],
    ]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                fill=False,
                edgecolor=contour_colors[i][j],
                linewidth=2.8,
                clip_on=False,
            ))

    # ── Labels et titre ────────────────────────────────────
    ax.set_xlabel("Classe Prédite",  fontsize=12,
                  fontweight="bold", color=GRAY, labelpad=10)
    ax.set_ylabel("Classe Réelle",   fontsize=12,
                  fontweight="bold", color=GRAY, labelpad=10)
    ax.set_xticklabels(class_names,  fontsize=11, fontweight="bold")
    ax.set_yticklabels(class_names,  fontsize=11, fontweight="bold",
                       rotation=0)
    ax.tick_params(colors=GRAY, length=0)
    ax.set_title(
        f"Matrice de Confusion\n"
        f"Seuil de décision = {threshold}  —  BrainScan AI / ResNet50",
        fontsize=12, fontweight="bold", color=GRAY, pad=14,
    )

    # ── Légende externe ────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=ANNOT_COLORS["VP"],
                       label=f"VP = {tp}  — Tumoral prédit Tumoral"),
        mpatches.Patch(color=ANNOT_COLORS["VN"],
                       label=f"VN = {tn}  — Sain prédit Sain"),
        mpatches.Patch(color=ANNOT_COLORS["FN"],
                       label=f"FN = {fn}  — Tumoral prédit Sain  ⚠"),
        mpatches.Patch(color=ANNOT_COLORS["FP"],
                       label=f"FP = {fp}  — Sain prédit Tumoral"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=9,
        framealpha=0.92,
        edgecolor="#cccccc",
        title="Légende des cellules",
        title_fontsize=9.5,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close()
    print(f"[✓] Matrice de confusion  →  {save_path}  ({dpi} DPI)")


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE — bloc à coller après model.fit() sur Kaggle
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test autonome sans GPU.
    Remplacez history, model et test_gen par vos objets réels.
    """
    import numpy as np

    # ── Historique fictif ──────────────────────────────────
    rng = np.random.default_rng(42)
    N   = 35

    def _curve(start, end, noise):
        return (np.linspace(start, end, N)
                + rng.normal(0, noise, N)).clip(0.0, 2.0)

    fake_history = {
        "loss":         _curve(0.82, 0.21, 0.018),
        "val_loss":     _curve(0.88, 0.27, 0.025),
        "auc":          _curve(0.58, 0.96, 0.008),
        "val_auc":      _curve(0.55, 0.94, 0.012),
        "lr":           [1e-4 * (0.5 ** (i // 7)) for i in range(N)],
    }
    plot_learning_curves(fake_history)

    # ── Modèle et générateur fictifs ──────────────────────
    class DummyGen:
        """Simule un tf.keras.utils.Sequence avec __len__ et __getitem__."""
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            r = np.random.default_rng(idx)
            X = r.random((16, 224, 224, 3)).astype(np.float32)
            y = r.integers(0, 2, size=16)
            return X, y

    class DummyModel:
        def predict(self, X, verbose=0):
            return np.random.default_rng().random(
                (len(X), 1)
            ).astype(np.float32)

    y_true, y_probs = evaluate_model(DummyModel(), DummyGen())
    plot_confusion_matrix(y_true, y_probs)

    print("\n[✓] Test complet terminé.")
    print("    Fichiers générés :")
    print("      ├── courbes_apprentissage.png")
    print("      └── matrice_confusion.png")
