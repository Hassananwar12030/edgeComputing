"""
Train AudioCNN on UrbanSound8K (bootstrap)
==========================================

Loads the cached spectrograms produced by prepare_urbansound8k.py and trains
the scaffolded AudioCNN end-to-end.

Single-fold split (fold 1-8 train, fold 9 val, fold 10 test). Class-imbalance
handled via sklearn's balanced class weights. Adam + ReduceLROnPlateau +
EarlyStopping with restore_best_weights.

Outputs:
    models/audio_cnn_urbansound8k.keras   — best model (val_loss restore)
    data/urbansound8k/training_metrics.json — full per-class report + history
    /tmp/training_curves.png              — train/val loss + accuracy curves
    /tmp/confusion_matrix.png             — normalized confusion matrix on test fold

Run:
    .venv/bin/python scripts/train_audio_cnn.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402

from src.server.training.models.audio_cnn import AudioCNN  # noqa: E402

CACHE = Path("data/urbansound8k/cache.npz")
MODEL_OUT = Path("models/audio_cnn_urbansound8k.keras")
METRICS_OUT = Path("data/urbansound8k/training_metrics.json")
CURVES_OUT = Path("/tmp/training_curves.png")
CM_OUT = Path("/tmp/confusion_matrix.png")

# Training config — kept boring for the bootstrap baseline
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100   # bumped from 50 — first run hit max_epochs without EarlyStopping firing
LR = 1e-3
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5


def compute_balanced_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """sklearn-equivalent: w_i = N / (n_classes * count_i)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero
    weights = len(y) / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    """Plain numpy confusion matrix to avoid pulling sklearn just for this."""
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Precision/recall/F1 per class, plus macro/weighted averages."""
    n = len(classes)
    cm = confusion_matrix(y_true, y_pred, n)
    report = {}
    macro_p = macro_r = macro_f = 0.0
    weighted_p = weighted_r = weighted_f = 0.0
    total = cm.sum()

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[cls] = {
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "support": int(support),
        }
        macro_p += prec; macro_r += rec; macro_f += f1
        weighted_p += prec * support
        weighted_r += rec * support
        weighted_f += f1 * support

    report["macro avg"] = {
        "precision": macro_p / n, "recall": macro_r / n, "f1": macro_f / n,
        "support": int(total),
    }
    report["weighted avg"] = {
        "precision": weighted_p / total, "recall": weighted_r / total,
        "f1": weighted_f / total, "support": int(total),
    }
    return report


def format_report(report: dict) -> str:
    lines = [f"{'class':<22} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}"]
    lines.append("-" * 60)
    for cls, r in report.items():
        if cls in ("macro avg", "weighted avg"):
            continue
        lines.append(f"{cls:<22} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                     f"{r['f1']:>8.3f} {r['support']:>9}")
    lines.append("-" * 60)
    for cls in ("macro avg", "weighted avg"):
        r = report[cls]
        lines.append(f"{cls:<22} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                     f"{r['f1']:>8.3f} {r['support']:>9}")
    return "\n".join(lines)


def main() -> int:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not CACHE.exists():
        print(f"ERROR: {CACHE} not found. Run prepare_urbansound8k.py first.",
              file=sys.stderr)
        return 2

    print(f"Loading cache: {CACHE}")
    data = np.load(CACHE, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    classes = [str(c) for c in data["classes"]]
    n_classes = len(classes)

    print(f"  train: X={X_train.shape}  y={y_train.shape}")
    print(f"  val:   X={X_val.shape}  y={y_val.shape}")
    print(f"  test:  X={X_test.shape}  y={y_test.shape}")
    print(f"  classes ({n_classes}): {classes}")

    class_weight = compute_balanced_class_weights(y_train, n_classes)
    print("\nClass weights (balanced):")
    for i, w in class_weight.items():
        print(f"  {i} {classes[i]:<22} weight={w:.3f}")

    print("\nBuilding model...")
    model = AudioCNN(input_shape=X_train.shape[1:], num_classes=n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  params: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=LR_PATIENCE,
            min_lr=1e-5, verbose=1,
        ),
    ]

    print(f"\nTraining: batch={BATCH_SIZE}, max epochs={EPOCHS}, "
          f"early stop patience={EARLY_STOP_PATIENCE}")
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,  # one line per epoch
    )
    elapsed = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])
    print(f"\nTraining time: {elapsed:.1f}s "
          f"({elapsed/60:.1f} min, {elapsed/max(epochs_run,1):.1f}s/epoch)")

    # ---------- Evaluate on test fold ----------
    print("\nEvaluating on test fold (fold 10)...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  test loss: {test_loss:.4f}")
    print(f"  test accuracy: {test_acc:.4f}")

    y_proba = model.predict(X_test, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    report = per_class_report(y_test, y_pred, classes)
    print("\nPer-class report (test fold):")
    print(format_report(report))

    # ---------- Save artifacts ----------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nSaved model: {MODEL_OUT}")

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_run": epochs_run,
        "training_time_s": elapsed,
        "classes": classes,
        "config": {
            "seed": SEED, "batch_size": BATCH_SIZE, "max_epochs": EPOCHS,
            "lr": LR, "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "class_weight": class_weight,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "per_class_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred, n_classes).tolist(),
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {METRICS_OUT}")

    # ---------- Plots ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="train")
    ax1.plot(history.history["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history.history["accuracy"], label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_OUT, dpi=100, bbox_inches="tight")
    print(f"Saved training curves: {CURVES_OUT}")

    cm = confusion_matrix(y_test, y_pred, n_classes)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    for i in range(n_classes):
        for j in range(n_classes):
            txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (test fold) — accuracy {test_acc:.3f}")
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=100, bbox_inches="tight")
    print(f"Saved confusion matrix: {CM_OUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
