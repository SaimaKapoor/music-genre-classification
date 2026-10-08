"""Robust controlled CNN hyperparameter tuning for GTZAN."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.deep_learning.dataset import GTZANMelDataset
from src.deep_learning.cnn import BaselineCNN
from src.deep_learning.train import fit
from src.deep_learning.evaluate import predict, classification_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_audio_root(manifest_path, audio_root):
    """Find the root that correctly joins with manifest relative_path."""
    manifest = pd.read_csv(manifest_path)
    paths = manifest["relative_path"].astype(str).tolist()[:50]

    candidates = []
    requested = Path(audio_root)
    candidates.extend([
        requested,
        requested / "genres",
        requested.parent,
        Path("/content/gtzan"),
        Path("/content/gtzan/genres"),
    ])

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for c in candidates:
        c = c.resolve()
        if str(c) not in seen:
            unique.append(c)
            seen.add(str(c))

    scores = []
    for root in unique:
        score = sum((root / p).is_file() for p in paths)
        scores.append((score, root))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_root = scores[0]

    if best_score == 0:
        raise FileNotFoundError(
            "Could not match manifest relative_path values to GTZAN files.\n"
            f"Requested root: {audio_root}\n"
            f"Example manifest path: {paths[0] if paths else 'NONE'}\n"
            f"Checked roots: {[str(r) for r in unique]}"
        )

    print(f"Resolved audio root: {best_root}")
    print(f"Matched {best_score}/{len(paths)} manifest paths during root check.")
    return str(best_root)


def run_experiment(name, lr, dropout, batch_size, manifest_path,
                   audio_root, epochs=15, seed=42):
    print("\n" + "=" * 70)
    print(f"Running: {name}")
    print("=" * 70)

    set_seed(seed)

    manifest = pd.read_csv(manifest_path)

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"].isin(["val", "validation"])].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    print(f"Songs: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_ds = GTZANMelDataset(train_df, audio_root, random_windows=True)
    val_ds = GTZANMelDataset(val_df, audio_root, random_windows=False)
    test_ds = GTZANMelDataset(test_df, audio_root, random_windows=False)

    # num_workers=0 deliberately: easier debugging and avoids hidden worker
    # tracebacks when an audio path is missing.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = BaselineCNN(dropout=dropout).to(device)

    # train.py expects learning_rate, NOT lr.
    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=lr,
        device=device,
    )

    y_true, y_pred = predict(model, test_loader, device=device)
    metrics = classification_metrics(y_true, y_pred)

    best_val = max(h["val_accuracy"] for h in history)

    result = {
        "experiment": name,
        "learning_rate": lr,
        "dropout": dropout,
        "batch_size": batch_size,
        "best_val_accuracy": best_val,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
    }

    print(
        f"RESULT | {name} | "
        f"best val acc={best_val:.4f} | "
        f"test acc={metrics['accuracy']:.4f} | "
        f"test macro-F1={metrics['macro_f1']:.4f}"
    )

    return result, history


def run_all(manifest_path, audio_root, epochs=15, seed=42):
    # Automatically correct /content/gtzan vs /content/gtzan/genres mismatch.
    audio_root = resolve_audio_root(manifest_path, audio_root)

    configs = [
        {"name": "baseline", "lr": 1e-3, "dropout": 0.30, "batch_size": 32},
        {"name": "low_lr", "lr": 5e-4, "dropout": 0.30, "batch_size": 32},
        {"name": "high_dropout", "lr": 1e-3, "dropout": 0.50, "batch_size": 32},
        {"name": "larger_batch", "lr": 1e-3, "dropout": 0.30, "batch_size": 64},
        {"name": "low_lr_high_dropout", "lr": 5e-4, "dropout": 0.50, "batch_size": 32},
    ]

    results = []
    histories = {}

    for cfg in configs:
        result, history = run_experiment(
            manifest_path=manifest_path,
            audio_root=audio_root,
            epochs=epochs,
            seed=seed,
            **cfg,
        )
        results.append(result)
        histories[cfg["name"]] = history

    return pd.DataFrame(results), histories
