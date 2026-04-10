"""
Обучение DPI-классификатора.

Запуск:
  python train/train_classifier.py

После обучения модель сохраняется в saved_models/classifier.pt
Это первый шаг — без обученного классификатора трансформер обучать нельзя.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.prepare import get_dataloaders
from models.classifier import DPIClassifier
from utils.features import TRAFFIC_CLASSES


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


def main():
    # ── Гиперпараметры ───────────────────────────────────────────────────────
    EPOCHS     = 30
    BATCH_SIZE = 256
    LR         = 1e-3
    N_SAMPLES  = 50_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Данные ───────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        n_samples=N_SAMPLES, batch_size=BATCH_SIZE
    )

    # ── Модель ───────────────────────────────────────────────────────────────
    model     = DPIClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    # Уменьшаем LR если val_loss не падает 5 эпох.
    # В некоторых версиях PyTorch у ReduceLROnPlateau нет аргумента verbose.
    try:
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nМодель: {sum(p.numel() for p in model.parameters())} параметров\n")

    # ── Обучение ─────────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_model_path = "saved_models/classifier.pt"
    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
            }, best_model_path)
            print(f"  ✓ Новый лучший val_acc={val_acc:.3f}, сохранено")

    print(f"\nОбучение завершено. Лучший val_acc={best_val_acc:.3f}")
    print(f"Модель сохранена: {best_model_path}")


if __name__ == "__main__":
    main()
