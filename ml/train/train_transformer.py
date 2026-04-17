"""
Обучение трансформера v4 — CW loss + temperature scaling.

Проблема v3: градиенты через softmax были нулевые (классификатор
слишком уверен: conf=1.000 → softmax насыщен → grad≈0).

Фиксы v4:
  1. Temperature scaling (T=5.0) в predict_proba — размягчает softmax
  2. CW loss работает на сырых логитах — градиент всегда ненулевой
  3. Увеличен max_delta (0.3 → 0.5) — больше пространство для манёвра
  4. Больше эпох (60 → 100) с cosine annealing

Ожидаемая динамика:
  Epoch 01: margin=15.0  conf=0.98  entropy=0.05
  Epoch 20: margin=5.0   conf=0.70  entropy=0.80
  Epoch 50: margin=1.0   conf=0.40  entropy=1.30
  Epoch 80: margin=0.2   conf=0.28  entropy=1.50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.classifier import DPIClassifier
from models.transformer import TrafficTransformer, TransformerLoss
from data.prepare import TrafficDataset
from torch.utils.data import DataLoader


# ── Гиперпараметры ──────────────────────────────────────────────────────────

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 3e-3
MAX_DELTA   = 0.5       # макс. изменение каждого признака
TEMPERATURE = 5.0       # размягчение softmax для градиентов
NOISE_STD   = 0.03      # шум к входу (anti mode-collapse)
ALPHA       = 0.1       # вес utility loss (снижен для более агрессивных трансформаций)
BETA        = 0.1       # вес boundary loss
GAMMA       = 0.7       # вес CW loss vs entropy loss


def add_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Классификатор (заморожен) ─────────────────────────────────────────
    clf_path = "saved_models/classifier.pt"
    if not os.path.exists(clf_path):
        print("ERROR: сначала обучи классификатор: python train/train_classifier.py")
        sys.exit(1)

    classifier = DPIClassifier().to(device)
    ckpt = torch.load(clf_path, map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"Классификатор загружен (val_acc={ckpt['val_acc']:.3f}), заморожен")

    # ── Данные ────────────────────────────────────────────────────────────
    dataset     = TrafficDataset(n_samples=50_000)
    loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = TrafficDataset(n_samples=5_000, seed=999)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Трансформер ───────────────────────────────────────────────────────
    transformer = TrafficTransformer(feature_size=10, max_delta=MAX_DELTA).to(device)
    optimizer   = Adam(transformer.parameters(), lr=LR, weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion   = TransformerLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA)

    max_entropy = float(np.log(5))
    n_params    = sum(p.numel() for p in transformer.parameters())
    print(f"Трансформер: {n_params} параметров | max_delta={MAX_DELTA} | T={TEMPERATURE}")
    print(f"Loss: gamma={GAMMA} (CW weight) | alpha={ALPHA} (utility) | beta={BETA} (boundary)")
    print(f"max_entropy={max_entropy:.3f}\n")

    os.makedirs("saved_models", exist_ok=True)
    best_conf = float("inf")

    for epoch in range(1, EPOCHS + 1):
        transformer.train()
        ep = {"total": 0, "adversarial": 0, "cw": 0, "entropy": 0, "utility": 0, "margin": 0}
        n = 0

        for flow_features, _ in loader:
            flow_features = flow_features.to(device)
            noisy_input = add_noise(flow_features, NOISE_STD)

            modified = transformer(noisy_input)

            # Логиты для CW loss (без temperature — сырые)
            logits = classifier.get_logits(modified)
            # Вероятности для entropy loss (с temperature — мягкие)
            proba = classifier.predict_proba(modified, detach=False, temperature=TEMPERATURE)

            optimizer.zero_grad()
            loss, components = criterion(flow_features, modified, logits, proba)
            loss.backward()
            nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()

            for k in ep:
                if k in components:
                    ep[k] += components[k]
            n += 1

        scheduler.step()
        for k in ep:
            ep[k] /= max(n, 1)

        # ── Валидация (без шума, без temperature — честная оценка) ─────
        transformer.eval()
        val_confs, val_entropies, val_margins = [], [], []

        with torch.no_grad():
            for flow_features, _ in val_loader:
                flow_features = flow_features.to(device)
                modified = transformer(flow_features)

                logits = classifier.get_logits(modified)
                proba  = classifier.predict_proba(modified, temperature=1.0)

                conf    = proba.max(dim=-1).values
                eps_val = 1e-8
                entropy = -(proba * (proba + eps_val).log()).sum(dim=-1)

                sorted_l, _ = logits.sort(dim=-1, descending=True)
                margin = sorted_l[:, 0] - sorted_l[:, 1]

                val_confs.extend(conf.cpu().tolist())
                val_entropies.extend(entropy.cpu().tolist())
                val_margins.extend(margin.cpu().tolist())

        mean_conf    = float(np.mean(val_confs))
        mean_entropy = float(np.mean(val_entropies))
        mean_margin  = float(np.mean(val_margins))
        p25_conf     = float(np.percentile(val_confs, 25))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={ep['total']:.4f} cw={ep['cw']:.3f} ent={ep['entropy']:.3f} | "
            f"val: conf={mean_conf:.3f} (p25={p25_conf:.3f}) "
            f"margin={mean_margin:.1f} entropy={mean_entropy:.3f}/{max_entropy:.3f}"
        )

        if mean_conf < best_conf:
            best_conf = mean_conf
            torch.save({
                "epoch":        epoch,
                "state_dict":   transformer.state_dict(),
                "val_conf":     mean_conf,
                "val_conf_p25": p25_conf,
                "val_entropy":  mean_entropy,
                "val_margin":   mean_margin,
                "max_entropy":  max_entropy,
                "feature_size": 10,
                "max_delta":    MAX_DELTA,
            }, "saved_models/transformer.pt")
            print(f"  ✓ Сохранено (conf={mean_conf:.3f} margin={mean_margin:.1f})")

    print(f"\nГотово. Лучший conf={best_conf:.3f}")
    if best_conf > 0.6:
        print("Совет: увеличь TEMPERATURE или уменьши ALPHA")
    elif best_conf < 0.35:
        print("Отличный результат!")


if __name__ == "__main__":
    main()
