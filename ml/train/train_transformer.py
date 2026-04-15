"""
Обучение трансформера v3 — с фиксом mode collapse.

Ключевые изменения:
  1. criterion получает proba (batch, 5), а не max confidence
  2. Loss = (max_entropy - H(proba)) — максимизируем энтропию
  3. Шум к входным данным при обучении — разрывает collapse
  4. Логируем entropy и max_confidence отдельно
  5. Сохраняем по max_confidence (а не по loss) — честнее

Ожидаемая динамика:
  Epoch 01: conf=0.85  entropy=0.45  (классификатор уверен)
  Epoch 15: conf=0.60  entropy=0.90
  Epoch 30: conf=0.40  entropy=1.20
  Epoch 50: conf=0.30  entropy=1.40  (близко к max=1.61)
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


def add_noise(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """
    Добавляем небольшой шум к входным признакам при обучении.

    Это предотвращает mode collapse: разные входы → разные шумы →
    трансформер вынужден выдавать разные выходы, не может
    выучить константный маппинг.
    """
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)


def main():
    EPOCHS     = 60
    BATCH_SIZE = 256   # меньше батч — разнообразнее градиенты
    LR         = 1e-3
    ALPHA      = 0.2  # вес utility loss (умеренный)
    BETA       = 0.1   # вес boundary loss
    NOISE_STD  = 0.05  # шум к входу при обучении

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Классификатор (заморожен) ─────────────────────────────────────────
    clf_path = "saved_models/classifier.pt"
    if not os.path.exists(clf_path):
        print("ERROR: сначала обучи классификатор")
        sys.exit(1)

    classifier = DPIClassifier().to(device)
    ckpt = torch.load(clf_path, map_location=device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"Классификатор загружен (val_acc={ckpt['val_acc']:.3f}), заморожен")

    # ── Данные ───────────────────────────────────────────────────────────
    dataset     = TrafficDataset(n_samples=50_000)
    loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataset = TrafficDataset(n_samples=5_000, seed=999)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Трансформер ───────────────────────────────────────────────────────
    transformer = TrafficTransformer(feature_size=10, max_delta=0.3).to(device)
    optimizer   = Adam(transformer.parameters(), lr=LR, weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion   = TransformerLoss(alpha=ALPHA, beta=BETA)

    max_entropy = float(np.log(5))  # log(num_classes) ≈ 1.609
    n_params    = sum(p.numel() for p in transformer.parameters())
    print(f"Трансформер: {n_params} параметров | max_entropy={max_entropy:.3f}\n")

    os.makedirs("saved_models", exist_ok=True)
    best_conf = float("inf")

    for epoch in range(1, EPOCHS + 1):
        transformer.train()
        ep = {"total": 0.0, "adversarial": 0.0, "entropy": 0.0, "utility": 0.0}
        n  = 0

        for flow_features, _ in loader:
            flow_features = flow_features.to(device)

            # Шум к входу — ключевой фикс mode collapse
            noisy_input = add_noise(flow_features, NOISE_STD)

            # Трансформер: вход + дельта = модифицированные признаки
            modified = transformer(noisy_input)

            # Классификатор выдаёт полное распределение вероятностей
            proba = classifier.predict_proba(modified, detach=False)   # (batch, 5)

            optimizer.zero_grad()
            loss, components = criterion(flow_features, modified, proba)
            loss.backward()
            nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=0.5)
            optimizer.step()

            for k in ep:
                if k in components:
                    ep[k] += components[k]
            n += 1

        scheduler.step()
        for k in ep:
            ep[k] /= n

        # ── Валидация (без шума — честная оценка) ─────────────────────
        transformer.eval()
        val_confs, val_entropies = [], []

        with torch.no_grad():
            for flow_features, _ in val_loader:
                flow_features = flow_features.to(device)
                modified = transformer(flow_features)  # без шума!
                proba    = classifier.predict_proba(modified)

                conf    = proba.max(dim=-1).values
                eps     = 1e-8
                entropy = -(proba * (proba + eps).log()).sum(dim=-1)

                val_confs.extend(conf.cpu().tolist())
                val_entropies.extend(entropy.cpu().tolist())

        mean_conf    = float(np.mean(val_confs))
        mean_entropy = float(np.mean(val_entropies))
        p25_conf     = float(np.percentile(val_confs, 25))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={ep['total']:.4f} "
            f"adv={ep['adversarial']:.4f} "
            f"util={ep['utility']:.4f} | "
            f"conf={mean_conf:.3f} (p25={p25_conf:.3f}) "
            f"entropy={mean_entropy:.3f}/{max_entropy:.3f}"
        )

        if mean_conf < best_conf:
            best_conf = mean_conf
            torch.save({
                "epoch":        epoch,
                "state_dict":   transformer.state_dict(),
                "val_conf":     mean_conf,
                "val_conf_p25": p25_conf,
                "val_entropy":  mean_entropy,
                "max_entropy":  max_entropy,
                "feature_size": 10,
                "max_delta":    0.3,
            }, "saved_models/transformer.pt")
            print(f"  ✓ Сохранено (conf={mean_conf:.3f} entropy={mean_entropy:.3f})")

    print(f"\nГотово. Лучший conf={best_conf:.3f}")
    if best_conf > 0.6:
        print("Совет: уменьши ALPHA до 0.2 или увеличь NOISE_STD до 0.05")
    elif best_conf < 0.35:
        print("Отличный результат!")


if __name__ == "__main__":
    main()
