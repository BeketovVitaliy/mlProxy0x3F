"""
Adversarial обучение трансформера трафика.

Запуск (после train_classifier.py):
  python train/train_transformer.py

Что происходит на каждом шаге:
  1. Берём батч признаков пакетов (packet_size, entropy)
  2. Трансформер предсказывает параметры {padding, delay, chunk}
  3. Применяем трансформацию к flow-фичам (симулируем эффект)
  4. Классификатор (замороженный) смотрит на изменённые фичи
  5. Loss = уверенность классификатора (хотим её минимизировать)
  6. Обновляем только трансформер, классификатор не трогаем
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
from utils.features import (
    extract_packet_features, MAX_PACKET_SIZE, MAX_ENTROPY, PARAM_RANGES
)


def simulate_transform(
    flow_features: torch.Tensor,   # (batch, 10) — признаки потока
    transform_params: torch.Tensor # (batch, 3)  — [padding_n, delay_n, chunk_n]
) -> torch.Tensor:
    """
    Симулирует эффект трансформации на статистических признаках потока.

    В реальности трансформация меняет сырые пакеты, но во время обучения
    у нас нет реальных пакетов — мы работаем с векторами признаков.
    Поэтому симулируем: как изменятся признаки если применить параметры.

    Эффекты:
      padding_norm  → увеличивает mean_size и std_size (фичи 0, 1)
      delay_norm    → увеличивает mean_iat и std_iat (фичи 4, 5)
      chunk_norm    → уменьшает mean_size (дробим пакеты), увеличивает count (фича 8)

    Все операции дифференцируемы — градиент проходит через симуляцию
    обратно к трансформеру. Это ключевое для adversarial обучения.
    """
    padding_n = transform_params[:, 0:1]  # (batch, 1)
    delay_n   = transform_params[:, 1:2]
    chunk_n   = transform_params[:, 2:3]

    modified = flow_features.clone()

    # Padding увеличивает средний и макс размер пакета
    # padding_n=1.0 → добавляем MAX_PADDING/MAX_PKT_SIZE к нормализованному размеру
    max_padding_norm = PARAM_RANGES["padding_bytes"][1] / MAX_PACKET_SIZE
    modified[:, 0:1] = torch.clamp(modified[:, 0:1] + padding_n * max_padding_norm, 0, 1)
    modified[:, 1:2] = torch.clamp(modified[:, 1:2] + padding_n * 0.1, 0, 1)  # std тоже растёт

    # Задержки увеличивают IAT
    max_delay_norm = PARAM_RANGES["delay_ms"][1] / 5000.0
    modified[:, 4:5] = torch.clamp(modified[:, 4:5] + delay_n * max_delay_norm, 0, 1)
    modified[:, 5:6] = torch.clamp(modified[:, 5:6] + delay_n * 0.05, 0, 1)

    # Фрагментация уменьшает размер пакетов, увеличивает их количество
    # chunk_n=1.0 → максимальная фрагментация → маленькие чанки
    frag_effect = chunk_n * 0.3
    modified[:, 0:1] = torch.clamp(modified[:, 0:1] - frag_effect, 0.01, 1)
    modified[:, 8:9] = torch.clamp(modified[:, 8:9] + frag_effect, 0, 1)

    return modified


def generate_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Генерирует батч синтетических признаков пакетов для обучения трансформера.

    Возвращает:
      packet_features: (batch, 2) — [norm_size, norm_entropy]
      flow_features:   (batch, 10) — статистика потока
    """
    # Случайные размеры и энтропии
    sizes    = np.random.randint(64, 65535, batch_size)
    entropys = np.random.uniform(3.0, 8.0, batch_size)

    packet_feats = np.stack([
        sizes / MAX_PACKET_SIZE,
        entropys / MAX_ENTROPY,
    ], axis=1).astype(np.float32)

    # Генерируем реалистичные flow-фичи для каждого пакета
    # (берём из тех же размеров и энтропий)
    flow_feats = []
    for i in range(batch_size):
        n_pkt  = np.random.randint(5, 50)
        s_mean, s_std = sizes[i], max(sizes[i] * 0.1, 10)
        pkt_sizes = np.clip(np.random.normal(s_mean, s_std, n_pkt), 64, 65535)
        iats      = np.clip(np.random.exponential(20, n_pkt - 1), 0, 5000)

        from utils.features import extract_flow_features
        flow_feats.append(extract_flow_features(pkt_sizes.tolist(), iats.tolist()))

    flow_feats = np.array(flow_feats, dtype=np.float32)

    return (
        torch.tensor(packet_feats, device=device),
        torch.tensor(flow_feats,   device=device),
    )


def main():
    # ── Гиперпараметры ───────────────────────────────────────────────────────
    EPOCHS        = 50
    BATCH_SIZE    = 512
    LR            = 5e-4
    ALPHA         = 0.1   # вес utility loss
    STEPS_PER_EPOCH = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Загружаем замороженный классификатор ─────────────────────────────────
    classifier_path = "saved_models/classifier.pt"
    if not os.path.exists(classifier_path):
        print(f"ERROR: Сначала обучи классификатор: python train/train_classifier.py")
        sys.exit(1)

    classifier = DPIClassifier().to(device)
    ckpt       = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.eval()
    # Замораживаем — его веса не меняются во время обучения трансформера
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"Классификатор загружен (val_acc={ckpt['val_acc']:.3f}), заморожен")

    # ── Трансформер ──────────────────────────────────────────────────────────
    transformer = TrafficTransformer().to(device)
    optimizer   = Adam(transformer.parameters(), lr=LR)
    scheduler   = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion   = TransformerLoss(alpha=ALPHA)

    print(f"Трансформер: {sum(p.numel() for p in transformer.parameters())} параметров\n")

    os.makedirs("saved_models", exist_ok=True)
    best_loss = float("inf")

    # ── Цикл обучения ────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        transformer.train()
        epoch_losses = {"total": 0, "adversarial": 0, "utility": 0}

        for _ in range(STEPS_PER_EPOCH):
            packet_feats, flow_feats = generate_batch(BATCH_SIZE, device)

            # 1. Трансформер предсказывает параметры по признакам пакета
            transform_params = transformer(packet_feats)  # (batch, 3)

            # 2. Симулируем эффект трансформации на flow-фичах
            modified_flow = simulate_transform(flow_feats, transform_params)

            # 3. Классификатор смотрит на изменённый трафик
            confidence = classifier.max_confidence(modified_flow)  # (batch,)

            # 4. Считаем loss и делаем шаг
            optimizer.zero_grad()
            loss, components = criterion(transform_params, confidence)
            loss.backward()
            # Градиентный клиппинг — стабилизирует обучение
            nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in components.items():
                epoch_losses[k] += v

        scheduler.step()

        # Нормализуем по шагам
        for k in epoch_losses:
            epoch_losses[k] /= STEPS_PER_EPOCH

        # Метрика: средняя уверенность классификатора на валидации
        transformer.eval()
        with torch.no_grad():
            val_pkt, val_flow = generate_batch(2048, device)
            val_params   = transformer(val_pkt)
            val_modified = simulate_transform(val_flow, val_params)
            val_conf     = classifier.max_confidence(val_modified).mean().item()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={epoch_losses['total']:.4f} "
            f"(adv={epoch_losses['adversarial']:.4f}, util={epoch_losses['utility']:.4f}) | "
            f"DPI confidence={val_conf:.3f}"
        )

        # Сохраняем если лучше
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save({
                "epoch":      epoch,
                "state_dict": transformer.state_dict(),
                "val_conf":   val_conf,
            }, "saved_models/transformer.pt")
            print(f"  ✓ Сохранено (DPI confidence={val_conf:.3f})")

    print(f"\nОбучение завершено. Лучший loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
