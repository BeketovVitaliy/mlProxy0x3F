"""
TrafficTransformer v3 — с фиксом mode collapse.

Проблема v2: трансформер схлопывался в одну точку (mode collapse).
Все входы маппились в ~[0.23, 0.21, 0.31, ...] с confidence=1.0.

Причина: loss минимизировал max_confidence, трансформер нашёл
точку где классификатор уверен в другом классе и застрял там.

Фиксы:
  1. Loss теперь максимизирует ЭНТРОПИЮ распределения классов
     (хотим чтобы все 5 классов были равновероятны ~0.2,
      а не чтобы один класс сменился другим)

  2. Жёсткое ограничение delta: выход не может отличаться
     от входа более чем на max_delta по каждому признаку.
     Это физически невозможно убежать далеко от оригинала.

  3. Шум при обучении (в train): разные входы → разные выходы,
     сеть не может выучить константный маппинг.

  4. Residual connection: выход = вход + small_delta.
     Сеть учит дельту, а не абсолютные значения.
     Это само по себе предотвращает collapse.
"""

import torch
import torch.nn as nn


class TrafficTransformer(nn.Module):
    """
    Residual трансформер: выход = clamp(вход + delta, 0, 1)

    Сеть учит ДЕЛЬТУ (изменение признаков), а не абсолютные значения.
    Это ключевое — нельзя выдать константу если вход разный,
    потому что выход = вход + delta, и вход всегда разный.

    max_delta ограничивает максимальное изменение каждого признака.
    """

    def __init__(self, feature_size: int = 10, max_delta: float = 0.3):
        super().__init__()
        self.feature_size = feature_size
        self.max_delta    = max_delta

        # Сеть учит дельту
        self.delta_net = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, feature_size),
            nn.Tanh(),   # выход в [-1, 1], потом масштабируем на max_delta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (batch, 10) — исходные признаки
        returns: (batch, 10) — модифицированные признаки

        Формула: out = clamp(x + delta * max_delta, 0, 1)
        delta ∈ [-1, 1] после Tanh
        """
        delta    = self.delta_net(x) * self.max_delta
        modified = torch.clamp(x + delta, 0.0, 1.0)
        return modified


def extract_transform_params(original: torch.Tensor, modified: torch.Tensor) -> dict:
    """
    Переводит дельту признаков потока в реальные параметры трансформации.

    Признаки (индексы):
      0: mean_size  1: std_size  2: min_size  3: max_size
      4: mean_iat   5: std_iat   6: min_iat   7: max_iat
      8: pkt_count  9: bytes_total

    Маппинг:
      Δmean_size > 0  →  padding_bytes
      Δmean_iat  > 0  →  delay_ms
      Δmean_size < 0  →  chunk_size (фрагментация)
    """
    if original.dim() == 2:
        original = original[0]
        modified = modified[0]

    orig = original.detach().cpu()
    mod  = modified.detach().cpu()

    delta_size = (mod[0] - orig[0]).item()
    delta_iat  = (mod[4] - orig[4]).item()

    padding_bytes = int(max(0.0, delta_size) * 1024)
    padding_bytes = min(padding_bytes, 1024)

    delay_ms = int(max(0.0, delta_iat) * 200)
    delay_ms = min(delay_ms, 200)

    if delta_size < 0:
        frag       = min(abs(delta_size) / 0.3, 1.0)  # нормируем на max_delta
        chunk_size = int(8192 * (1.0 - frag * 0.9))
        chunk_size = max(chunk_size, 64)
    else:
        chunk_size = 0

    return {"padding_bytes": padding_bytes, "delay_ms": delay_ms, "chunk_size": chunk_size}


# ── Loss ──────────────────────────────────────────────────────────────────────

class TransformerLoss(nn.Module):
    """
    Entropy-based adversarial loss.

    Цель: максимизировать энтропию распределения классов.

    Энтропия равномерного распределения по 5 классам = log(5) ≈ 1.609
    Энтропия уверенного предсказания (один класс=1.0) = 0

    Мы хотим энтропию → log(num_classes), то есть:
      loss_adv = log(num_classes) - H(proba)  →  минимизируем это
      При идеальном результате loss_adv = 0

    Компоненты:
      loss_adv  — (log(K) - H(p))  — главная цель
      loss_util — L2 дельты признаков — не слишком большие изменения
      loss_bound — штраф за выход за [0.05, 0.95] — реалистичность
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(
        self,
        original:  torch.Tensor,  # (batch, 10)
        modified:  torch.Tensor,  # (batch, 10)
        proba:     torch.Tensor,  # (batch, num_classes) — полное softmax распределение
    ) -> tuple[torch.Tensor, dict]:

        num_classes = proba.shape[-1]
        max_entropy = torch.log(torch.tensor(float(num_classes)))

        # Энтропия Шеннона: H = -sum(p * log(p))
        eps     = 1e-8
        entropy = -(proba * (proba + eps).log()).sum(dim=-1)  # (batch,)

        # Хотим H → max_entropy, поэтому минимизируем (max_entropy - H)
        loss_adv = (max_entropy - entropy).mean()

        # Штраф за большие изменения признаков
        loss_util = ((modified - original) ** 2).mean()

        # Штраф за выход за разумные границы
        loss_bound = (
            torch.relu(modified - 0.95) +
            torch.relu(0.05 - modified)
        ).mean()

        total = loss_adv + self.alpha * loss_util + self.beta * loss_bound

        return total, {
            "total":       total.item(),
            "adversarial": loss_adv.item(),
            "entropy":     entropy.mean().item(),
            "max_entropy": max_entropy.item(),
            "utility":     loss_util.item(),
        }
