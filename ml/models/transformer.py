"""
TrafficTransformer — генерирует параметры трансформации трафика.

Задача: по признакам пакета предсказать {padding, delay, chunk_size}
так, чтобы после применения этих параметров DPI-классификатор
не смог уверенно определить тип трафика.

Архитектура:
  Input(2) → Linear(32) → ReLU
           → Linear(64) → ReLU
           → Linear(32) → ReLU
           → Linear(3)  → Sigmoid   ← выход всегда в [0, 1]

Почему Sigmoid на выходе:
  Нам нужны значения в диапазоне [0, 1] которые мы потом масштабируем.
  Sigmoid это гарантирует. ReLU бы не подошёл — обрезает отрицательные,
  Tanh даёт [-1, 1] и нужен сдвиг.

Почему сеть маленькая:
  Входных фич всего 2 (packet_size, entropy).
  Задача простая — регрессия трёх чисел.
  Большая сеть переобучится на синтетических данных.
"""

import torch
import torch.nn as nn


class TrafficTransformer(nn.Module):
    def __init__(self, input_size: int = 2, output_size: int = 3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size),
            nn.Sigmoid(),   # выход в [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 2) — [norm_packet_size, norm_entropy]
        returns: (batch_size, 3) — [padding_norm, delay_norm, chunk_norm]
        """
        return self.network(x)


# ── Loss функция ─────────────────────────────────────────────────────────────

class TransformerLoss(nn.Module):
    """
    Составная функция потерь для трансформера.

    Состоит из двух частей:

    1. adversarial_loss — главная цель:
       Минимизировать уверенность DPI-классификатора.
       Чем ниже max_confidence классификатора — тем лучше.
       loss_adv = mean(max_confidence(classifier(transformed_features)))

    2. utility_loss — ограничение:
       Трансформация не должна быть слишком агрессивной.
       Большой padding или задержка ломают UX.
       loss_util = mean(output²) — штрафуем за большие значения параметров.

    Итого: loss = adversarial_loss + alpha * utility_loss

    alpha контролирует баланс между "хорошо обманывать DPI"
    и "не слишком ломать трафик".
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        # alpha — вес utility loss.
        # 0.1 = лёгкий штраф за большие параметры.
        # Увеличь до 0.5 если трансформер выдаёт слишком большие задержки.
        self.alpha = alpha

    def forward(
        self,
        transformer_output: torch.Tensor,   # (batch, 3) — выход трансформера
        classifier_confidence: torch.Tensor, # (batch,)  — уверенность DPI
    ) -> tuple[torch.Tensor, dict]:
        """
        Возвращает (total_loss, dict с компонентами для логирования).
        """
        # Главная цель: DPI должен быть неуверен
        # Хотим чтобы confidence → 0, поэтому просто берём mean
        loss_adv = classifier_confidence.mean()

        # Штраф за большие параметры трансформации
        # transformer_output уже в [0,1], поэтому mean(x²) в [0,1]
        loss_util = (transformer_output ** 2).mean()

        total = loss_adv + self.alpha * loss_util

        return total, {
            "total":       total.item(),
            "adversarial": loss_adv.item(),
            "utility":     loss_util.item(),
        }
