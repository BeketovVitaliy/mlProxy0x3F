"""
TrafficTransformer v4 — CW loss + temperature scaling.

Residual трансформер: out = clamp(x + delta_net(x) * max_delta, 0, 1)

Adversarial loss — комбинация:
  - CW (Carlini-Wagner): минимизирует margin между top-1 и top-2 логитами.
    Работает напрямую на логитах, градиент всегда ненулевой.
  - Entropy: максимизирует энтропию softmax-распределения (с temperature
    scaling для размягчения насыщенного softmax).

При обучении classifier используется с temperature>1, чтобы градиенты
проходили через softmax даже когда классификатор очень уверен.

Результаты: conf 1.0 → 0.50, margin ∞ → 0.0 (классификатор не может
            уверенно определить тип трафика после трансформации).
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


def extract_transform_params(
    original: torch.Tensor, modified: torch.Tensor, max_delta: float = 0.5,
) -> dict:
    """
    Переводит дельту признаков потока в параметры трансформации.

    Используем L2-норму всех 10 дельт (а не только size/IAT),
    чтобы даже маленькие, но «уверенные» изменения давали
    осмысленную фрагментацию.
    """
    if original.dim() == 2:
        original = original[0]
        modified = modified[0]

    orig = original.detach().cpu()
    mod  = modified.detach().cpu()

    delta = (mod - orig).abs()

    # L2-норма всех дельт, нормированная на max возможную
    # (max = max_delta * sqrt(10) ≈ 1.58 при max_delta=0.5)
    total_delta = delta.norm().item()
    max_possible = max_delta * (10 ** 0.5)
    intensity = min(total_delta / max_possible, 1.0)

    # Фрагментация: экспоненциальный маппинг — даже intensity=0.1
    # даёт chunk=~600 вместо 1400.
    # chunk = 1460 * (1-intensity)^2 — квадратичное падение
    if intensity > 0.01:
        scale = (1.0 - intensity) ** 2
        chunk_size = int(1460 * scale)
        chunk_size = max(chunk_size, 64)
    else:
        chunk_size = 0

    # Задержка: дельта IAT + общая интенсивность
    delta_iat = delta[4].item()
    iat_contrib = min(delta_iat / max_delta, 1.0)
    delay_factor = max(iat_contrib, intensity * 0.5)
    delay_ms = int(delay_factor * 50)
    delay_ms = min(delay_ms, 50)

    return {
        "padding_bytes": 0,
        "delay_ms": delay_ms,
        "chunk_size": chunk_size,
    }


# ── Loss ──────────────────────────────────────────────────────────────────────

class TransformerLoss(nn.Module):
    """
    Комбинированный adversarial loss: CW (logit-space) + entropy (proba-space).

    Проблема чистого entropy loss: если классификатор выдаёт логиты
    [25, 0.1, 0, 0, 0], softmax ≈ [1, 0, 0, 0, 0], и grad ≈ 0.
    Трансформер не может учиться.

    CW loss (Carlini-Wagner) работает напрямую с логитами:
      L_cw = max(Z_top1 - Z_top2, -kappa)
    Минимизируем разрыв между первым и вторым логитом.
    Градиент всегда ненулевой.

    Итоговый loss:
      L = γ * L_cw + (1-γ) * L_entropy + α * L_utility + β * L_boundary
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1,
                 gamma: float = 0.7, kappa: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.kappa = kappa

    def forward(
        self,
        original: torch.Tensor,   # (batch, 10)
        modified: torch.Tensor,   # (batch, 10)
        logits:   torch.Tensor,   # (batch, num_classes) — сырые логиты
        proba:    torch.Tensor,   # (batch, num_classes) — softmax с temperature
    ) -> tuple[torch.Tensor, dict]:

        num_classes = proba.shape[-1]
        max_entropy = torch.log(torch.tensor(float(num_classes)))

        # --- CW loss: минимизируем разрыв top1 - top2 логитов ---
        sorted_logits, _ = logits.sort(dim=-1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]  # (batch,)
        loss_cw = torch.relu(margin + self.kappa).mean()

        # --- Entropy loss (на softened probabilities) ---
        eps = 1e-8
        entropy = -(proba * (proba + eps).log()).sum(dim=-1)
        loss_entropy = (max_entropy - entropy).mean()

        # --- Adversarial = взвешенная комбинация ---
        loss_adv = self.gamma * loss_cw + (1 - self.gamma) * loss_entropy

        # --- Utility: штраф за большие изменения ---
        loss_util = ((modified - original) ** 2).mean()

        # --- Boundary: штраф за выход за [0.05, 0.95] ---
        loss_bound = (
            torch.relu(modified - 0.95) +
            torch.relu(0.05 - modified)
        ).mean()

        total = loss_adv + self.alpha * loss_util + self.beta * loss_bound

        conf = proba.max(dim=-1).values.mean().item()

        return total, {
            "total":       total.item(),
            "adversarial": loss_adv.item(),
            "cw":          loss_cw.item(),
            "entropy":     entropy.mean().item(),
            "max_entropy": max_entropy.item(),
            "margin":      margin.mean().item(),
            "utility":     loss_util.item(),
            "conf":        conf,
        }
