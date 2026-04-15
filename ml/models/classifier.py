"""
DPI-классификатор — имитирует работу реального Deep Packet Inspection.

Задача: по статистическим признакам потока определить тип трафика.
Этот классификатор мы будем "обманывать" трансформером.

Архитектура:
  Input(10) → Linear(64) → BatchNorm → ReLU → Dropout
            → Linear(128) → BatchNorm → ReLU → Dropout
            → Linear(64)  → BatchNorm → ReLU → Dropout
            → Linear(NUM_CLASSES) → Softmax

Почему такая сеть:
  - BatchNorm стабилизирует обучение (градиенты не взрываются)
  - Dropout предотвращает переобучение на синтетических данных
  - Три скрытых слоя достаточно для разделения 5 классов по 10 фичам
"""

import torch
import torch.nn as nn
from utils.features import NUM_CLASSES


class DPIClassifier(nn.Module):
    def __init__(self, input_size: int = 10, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.network = nn.Sequential(
            # Слой 1
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Слой 2
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Слой 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Выход
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size)
        returns: (batch_size, num_classes) — логиты (без softmax)

        Softmax не включаем в forward — CrossEntropyLoss принимает логиты сам.
        При инференсе применяем softmax вручную чтобы получить вероятности.
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor, detach: bool = True) -> torch.Tensor:
        """
        Возвращает вероятности классов (с softmax).

        detach=True подходит для обычного инференса (без графа градиентов).
        detach=False нужен для adversarial обучения трансформера, чтобы
        градиент доходил до входа x через замороженный классификатор.
        """
        logits = self.forward(x)
        proba = torch.softmax(logits, dim=-1)
        return proba.detach() if detach else proba

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает индекс класса с максимальной вероятностью."""
        return self.predict_proba(x, detach=True).argmax(dim=-1)

    def max_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает максимальную уверенность классификатора.

        Это ключевая метрика для трансформера:
        если max_confidence низкий — DPI "не знает" что это за трафик.
        Цель трансформера: минимизировать это значение.
        """
        return self.predict_proba(x, detach=True).max(dim=-1).values
