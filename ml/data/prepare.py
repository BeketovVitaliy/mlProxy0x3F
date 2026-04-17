"""
Генерация синтетического датасета трафика.

Почему синтетический, а не реальный:
  Публичные датасеты (CICIDS2017, CAIDA) весят десятки гигабайт
  и требуют сложной предобработки. Для диплома синтетика достаточна —
  мы моделируем статистические характеристики каждого типа трафика
  на основе реальных исследований.

Характеристики каждого класса основаны на:
  - HTTPS: крупные пакеты (~1400 байт), низкий IAT, очень высокая энтропия
  - Telegram: смешанные размеры (медиа + текст), переменный IAT
  - VPN: максимальный размер пакетов (MTU 1500), равномерная энтропия
  - Tor: маленькие фиксированные ячейки 512 байт, специфичный паттерн
  - Unknown: случайные характеристики
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils.features import extract_flow_features, NUM_CLASSES


# Статистические профили каждого типа трафика.
# Формат: {mean_size, std_size, mean_iat_ms, std_iat_ms, entropy_mean, entropy_std}
TRAFFIC_PROFILES = {
    0: {  # HTTPS
        "size_mean": 1350, "size_std": 200,
        "iat_mean":  15,   "iat_std":  10,
        "entropy":   7.8,  "e_std":    0.15,
    },
    1: {  # Telegram
        "size_mean": 600,  "size_std": 500,
        "iat_mean":  80,   "iat_std":  120,
        "entropy":   7.5,  "e_std":    0.3,
    },
    2: {  # VPN (WireGuard/OpenVPN)
        "size_mean": 1400, "size_std": 80,
        "iat_mean":  5,    "iat_std":  3,
        "entropy":   7.95, "e_std":    0.05,  # очень равномерная
    },
    3: {  # Tor
        "size_mean": 512,  "size_std": 20,    # ячейки Tor почти фиксированы
        "iat_mean":  50,   "iat_std":  30,
        "entropy":   7.6,  "e_std":    0.2,
    },
    4: {  # Unknown
        "size_mean": 800,  "size_std": 600,
        "iat_mean":  100,  "iat_std":  200,
        "entropy":   6.0,  "e_std":    1.5,
    },
}


def generate_flow(label: int, n_packets: int = 20) -> np.ndarray:
    """
    Генерирует один поток трафика заданного класса.

    Возвращает вектор признаков shape (10,).
    """
    p = TRAFFIC_PROFILES[label]

    # Генерируем размеры пакетов с нормальным распределением
    sizes = np.random.normal(p["size_mean"], p["size_std"], n_packets)
    sizes = np.clip(sizes, 64, 65535).astype(int)

    # Генерируем межпакетные задержки
    iats = np.random.normal(p["iat_mean"], p["iat_std"], n_packets - 1)
    iats = np.clip(iats, 0, 5000)

    return extract_flow_features(sizes.tolist(), iats.tolist())


class TrafficDataset(Dataset):
    """
    PyTorch Dataset с синтетическим трафиком.

    Каждый элемент: (features: Tensor(10,), label: int)
    """

    def __init__(self, n_samples: int = 50_000, seed: int = 42):
        np.random.seed(seed)

        samples_per_class = n_samples // NUM_CLASSES
        features_list = []
        labels_list   = []

        print(f"Генерируем датасет: {n_samples} потоков ({samples_per_class} на класс)...")

        for label in range(NUM_CLASSES):
            for _ in range(samples_per_class):
                # Случайное количество пакетов в потоке (реалистично)
                n_pkt = np.random.randint(5, 100)
                feat  = generate_flow(label, n_pkt)
                features_list.append(feat)
                labels_list.append(label)

        self.features = torch.tensor(np.array(features_list), dtype=torch.float32)
        self.labels   = torch.tensor(labels_list, dtype=torch.long)

        print(f"Датасет готов: {len(self)} образцов, shape={self.features.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def get_dataloaders(
    n_samples: int  = 50_000,
    batch_size: int = 256,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Создаёт train/val DataLoader'ы.

    val_split=0.2 означает 80% train, 20% validation.
    """
    dataset = TrafficDataset(n_samples=n_samples, seed=seed)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
