"""
Функции для извлечения признаков из сетевого трафика.

DPI-системы смотрят именно на эти характеристики потока —
не на содержимое пакетов (оно зашифровано), а на их статистику.
"""

import numpy as np


# ── Константы масштабирования ────────────────────────────────────────────────

# Максимальные значения для нормализации входных фич в диапазон [0, 1].
# Нейронка работает лучше когда все входы примерно одного масштаба.
MAX_PACKET_SIZE = 65535   # максимальный размер TCP сегмента
MAX_ENTROPY     = 8.0     # теоретический максимум энтропии Шеннона для байтов

# Диапазоны выходных параметров трансформации.
# Трансформер выдаёт значения 0..1, мы масштабируем сюда.
PARAM_RANGES = {
    "padding_bytes": (0, 1024),
    "delay_ms":      (0, 200),
    "chunk_size":    (64, 8192),  # минимум 64 — меньше не имеет смысла
}

# Метки классов трафика (для классификатора-DPI)
TRAFFIC_CLASSES = {
    0: "HTTPS",
    1: "Telegram",
    2: "VPN",
    3: "Tor",
    4: "Unknown",
}
NUM_CLASSES = len(TRAFFIC_CLASSES)


# ── Признаки одного пакета ───────────────────────────────────────────────────

def calc_entropy(data: bytes) -> float:
    """
    Энтропия Шеннона — мера случайности байтов в пакете.

    Зашифрованный трафик (HTTPS, VPN) имеет высокую энтропию ~7.9-8.0,
    потому что шифрование делает байты равномерно распределёнными.
    Plaintext HTML имеет низкую энтропию ~4-5 из-за повторяющихся символов.

    DPI использует это чтобы отличить зашифрованный трафик от открытого.
    """
    if not data:
        return 0.0

    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    # убираем нули чтобы не считать log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def extract_packet_features(packet_size: int, entropy: float) -> np.ndarray:
    """
    Нормализует признаки одного пакета в вектор для нейронки.

    Возвращает np.ndarray shape (2,) со значениями в [0, 1].
    """
    return np.array([
        packet_size / MAX_PACKET_SIZE,
        entropy / MAX_ENTROPY,
    ], dtype=np.float32)


# ── Признаки потока (flow) ───────────────────────────────────────────────────

def extract_flow_features(
    packet_sizes: list[int],
    inter_arrival_times: list[float],  # в миллисекундах
) -> np.ndarray:
    """
    Статистические признаки потока — именно их использует настоящий DPI.

    Поток = последовательность пакетов одного TCP-соединения.
    Возвращает вектор shape (10,).

    Признаки (все нормализованы в [0, 1]):
      0  mean_size      — средний размер пакета
      1  std_size       — стандартное отклонение размера
      2  min_size       — минимальный размер
      3  max_size       — максимальный размер
      4  mean_iat       — среднее межпакетное время (IAT)
      5  std_iat        — разброс IAT
      6  min_iat        — минимальный IAT
      7  max_iat        — максимальный IAT
      8  packet_count   — количество пакетов (нормализовано)
      9  bytes_total    — суммарный объём (нормализовано)
    """
    sizes = np.array(packet_sizes, dtype=np.float32)
    iats  = np.array(inter_arrival_times, dtype=np.float32) if inter_arrival_times else np.array([0.0])

    MAX_IAT        = 5000.0   # мс
    MAX_PKT_COUNT  = 1000.0
    MAX_BYTES      = 10_000_000.0

    features = np.array([
        sizes.mean()  / MAX_PACKET_SIZE,
        sizes.std()   / MAX_PACKET_SIZE,
        sizes.min()   / MAX_PACKET_SIZE,
        sizes.max()   / MAX_PACKET_SIZE,
        iats.mean()   / MAX_IAT,
        iats.std()    / MAX_IAT,
        iats.min()    / MAX_IAT,
        iats.max()    / MAX_IAT,
        len(sizes)    / MAX_PKT_COUNT,
        sizes.sum()   / MAX_BYTES,
    ], dtype=np.float32)

    return np.clip(features, 0.0, 1.0)


# ── Масштабирование выходов трансформера ─────────────────────────────────────

def scale_transformer_output(raw: np.ndarray) -> dict:
    """
    Преобразует выход трансформера (3 значения 0..1) в реальные параметры.

    raw[0] → padding_bytes  0..1024
    raw[1] → delay_ms       0..200
    raw[2] → chunk_size     64..8192
    """
    assert len(raw) == 3, "трансформер должен возвращать ровно 3 значения"

    result = {}
    keys = list(PARAM_RANGES.keys())
    for i, key in enumerate(keys):
        lo, hi = PARAM_RANGES[key]
        result[key] = int(lo + raw[i] * (hi - lo))

    return result
