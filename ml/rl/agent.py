"""
Inference wrapper для деплоя обученной RL-политики в Flask-сервере.

Проблема:
  Go-прокси вызывает /predict с {packet_size, entropy} для каждого пакета
  одного соединения. Но RL-политика обучалась на 10-мерных flow-признаках.

Решение — накапливать статистику по пакетам каждого соединения:
  - RLAgent хранит ConnectionState для каждого активного соединения.
  - На каждый пакет обновляет скользящие статистики (mean, std, count, ...).
  - Строит 10-мерный вектор и спрашивает политику.
  - Устаревшие соединения автоматически вытесняются (LRU TTL).

Схема:
  Go → POST /predict {packet_size, entropy, conn_id?}
    → RLAgent.predict(packet_size, entropy, conn_id)
      → ConnectionState.update(packet_size, entropy)
      → build_flow_features() → [10 dim]
      → SB3 model.predict(obs) → action [-1,1]³
      → scale_action → {chunk_size, delay_ms}
    → {"chunk_size": ..., "delay_ms": ..., "padding_bytes": 0}

Если conn_id не передан, каждый запрос обрабатывается как новое соединение.
"""

import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl.env import scale_action, CHUNK_MIN, CHUNK_MAX, DELAY_MIN, DELAY_MAX
from utils.features import (
    MAX_PACKET_SIZE,
    MAX_ENTROPY,
    extract_flow_features,
)


# ── ConnectionState ───────────────────────────────────────────────────────────

class ConnectionState:
    """
    Накапливает статистику пакетов одного TCP-соединения.

    Используется для построения 10-мерного flow-вектора
    (того же формата, что принимает DPIClassifier / RL-политика).

    Статистики вычисляются инкрементально — без хранения всех пакетов.
    Алгоритм Вэлфорда для среднего и дисперсии.
    """

    def __init__(self):
        self.created_at = time.time()
        self.last_seen  = time.time()

        # Для packet_sizes: инкрементальные статистики
        self._count    = 0
        self._mean_s   = 0.0
        self._M2_s     = 0.0   # sum of squared deviations (Welford)
        self._min_s    = float("inf")
        self._max_s    = 0.0
        self._total_b  = 0

        # Для IAT: между текущим и предыдущим пакетом
        self._last_time = time.time()
        self._iat_count = 0
        self._mean_iat  = 0.0
        self._M2_iat    = 0.0
        self._min_iat   = float("inf")
        self._max_iat   = 0.0

    def update(self, packet_size: int) -> None:
        """Обновляет статистики новым пакетом."""
        now = time.time()
        iat_ms = (now - self._last_time) * 1000.0
        self._last_time = now
        self.last_seen  = now

        # Welford для packet_size
        self._count += 1
        delta      = packet_size - self._mean_s
        self._mean_s += delta / self._count
        self._M2_s   += delta * (packet_size - self._mean_s)
        self._min_s   = min(self._min_s, packet_size)
        self._max_s   = max(self._max_s, packet_size)
        self._total_b += packet_size

        # Welford для IAT (начиная со второго пакета)
        if self._count > 1:
            self._iat_count += 1
            delta_iat        = iat_ms - self._mean_iat
            self._mean_iat  += delta_iat / self._iat_count
            self._M2_iat    += delta_iat * (iat_ms - self._mean_iat)
            self._min_iat    = min(self._min_iat, iat_ms)
            self._max_iat    = max(self._max_iat, iat_ms)

    def build_flow_features(self) -> np.ndarray:
        """
        Строит 10-мерный flow-вектор из накопленных статистик.

        Формат совпадает с extract_flow_features() из utils/features.py:
          [mean_size, std_size, min_size, max_size,
           mean_iat, std_iat, min_iat, max_iat,
           pkt_count, bytes_total]
        """
        MAX_IAT       = 5000.0
        MAX_PKT_COUNT = 1000.0
        MAX_BYTES     = 10_000_000.0

        if self._count == 0:
            return np.zeros(10, dtype=np.float32)

        std_s   = (self._M2_s   / self._count) ** 0.5 if self._count > 1 else 0.0
        mean_iat = self._mean_iat if self._iat_count > 0 else 0.0
        std_iat  = (self._M2_iat / self._iat_count) ** 0.5 if self._iat_count > 1 else 0.0
        min_iat  = self._min_iat if self._iat_count > 0 else 0.0
        max_iat  = self._max_iat if self._iat_count > 0 else 0.0

        features = np.array([
            self._mean_s / MAX_PACKET_SIZE,
            std_s        / MAX_PACKET_SIZE,
            (self._min_s if self._min_s != float("inf") else 0) / MAX_PACKET_SIZE,
            self._max_s  / MAX_PACKET_SIZE,
            mean_iat     / MAX_IAT,
            std_iat      / MAX_IAT,
            (min_iat if min_iat != float("inf") else 0) / MAX_IAT,
            max_iat      / MAX_IAT,
            self._count  / MAX_PKT_COUNT,
            self._total_b / MAX_BYTES,
        ], dtype=np.float32)

        return np.clip(features, 0.0, 1.0)


# ── RLAgent ────────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Thread-safe обёртка над обученной SB3-политикой для Flask-инференса.

    Поддерживает накопление статистик по conn_id (connection ID).
    Устаревшие соединения удаляются через conn_ttl_sec секунд бездействия.
    При достижении max_connections вытесняется самое старое (LRU).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conn_ttl_sec: float = 60.0,
        max_connections: int = 10_000,
    ):
        from stable_baselines3 import SAC

        self.model = SAC.load(model_path, device=device)
        self.model.set_env(None)   # inference-only, env не нужен

        self._device       = device
        self._conn_ttl     = conn_ttl_sec
        self._max_conns    = max_connections

        # OrderedDict для LRU: ключ = conn_id, значение = ConnectionState
        self._connections: OrderedDict[str, ConnectionState] = OrderedDict()
        self._lock = threading.Lock()

        print(f"[RLAgent] loaded: {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(
        self,
        packet_size: int,
        entropy: float,
        conn_id: Optional[str] = None,
    ) -> dict:
        """
        Основной метод для Flask: принимает packet_size + entropy,
        возвращает параметры трансформации.

        conn_id используется для накопления flow-статистики.
        Если None — каждый запрос обрабатывается независимо.
        """
        with self._lock:
            if conn_id:
                state = self._get_or_create(conn_id)
                state.update(packet_size)
                obs = state.build_flow_features()
            else:
                # Нет conn_id: аппроксимируем flow из одного пакета
                obs = self._single_packet_obs(packet_size, entropy)

            self._evict_stale()

        action, _ = self.model.predict(obs, deterministic=True)
        chunk_size, delay_ms, first_frag = scale_action(action)

        return {
            "padding_bytes":   0,
            "delay_ms":        max(0, int(delay_ms)),
            "chunk_size":      max(64, chunk_size),
            "first_frag_size": first_frag,
        }

    def reset_connection(self, conn_id: str) -> None:
        """Явно удаляет состояние соединения (вызывать при закрытии)."""
        with self._lock:
            self._connections.pop(conn_id, None)

    def get_stats(self) -> dict:
        """Возвращает статистику агента для /stats endpoint."""
        with self._lock:
            return {
                "active_connections": len(self._connections),
                "max_connections":    self._max_conns,
                "conn_ttl_sec":       self._conn_ttl,
            }

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create(self, conn_id: str) -> ConnectionState:
        if conn_id in self._connections:
            self._connections.move_to_end(conn_id)
            return self._connections[conn_id]

        # LRU eviction при переполнении
        if len(self._connections) >= self._max_conns:
            self._connections.popitem(last=False)

        state = ConnectionState()
        self._connections[conn_id] = state
        return state

    def _evict_stale(self) -> None:
        """Удаляет соединения без активности дольше TTL."""
        now = time.time()
        stale = [
            cid for cid, st in self._connections.items()
            if now - st.last_seen > self._conn_ttl
        ]
        for cid in stale:
            del self._connections[cid]

    @staticmethod
    def _single_packet_obs(packet_size: int, entropy: float) -> np.ndarray:
        """
        Аппроксимирует 10-мерный flow-вектор из одного пакета.
        Используется если conn_id не передан.
        """
        norm_s = packet_size / MAX_PACKET_SIZE
        return np.array([
            norm_s,
            norm_s * 0.1,
            norm_s * 0.5,
            min(norm_s * 1.2, 1.0),
            0.01, 0.005, 0.001, 0.05,
            0.05,
            norm_s * 0.05,
        ], dtype=np.float32).clip(0.0, 1.0)


def load_rl_agent(
    model_path: str = "saved_models/rl_agent.zip",
    device: str = "cpu",
) -> Optional["RLAgent"]:
    """
    Пытается загрузить RL-агент. Возвращает None если файл не найден.

    Используется в app.py при старте Flask.
    """
    if not os.path.exists(model_path):
        print(f"[RLAgent] модель не найдена: {model_path}")
        print("          Запусти: python rl/train.py")
        return None
    try:
        return RLAgent(model_path, device=device)
    except Exception as e:
        print(f"[RLAgent] ошибка загрузки: {e}")
        return None
