"""
Gymnasium environment для обучения RL-агента маскировки трафика.

Формулировка задачи:
  Один эпизод = одно TCP-соединение (поток из N пакетов).
  На каждом шаге агент обрабатывает один пакет и выбирает параметры
  трансформации. По завершении потока DPI-классификатор оценивает
  агрегированные статистики и выдаёт награду.

Observation space (10 dim):
  Текущие накопленные статистики потока — те же 10 признаков, что
  использует DPIClassifier. Все в диапазоне [0, 1].

Action space (3 dim, continuous [-1, 1]):
  0: chunk_size     → масштабируется в 64..1460 байт
  1: delay_ms       → масштабируется в 0..50 мс
  2: first_frag_size → масштабируется в 50..500 байт (только для 1-го пакета)

Симуляция трансформации:
  TCP-фрагментация с chunk_size уменьшает эффективный размер пакетов.
  Задержка delay_ms увеличивает IAT (inter-arrival time).
  first_frag_size дополнительно разбивает первый пакет (TLS ClientHello).
  Эти изменения статистики потока напрямую влияют на DPI-решение.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym

from data.prepare import TRAFFIC_PROFILES
from utils.features import (
    extract_flow_features,
    NUM_CLASSES,
    MAX_PACKET_SIZE,
    MAX_ENTROPY,
)


# ── Диапазоны действий ────────────────────────────────────────────────────────

CHUNK_MIN,      CHUNK_MAX      = 64,   1460   # размер фрагмента, байт
DELAY_MIN,      DELAY_MAX      = 0,    50     # задержка, мс
FIRST_FRAG_MIN, FIRST_FRAG_MAX = 50,   500    # размер первого фрагмента, байт

# Число обучающих классов (исключаем Unknown=4 — он цель, не источник)
N_TRAIN_CLASSES = NUM_CLASSES - 1  # 0..3: HTTPS, Telegram, VPN, Tor


def scale_action(action: np.ndarray) -> tuple[int, float, int]:
    """
    Масштабирует действие из [-1, 1] в реальные параметры трансформации.

    Возвращает (chunk_size, delay_ms, first_frag_size).
    """
    a = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0  # [0, 1]
    chunk_size     = int(CHUNK_MIN + a[0] * (CHUNK_MAX - CHUNK_MIN))
    delay_ms       = float(DELAY_MIN + a[1] * (DELAY_MAX - DELAY_MIN))
    first_frag_size = int(FIRST_FRAG_MIN + a[2] * (FIRST_FRAG_MAX - FIRST_FRAG_MIN))
    return chunk_size, delay_ms, first_frag_size


def simulate_fragmentation(
    orig_sizes: list[int],
    orig_iats: list[float],
    chunk_size: int,
    delay_ms: float,
    first_frag_size: int,
) -> tuple[list[int], list[float]]:
    """
    Симулирует эффект TCP-фрагментации на статистики потока.

    Реальный proxy дробит каждый Write() на куски chunk_size байт,
    и добавляет delay_ms между фрагментами. Здесь мы симулируем
    что это делает с packet_sizes и IAT'ами.

    Логика:
      - Каждый пакет размером S разбивается на ceil(S/chunk_size) фрагментов.
      - Каждый фрагмент → отдельная запись в modified_sizes.
      - IAT первого фрагмента = оригинальный IAT, остальные += delay_ms.
      - Первый пакет дополнительно ограничивается first_frag_size.
    """
    mod_sizes: list[int] = []
    mod_iats:  list[float] = []

    for pkt_idx, orig_s in enumerate(orig_sizes):
        effective_s = orig_s

        # Первый пакет (TLS ClientHello): агрессивная фрагментация
        if pkt_idx == 0 and first_frag_size > 0:
            effective_s = min(first_frag_size, orig_s)

        # Делим пакет на фрагменты chunk_size байт
        if chunk_size > 0 and chunk_size < effective_s:
            n_frags = max(1, effective_s // chunk_size)
            for frag_idx in range(n_frags):
                frag_s = chunk_size if frag_idx < n_frags - 1 else effective_s - chunk_size * (n_frags - 1)
                mod_sizes.append(max(1, frag_s))
        else:
            mod_sizes.append(effective_s)

        # IAT: оригинальный + delay для каждого дополнительного фрагмента
        if pkt_idx > 0:
            orig_iat = orig_iats[pkt_idx - 1] if pkt_idx - 1 < len(orig_iats) else 0.0
            # Между пакетами добавляем delay_ms
            mod_iats.append(orig_iat + delay_ms)

    # Заполняем IAT'ы до нужной длины
    while len(mod_iats) < len(mod_sizes) - 1:
        mod_iats.append(delay_ms)

    return mod_sizes, mod_iats


class RLProxyEnv(gym.Env):
    """
    Gymnasium environment для обучения политики маскировки трафика.

    Каждый эпизод симулирует одно TCP-соединение:
      1. reset()  → генерирует поток из N пакетов определённого класса
      2. step()   → агент обрабатывает один пакет, трансформация накапливается
      3. После N пакетов → DPI-классификатор оценивает поток → reward

    Промежуточные шаги возвращают reward=0 (sparse reward).
    Финальный reward учитывает уверенность классификатора и задержку.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        classifier,                    # DPIClassifier (замороженный)
        device: str = "cpu",
        min_packets: int = 10,
        max_packets: int = 50,
        latency_penalty_weight: float = 0.2,
        diversity_bonus_weight: float = 0.1,
    ):
        super().__init__()

        self.classifier = classifier
        self.classifier.eval()
        self._device = device

        self.min_packets = min_packets
        self.max_packets = max_packets
        self.latency_weight   = latency_penalty_weight
        self.diversity_weight = diversity_bonus_weight

        # ── Spaces ────────────────────────────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Состояние эпизода (инициализируется в reset)
        self._label:          int        = 0
        self._n_packets:      int        = 0
        self._orig_sizes:     list[int]  = []
        self._orig_iats:      list[float]= []
        self._mod_sizes:      list[int]  = []
        self._mod_iats:       list[float]= []
        self._step_idx:       int        = 0
        self._total_delay_ms: float      = 0.0

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Случайный класс трафика (кроме Unknown — он цель)
        self._label = int(self.np_random.integers(0, N_TRAIN_CLASSES))
        profile = TRAFFIC_PROFILES[self._label]

        # Количество пакетов в соединении
        self._n_packets = int(
            self.np_random.integers(self.min_packets, self.max_packets + 1)
        )

        # Генерируем оригинальные пакеты из статистического профиля
        self._orig_sizes = np.clip(
            self.np_random.normal(profile["size_mean"], profile["size_std"], self._n_packets),
            64, 65535,
        ).astype(int).tolist()

        self._orig_iats = np.clip(
            self.np_random.normal(profile["iat_mean"], profile["iat_std"], self._n_packets - 1),
            0.0, 5000.0,
        ).tolist()

        # Сбрасываем накопленные данные
        self._mod_sizes      = []
        self._mod_iats       = []
        self._step_idx       = 0
        self._total_delay_ms = 0.0

        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        chunk_size, delay_ms, first_frag = scale_action(action)

        # Применяем трансформацию к текущему пакету
        pkt_s = self._orig_sizes[self._step_idx]
        is_first = self._step_idx == 0

        if is_first and first_frag > 0:
            eff_s = min(first_frag, pkt_s)
        elif chunk_size > 0 and chunk_size < pkt_s:
            eff_s = chunk_size
        else:
            eff_s = pkt_s

        self._mod_sizes.append(eff_s)

        if self._step_idx > 0:
            orig_iat = (
                self._orig_iats[self._step_idx - 1]
                if self._step_idx - 1 < len(self._orig_iats)
                else 0.0
            )
            self._mod_iats.append(orig_iat + delay_ms)

        self._total_delay_ms += delay_ms
        self._step_idx += 1

        terminated = self._step_idx >= self._n_packets

        if terminated:
            reward, info = self._compute_reward()
        else:
            reward = 0.0
            info = {}

        return self._build_obs(), reward, terminated, False, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        Строит вектор наблюдения из накопленных статистик потока.

        Первый шаг (ни одного обработанного пакета): возвращает фичи
        первого оригинального пакета как начальное состояние.
        """
        if len(self._mod_sizes) == 0:
            obs = extract_flow_features(
                [self._orig_sizes[0]],
                [0.0],
            )
        else:
            iats = self._mod_iats if self._mod_iats else [0.0]
            obs = extract_flow_features(self._mod_sizes, iats)

        return obs.astype(np.float32)

    def _compute_reward(self) -> tuple[float, dict]:
        """
        Вычисляет финальную награду после завершения потока.

        Reward = conf_reward + unknown_bonus + diversity_bonus - latency_penalty

          conf_reward:     1 - max_confidence (выше = классификатор менее уверен)
          unknown_bonus:   +0.5 если предсказан класс Unknown (класс 4)
          diversity_bonus: нормированная энтропия предсказания
          latency_penalty: штраф за среднюю задержку
        """
        iats = self._mod_iats if self._mod_iats else [0.0]
        flow_features = extract_flow_features(self._mod_sizes, iats)

        x = torch.tensor(flow_features, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            proba = self.classifier.predict_proba(x, detach=True)

        max_conf   = proba.max().item()
        pred_class = proba.argmax().item()

        conf_reward    = 1.0 - max_conf
        unknown_bonus  = 0.5 if pred_class == NUM_CLASSES - 1 else 0.0

        # Энтропия предсказания (нормирована на log(K))
        pred_entropy  = -(proba * torch.log(proba + 1e-8)).sum().item()
        diversity     = pred_entropy / np.log(NUM_CLASSES)
        diversity_rew = self.diversity_weight * diversity

        avg_delay     = self._total_delay_ms / max(self._step_idx, 1)
        latency_pen   = self.latency_weight * (avg_delay / DELAY_MAX)

        reward = conf_reward + unknown_bonus + diversity_rew - latency_pen

        info = {
            "max_confidence":  max_conf,
            "pred_class":      pred_class,
            "true_label":      self._label,
            "avg_delay_ms":    avg_delay,
            "n_packets_orig":  self._n_packets,
            "n_packets_mod":   len(self._mod_sizes),
            "reward_conf":     conf_reward,
            "reward_unknown":  unknown_bonus,
            "reward_diversity":diversity_rew,
            "reward_latency": -latency_pen,
        }
        return reward, info
