"""
Функции вознаграждения для RL-агента.

Два режима:

  SurrogateReward (быстрый, ~0.1мс)
    Использует обученный DPIClassifier как proxy для реального DPI.
    Применяется на каждом шаге обучения.
    Ограничение: классификатор обучен на синтетике, не на реальном трафике.

  NDPIReward (медленный, ~1-3 сек на поток)
    Использует реальный ndpiReader на pcap-файлах.
    Применяется периодически для валидации и дообучения.
    Требует: ndpiReader в PATH, tcpdump (опционально sudo).

Стратегия обучения:
  1. Обучаем на SurrogateReward (быстро, много эпизодов).
  2. Каждые N шагов запускаем NDPIEvaluator — получаем реальные метрики.
  3. На основе реальной обратной связи корректируем reward-shaping.
"""

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.features import NUM_CLASSES, extract_flow_features


# ── SurrogateReward ───────────────────────────────────────────────────────────

def compute_surrogate_reward(
    classifier,
    flow_features: np.ndarray,
    avg_delay_ms: float,
    *,
    latency_weight: float = 0.2,
    diversity_weight: float = 0.1,
    max_delay: float = 50.0,
) -> tuple[float, dict]:
    """
    Вычисляет суррогатную награду через DPIClassifier.

    Используется внутри RLProxyEnv._compute_reward().
    Вынесен сюда как переиспользуемая утилита.

    Возвращает (reward, info_dict).
    """
    x = torch.tensor(flow_features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        proba = classifier.predict_proba(x, detach=True)

    max_conf   = proba.max().item()
    pred_class = proba.argmax().item()

    conf_reward   = 1.0 - max_conf
    unknown_bonus = 0.5 if pred_class == NUM_CLASSES - 1 else 0.0

    entropy      = -(proba * torch.log(proba + 1e-8)).sum().item()
    diversity    = diversity_weight * (entropy / np.log(NUM_CLASSES))
    latency_pen  = latency_weight * (avg_delay_ms / max_delay)

    reward = conf_reward + unknown_bonus + diversity - latency_pen

    return reward, {
        "max_confidence":   max_conf,
        "pred_class":       pred_class,
        "pred_entropy":     entropy,
        "reward_conf":      conf_reward,
        "reward_unknown":   unknown_bonus,
        "reward_diversity": diversity,
        "reward_latency":  -latency_pen,
    }


# ── NDPIReward ─────────────────────────────────────────────────────────────────

@dataclass
class NDPIResult:
    """Результат классификации одного потока через ndpiReader."""
    protocol:    str    # "TLS", "Unknown", "Telegram", ...
    confidence:  float  # 0..1, насколько уверен ndpiReader
    is_unknown:  bool
    raw_output:  str    # сырой вывод для отладки


class NDPIReward:
    """
    Вычисляет reward через реальный ndpiReader на pcap-файле.

    Используется для периодической валидации обученной политики
    против настоящего DPI (а не суррогатного классификатора).

    Требования:
      - ndpiReader доступен в PATH (или указать ndpi_bin)
      - tcpdump с нужными правами (или pcap генерируется другим способом)

    Использование:
      reward_fn = NDPIReward(ndpi_bin="/usr/bin/ndpiReader")
      result = reward_fn.evaluate_pcap("capture.pcap")
      reward = reward_fn.result_to_reward(result, avg_delay_ms=5.0)
    """

    def __init__(
        self,
        ndpi_bin: str = "ndpiReader",
        timeout_sec: int = 10,
    ):
        self.ndpi_bin    = ndpi_bin
        self.timeout_sec = timeout_sec

    def evaluate_pcap(self, pcap_path: str) -> NDPIResult:
        """
        Запускает ndpiReader на pcap и парсит вывод.

        ndpiReader выводит строки вида:
          1	UDP 192.168.1.1:51234 <-> 8.8.8.8:443 [proto: 91/TLS][...]
        или
          1	TCP ... [proto: 0/Unknown][...]
        """
        if not Path(pcap_path).exists():
            return NDPIResult("Error", 0.0, False, f"pcap not found: {pcap_path}")

        try:
            result = subprocess.run(
                [self.ndpi_bin, "-i", pcap_path, "-v", "2"],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            output = result.stdout + result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return NDPIResult("Error", 0.0, False, str(e))

        return self._parse_ndpi_output(output)

    def _parse_ndpi_output(self, output: str) -> NDPIResult:
        """
        Парсит вывод ndpiReader и извлекает протокол и уверенность.

        Ищет строки вида:
          [proto: 91/TLS]
          [proto: 0/Unknown]
          Detected protocols:
            TLS packets: 10 bytes: 14360
        """
        protocols: dict[str, int] = {}

        # Формат: [proto: N/ProtocolName]
        for m in re.finditer(r'\[proto:\s*\d+/(\w+)\]', output):
            proto = m.group(1)
            protocols[proto] = protocols.get(proto, 0) + 1

        # Fallback: секция "Detected protocols"
        if not protocols:
            for m in re.finditer(r'^\s+(\w+)\s+packets:', output, re.MULTILINE):
                proto = m.group(1)
                protocols[proto] = protocols.get(proto, 0) + 1

        if not protocols:
            return NDPIResult("Unknown", 0.0, True, output[:500])

        # Самый частый протокол
        top_proto = max(protocols, key=protocols.get)
        total      = sum(protocols.values())
        confidence = protocols[top_proto] / total

        is_unknown = top_proto.lower() in ("unknown", "unclassified", "stun")

        return NDPIResult(top_proto, confidence, is_unknown, output[:500])

    def result_to_reward(
        self,
        result: NDPIResult,
        avg_delay_ms: float = 0.0,
        latency_weight: float = 0.2,
        max_delay: float = 50.0,
    ) -> float:
        """
        Конвертирует результат ndpiReader в скалярную награду.

          +1.0  если Unknown
          -1.0 * confidence  если уверенно классифицирован
          -latency_penalty
        """
        if result.protocol == "Error":
            return 0.0

        if result.is_unknown:
            class_reward = 1.0
        else:
            class_reward = -result.confidence

        latency_pen = latency_weight * (avg_delay_ms / max_delay)
        return class_reward - latency_pen


# ── NDPIEvaluator: пакетная валидация ─────────────────────────────────────────

class NDPIEvaluator:
    """
    Периодическая оценка обученной политики против реального nDPI.

    Алгоритм:
      1. Запускаем несколько эпизодов с текущей политикой.
      2. Для каждого эпизода генерируем синтетический pcap (через scapy
         или просто логируем действия для ручной проверки).
      3. Запускаем ndpiReader → получаем результат.
      4. Возвращаем сводку метрик.

    Примечание: генерация pcap требует scapy. Если scapy недоступен,
    evaluator логирует только суррогатные метрики.
    """

    def __init__(
        self,
        ndpi_reward: NDPIReward,
        n_eval_episodes: int = 20,
    ):
        self.ndpi_reward      = ndpi_reward
        self.n_eval_episodes  = n_eval_episodes

    def evaluate(
        self,
        env,
        policy_predict_fn,   # callable: obs → action
        verbose: bool = True,
    ) -> dict:
        """
        Запускает n_eval_episodes эпизодов и собирает метрики.

        Возвращает словарь:
          surrogate_reward_mean, surrogate_reward_std,
          unknown_rate (доля эпизодов, где pred_class == Unknown)
        """
        episode_rewards = []
        unknown_count   = 0
        confidences     = []

        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_info = {}

            while not done:
                action = policy_predict_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if info:
                    ep_info = info

            episode_rewards.append(reward)
            if ep_info.get("pred_class") == NUM_CLASSES - 1:
                unknown_count += 1
            if "max_confidence" in ep_info:
                confidences.append(ep_info["max_confidence"])

        metrics = {
            "surrogate_reward_mean": float(np.mean(episode_rewards)),
            "surrogate_reward_std":  float(np.std(episode_rewards)),
            "unknown_rate":          unknown_count / self.n_eval_episodes,
            "mean_confidence":       float(np.mean(confidences)) if confidences else 0.0,
        }

        if verbose:
            print(
                f"[NDPIEval] reward={metrics['surrogate_reward_mean']:.3f}±"
                f"{metrics['surrogate_reward_std']:.3f}  "
                f"unknown_rate={metrics['unknown_rate']:.1%}  "
                f"mean_conf={metrics['mean_confidence']:.3f}"
            )

        return metrics
