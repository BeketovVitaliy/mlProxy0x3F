"""
Обучение RL-агента маскировки трафика.

Алгоритм: SAC (Soft Actor-Critic)
  - Off-policy: sample efficient, использует replay buffer
  - Continuous action space: естественный выбор для (chunk_size, delay_ms, frag)
  - Entropy regularization: встроена в SAC, не нужны доп. бонусы
  - Автоматическая настройка ent_coef: ent_coef="auto"

Архитектура Actor/Critic: MLP [256, 256]
  - Вход: 10-мерный flow-вектор
  - Актор: 10 → 256 → 256 → 3 (mean + log_std)
  - Критик: 10+3 → 256 → 256 → 1 (Q-value)

Запуск:
  # Базовое обучение (500k шагов, ~30 мин на GPU)
  python rl/train.py

  # С параметрами
  python rl/train.py --steps 1000000 --device cuda --eval-freq 20000

Результат:
  saved_models/rl_agent.zip        — финальная модель
  saved_models/rl_agent_best.zip   — лучшая по eval_reward
  saved_models/rl_training_log.csv — метрики обучения
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from models.classifier import DPIClassifier
from rl.env import RLProxyEnv
from rl.reward import NDPIEvaluator, NDPIReward


# ── Callbacks ──────────────────────────────────────────────────────────────────

class TrainingLogger(BaseCallback):
    """
    Логирует метрики обучения в CSV и stdout.

    Метрики (на каждый log_interval шагов):
      - mean_reward, std_reward         — награда за эпизод
      - mean_confidence, unknown_rate   — насколько путаем DPI
      - n_episodes                      — сколько эпизодов завершено
    """

    def __init__(
        self,
        log_path: str,
        log_interval: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_path     = log_path
        self.log_interval = log_interval

        self._ep_rewards:     list[float] = []
        self._ep_confidences: list[float] = []
        self._ep_unknown:     list[int]   = []
        self._csv_writer = None
        self._csv_file   = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._csv_file   = open(self.log_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
            "step", "episodes", "mean_reward", "std_reward",
            "mean_confidence", "unknown_rate", "elapsed_min",
        ])
        self._csv_writer.writeheader()
        self._t0 = time.time()

    def _on_step(self) -> bool:
        # Собираем info из завершённых эпизодов
        for info in self.locals.get("infos", []):
            if "max_confidence" in info:
                self._ep_confidences.append(info["max_confidence"])
            if "pred_class" in info:
                from utils.features import NUM_CLASSES
                self._ep_unknown.append(int(info["pred_class"] == NUM_CLASSES - 1))

        # Получаем episode rewards из Monitor
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])

        if self.num_timesteps % self.log_interval == 0 and self._ep_rewards:
            mean_r  = float(np.mean(self._ep_rewards[-100:]))
            std_r   = float(np.std(self._ep_rewards[-100:]))
            mean_c  = float(np.mean(self._ep_confidences[-100:])) if self._ep_confidences else 0.0
            unk_r   = float(np.mean(self._ep_unknown[-100:]))     if self._ep_unknown else 0.0
            elapsed = (time.time() - self._t0) / 60.0

            row = {
                "step":            self.num_timesteps,
                "episodes":        len(self._ep_rewards),
                "mean_reward":     round(mean_r, 4),
                "std_reward":      round(std_r, 4),
                "mean_confidence": round(mean_c, 4),
                "unknown_rate":    round(unk_r, 4),
                "elapsed_min":     round(elapsed, 1),
            }
            self._csv_writer.writerow(row)
            self._csv_file.flush()

            if self.verbose >= 1:
                print(
                    f"  step={self.num_timesteps:>8d}  "
                    f"reward={mean_r:+.3f}±{std_r:.3f}  "
                    f"conf={mean_c:.3f}  "
                    f"unknown={unk_r:.1%}  "
                    f"t={elapsed:.1f}min"
                )

        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()


class NDPIValidationCallback(BaseCallback):
    """
    Периодически оценивает политику через суррогатный NDPIEvaluator.

    Каждые eval_freq шагов запускает n_eval_episodes эпизодов на
    отдельной eval_env и логирует unknown_rate и mean_confidence.

    Сохраняет модель если unknown_rate улучшился.
    """

    def __init__(
        self,
        eval_env: RLProxyEnv,
        save_path: str,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.save_path       = save_path
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self._best_unknown_rate = -1.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        evaluator = NDPIEvaluator(
            ndpi_reward=NDPIReward(),   # surrogate mode (ndpiReader не запускаем)
            n_eval_episodes=self.n_eval_episodes,
        )

        def policy_fn(obs):
            action, _ = self.model.predict(obs, deterministic=True)
            return action

        metrics = evaluator.evaluate(
            self.eval_env, policy_fn, verbose=(self.verbose >= 1)
        )

        unknown_rate = metrics["unknown_rate"]
        if unknown_rate > self._best_unknown_rate:
            self._best_unknown_rate = unknown_rate
            self.model.save(self.save_path)
            if self.verbose >= 1:
                print(f"  → Новый рекорд unknown_rate={unknown_rate:.1%}, сохранено: {self.save_path}")

        return True


# ── Основная функция обучения ──────────────────────────────────────────────────

def train(
    total_timesteps: int = 500_000,
    device: str = "cpu",
    eval_freq: int = 20_000,
    n_eval_episodes: int = 100,
    classifier_path: str = "saved_models/classifier.pt",
    save_dir: str = "saved_models",
    log_interval: int = 2_000,
    seed: int = 42,
):
    """
    Основной цикл обучения RL-агента.

    1. Загружает DPIClassifier (уже обученный).
    2. Создаёт RLProxyEnv + eval_env.
    3. Инициализирует SAC.
    4. Обучает с логированием каждые log_interval шагов.
    5. Каждые eval_freq шагов — валидация, сохранение лучшей модели.
    """
    os.makedirs(save_dir, exist_ok=True)

    torch_device = torch.device(device)

    # ── 1. Загрузка классификатора ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RL Training: SAC + RLProxyEnv")
    print(f"  device={device}  steps={total_timesteps:,}  seed={seed}")
    print(f"{'='*60}\n")

    classifier = DPIClassifier()
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=torch_device, weights_only=False)
        classifier.load_state_dict(ckpt["state_dict"])
        print(f"[OK] Классификатор загружен: {classifier_path}  val_acc={ckpt.get('val_acc', '?'):.3f}")
    else:
        print(f"[WARN] Классификатор не найден: {classifier_path}")
        print("       Используем случайную инициализацию (качество будет плохим).")
        print("       Сначала запусти: python train/train_classifier.py")

    classifier.to(torch_device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    # ── 2. Создание environment ───────────────────────────────────────────────
    def make_env(eval_mode: bool = False):
        def _fn():
            env = RLProxyEnv(
                classifier=classifier,
                device=device,
                # Более короткие эпизоды для eval чтобы быстрее считать
                min_packets=10 if not eval_mode else 15,
                max_packets=50 if not eval_mode else 30,
            )
            return env
        return _fn

    # Обёртка VecEnv + Monitor для SB3 (даёт episode stats в info["episode"])
    train_env = VecMonitor(DummyVecEnv([make_env(eval_mode=False)]))
    eval_env  = make_env(eval_mode=True)()   # обычный env для callbacks

    # ── 3. Инициализация SAC ──────────────────────────────────────────────────
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200_000,      # replay buffer (хватит для 200k переходов)
        learning_starts=5_000,    # заполняем буфер случайными действиями
        batch_size=512,
        tau=0.005,                # soft update коэффициент
        gamma=0.99,               # discount factor
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",          # автоматическая настройка температуры
        target_entropy="auto",
        policy_kwargs=dict(
            net_arch=[256, 256],  # Actor + Critic по 2 слоя 256
            log_std_init=-3.0,    # начальная std для актора (консервативные действия)
        ),
        seed=seed,
        device=device,
        verbose=0,               # мы логируем сами через callback
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"SAC policy: {total_params:,} параметров")
    print(f"Replay buffer: {model.buffer_size:,} переходов")

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    best_model_path = os.path.join(save_dir, "rl_agent_best.zip")
    log_path        = os.path.join(save_dir, "rl_training_log.csv")

    logger_cb = TrainingLogger(
        log_path=log_path,
        log_interval=log_interval,
        verbose=1,
    )
    ndpi_cb = NDPIValidationCallback(
        eval_env=eval_env,
        save_path=best_model_path.replace(".zip", ""),  # SB3 сам добавит .zip
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=1,
    )

    # ── 5. Обучение ───────────────────────────────────────────────────────────
    print(f"\nСтарт обучения: {total_timesteps:,} шагов")
    print(f"  eval каждые {eval_freq:,} шагов  ({n_eval_episodes} эпизодов)")
    print(f"  log каждые {log_interval:,} шагов\n")

    t_start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger_cb, ndpi_cb],
        log_interval=None,   # отключаем внутренний SB3 logger
        reset_num_timesteps=True,
    )

    # ── 6. Сохранение финальной модели ────────────────────────────────────────
    final_path = os.path.join(save_dir, "rl_agent")
    model.save(final_path)

    elapsed = (time.time() - t_start) / 60.0
    print(f"\nОбучение завершено за {elapsed:.1f} мин")
    print(f"Финальная модель: {final_path}.zip")
    print(f"Лучшая модель:    {best_model_path}")
    print(f"Лог обучения:     {log_path}")

    # ── 7. Финальная оценка ───────────────────────────────────────────────────
    print("\nФинальная оценка (500 эпизодов)...")
    evaluator = NDPIEvaluator(NDPIReward(), n_eval_episodes=500)
    metrics = evaluator.evaluate(
        eval_env,
        lambda obs: model.predict(obs, deterministic=True)[0],
        verbose=True,
    )
    print(f"\nРезультат:")
    print(f"  unknown_rate    = {metrics['unknown_rate']:.1%}")
    print(f"  mean_confidence = {metrics['mean_confidence']:.4f}")
    print(f"  mean_reward     = {metrics['surrogate_reward_mean']:.4f}")

    train_env.close()
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for traffic obfuscation")
    parser.add_argument("--steps",      type=int,   default=500_000, help="Total timesteps")
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-freq",  type=int,   default=20_000,  help="Eval every N steps")
    parser.add_argument("--eval-eps",   type=int,   default=100,     help="Episodes per eval")
    parser.add_argument("--classifier", type=str,   default="saved_models/classifier.pt")
    parser.add_argument("--save-dir",   type=str,   default="saved_models")
    parser.add_argument("--log-interval", type=int, default=2_000)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        total_timesteps=args.steps,
        device=args.device,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_eps,
        classifier_path=args.classifier,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        seed=args.seed,
    )
