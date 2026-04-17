"""
Flask ML агент — HTTP API для Go прокси.

Режимы работы (приоритет: RL > Transformer):

  1. RL-агент (если saved_models/rl_agent.zip существует):
     POST /predict {"packet_size": 1400, "entropy": 7.9, "conn_id"?: "..."}
     → возвращает {delay_ms, chunk_size, padding_bytes}

  2. Transformer (если saved_models/transformer.pt существует):
     POST /predict {"packet_size": 1400, "entropy": 7.9}
     → возвращает {delay_ms, chunk_size, padding_bytes}

  3. Нет модели → возвращает DefaultParams (chunk_size=0, delay_ms=0).

Дополнительные эндпоинты:
  POST /predict/flow       — 10-мерный flow-вектор (для тестов)
  GET  /health             — статус
  GET  /stats              — метрики
  GET  /mode               — текущий режим (rl / transformer / none)

Запуск:
  python server/app.py [--host 127.0.0.1] [--port 8000]
                       [--rl-model saved_models/rl_agent.zip]
                       [--transformer saved_models/transformer.pt]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import numpy as np
import torch
from flask import Flask, request, jsonify

from models.transformer import TrafficTransformer, extract_transform_params
from utils.features import MAX_PACKET_SIZE, MAX_ENTROPY

app    = Flask(__name__)
lock   = threading.Lock()

# ── Глобальные модели ─────────────────────────────────────────────────────────
transformer:     "TrafficTransformer | None" = None
transformer_max_delta: float = 0.5

rl_agent:        "RLAgent | None" = None   # type: ignore[name-defined]

stats = {
    "total_requests": 0,
    "errors":         0,
    "avg_latency_ms": 0.0,
    "mode":           "none",   # "rl" | "transformer" | "none"
}


# ── Утилиты ───────────────────────────────────────────────────────────────────

def packet_to_flow_features(packet_size: int, entropy: float) -> np.ndarray:
    """
    Строит приближённый flow-вектор из двух признаков одного пакета.

    Используется Transformer-режимом (RL поддерживает собственное состояние
    через ConnectionState).
    """
    norm_size    = packet_size / MAX_PACKET_SIZE
    norm_entropy = entropy / MAX_ENTROPY

    features = np.array([
        norm_size,
        norm_size * 0.1,
        norm_size * 0.5,
        min(norm_size * 1.2, 1.0),
        0.01, 0.005, 0.001, 0.05,
        0.05,
        norm_size * 0.05,
    ], dtype=np.float32)
    return np.clip(features, 0.0, 1.0)


def _update_stats(latency_ms: float) -> None:
    stats["total_requests"] += 1
    stats["avg_latency_ms"]  = stats["avg_latency_ms"] * 0.95 + latency_ms * 0.05


# ── Загрузка моделей ──────────────────────────────────────────────────────────

def load_rl(path: str) -> bool:
    global rl_agent
    try:
        from rl.agent import load_rl_agent
        agent = load_rl_agent(path)
        if agent is None:
            return False
        with lock:
            rl_agent = agent
        stats["mode"] = "rl"
        print(f"[OK] RL-агент загружен: {path}")
        return True
    except Exception as e:
        print(f"[WARN] Не удалось загрузить RL-агент: {e}")
        return False


def load_transformer(path: str) -> bool:
    global transformer, transformer_max_delta
    if not os.path.exists(path):
        print(f"[WARN] Transformer не найден: {path}")
        return False
    try:
        ckpt    = torch.load(path, map_location="cpu", weights_only=False)
        fsize   = ckpt.get("feature_size", 10)
        mdelta  = ckpt.get("max_delta", 0.5)
        model   = TrafficTransformer(feature_size=fsize, max_delta=mdelta)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with lock:
            transformer          = model
            transformer_max_delta = mdelta

        if stats["mode"] == "none":
            stats["mode"] = "transformer"

        print(
            f"[OK] Transformer загружен: {path}  "
            f"val_conf={ckpt.get('val_conf', '?')}  max_delta={mdelta}"
        )
        return True
    except Exception as e:
        print(f"[WARN] Не удалось загрузить Transformer: {e}")
        return False


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Основной эндпоинт для Go прокси.

    Принимает {packet_size, entropy} (обязательно) и {conn_id} (опционально).
    conn_id позволяет RL-агенту накапливать статистику соединения.

    Возвращает {padding_bytes, delay_ms, chunk_size}.
    """
    t0 = time.time()
    try:
        data        = request.get_json(force=True)
        packet_size = int(data.get("packet_size", 1024))
        entropy     = float(data.get("entropy", 7.0))
        conn_id     = data.get("conn_id")   # опционально

        with lock:
            mode      = stats["mode"]
            _rl       = rl_agent
            _tfm      = transformer
            _max_delta = transformer_max_delta

        if mode == "rl" and _rl is not None:
            params = _rl.predict(packet_size, entropy, conn_id=conn_id)
        elif _tfm is not None:
            flow_np = packet_to_flow_features(packet_size, entropy)
            x_orig  = torch.tensor(flow_np).unsqueeze(0)
            with torch.no_grad():
                x_mod = _tfm(x_orig)
            params = extract_transform_params(x_orig, x_mod, max_delta=_max_delta)
        else:
            params = {"padding_bytes": 0, "delay_ms": 0, "chunk_size": 0}

        ms = (time.time() - t0) * 1000
        _update_stats(ms)
        return jsonify(params)

    except Exception as e:
        stats["errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/predict/flow", methods=["POST"])
def predict_flow():
    """
    Полный режим: принимает все 10 признаков потока.
    Используется в test_e2e.py.
    """
    try:
        data     = request.get_json(force=True)
        features = data.get("features")

        if not features or len(features) != 10:
            return jsonify({"error": "нужно ровно 10 признаков"}), 400

        with lock:
            _tfm       = transformer
            _max_delta = transformer_max_delta
            _rl        = rl_agent
            mode       = stats["mode"]

        feat_np = np.array(features, dtype=np.float32)

        if mode == "rl" and _rl is not None:
            # RL: передаём первые 2 признака как packet_size/entropy (аппроксимация)
            pkt_size = int(feat_np[0] * MAX_PACKET_SIZE)
            params = _rl.predict(pkt_size, 7.5)
            params["modified_features"] = feat_np.tolist()
            params["original_features"] = feat_np.tolist()
            return jsonify(params)

        if _tfm is not None:
            x_orig = torch.tensor(feat_np).unsqueeze(0)
            with torch.no_grad():
                x_mod = _tfm(x_orig)
            params = extract_transform_params(x_orig, x_mod, max_delta=_max_delta)
            params["modified_features"] = x_mod.squeeze(0).tolist()
            params["original_features"] = x_orig.squeeze(0).tolist()
            return jsonify(params)

        return jsonify({"error": "нет загруженной модели"}), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rl/reset", methods=["POST"])
def rl_reset_connection():
    """
    Сбрасывает накопленное состояние соединения в RL-агенте.
    Go прокси может вызывать при закрытии соединения.

    Body: {"conn_id": "..."}
    """
    data    = request.get_json(force=True)
    conn_id = data.get("conn_id")
    if not conn_id:
        return jsonify({"error": "conn_id required"}), 400

    with lock:
        _rl = rl_agent
    if _rl:
        _rl.reset_connection(conn_id)

    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "mode":   stats["mode"],
        "rl_loaded":          rl_agent is not None,
        "transformer_loaded": transformer is not None,
    })


@app.route("/stats")
def get_stats():
    with lock:
        _rl = rl_agent
    rl_stats = _rl.get_stats() if _rl else {}
    return jsonify({**stats, "rl_agent": rl_stats})


@app.route("/mode")
def get_mode():
    return jsonify({"mode": stats["mode"]})


# ── Запуск ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host",        default="127.0.0.1")
    p.add_argument("--port",        default=8000, type=int)
    p.add_argument("--rl-model",    default="saved_models/rl_agent.zip")
    p.add_argument("--transformer", default="saved_models/transformer.pt")
    args = p.parse_args()

    # Пробуем загрузить RL-агент первым (приоритет выше)
    if not load_rl(args.rl_model):
        load_transformer(args.transformer)

    if stats["mode"] == "none":
        print("[WARN] Нет ни одной модели. Прокси будет работать без трансформации.")
        print("       RL:          python rl/train.py")
        print("       Transformer: python train/train_transformer.py")

    print(f"\nML агент запущен: {args.host}:{args.port}  mode={stats['mode']}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
