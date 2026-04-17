"""
Flask ML агент — HTTP API для Go прокси.

Два режима вызова:

  1. Простой (от Go прокси — packet_size + entropy):
     POST /predict {"packet_size": 1400, "entropy": 7.9}
     → возвращает {delay_ms, chunk_size}

  2. Полный (для тестов — все 10 признаков потока):
     POST /predict/flow {"features": [0.21, 0.15, ...]}
     → возвращает {delay_ms, chunk_size, modified_features}

Запуск:
  python server/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import numpy as np
import torch
from flask import Flask, request, jsonify

from models.transformer import TrafficTransformer, extract_transform_params
from utils.features import MAX_PACKET_SIZE, MAX_ENTROPY

app   = Flask(__name__)
model: TrafficTransformer | None = None
model_max_delta: float = 0.5
device = torch.device("cpu")
lock   = threading.Lock()

stats = {"total_requests": 0, "errors": 0, "avg_latency_ms": 0.0, "model_loaded": False}


def packet_to_flow_features(packet_size: int, entropy: float) -> np.ndarray:
    """
    Строит приближённый flow-вектор из двух признаков одного пакета.

    Go прокси знает только packet_size и entropy текущего пакета.
    Мы аппроксимируем остальные признаки потока как типичные для
    трафика с такими характеристиками.

    Это упрощение — в идеале прокси накапливал бы статистику
    по нескольким последним пакетам соединения.
    """
    norm_size    = packet_size / MAX_PACKET_SIZE
    norm_entropy = entropy / MAX_ENTROPY

    # Аппроксимация: считаем что поток похож на текущий пакет
    features = np.array([
        norm_size,           # 0: mean_size
        norm_size * 0.1,     # 1: std_size (10% от среднего)
        norm_size * 0.5,     # 2: min_size
        min(norm_size * 1.2, 1.0),  # 3: max_size
        0.01,                # 4: mean_iat (типичный низкий IAT)
        0.005,               # 5: std_iat
        0.001,               # 6: min_iat
        0.05,                # 7: max_iat
        0.05,                # 8: packet_count (нормализованный)
        norm_size * 0.05,    # 9: bytes_total
    ], dtype=np.float32)

    return np.clip(features, 0.0, 1.0)


def load_model(path: str = "saved_models/transformer.pt") -> bool:
    global model, model_max_delta, stats
    if not os.path.exists(path):
        print(f"WARN: модель не найдена: {path}")
        print("      Запусти: python train/train_transformer.py")
        model = TrafficTransformer(feature_size=10)
        model.eval()
        stats["model_loaded"] = False
        return False

    ckpt  = torch.load(path, map_location=device, weights_only=False)
    fsize = ckpt.get("feature_size", 10)
    mdelta = ckpt.get("max_delta", 0.5)
    model = TrafficTransformer(feature_size=fsize, max_delta=mdelta)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    model_max_delta = mdelta

    print(f"Модель загружена: {path}")
    print(f"  val_conf={ckpt.get('val_conf', '?'):.3f}  "
          f"p25={ckpt.get('val_conf_p25', '?'):.3f}  "
          f"margin={ckpt.get('val_margin', '?')}  "
          f"max_delta={mdelta}  "
          f"epoch={ckpt.get('epoch', '?')}")
    stats["model_loaded"] = True
    return True


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Простой режим: принимает packet_size + entropy от Go прокси.
    Возвращает {padding_bytes, delay_ms, chunk_size}.
    """
    global stats
    t0 = time.time()
    try:
        data        = request.get_json(force=True)
        packet_size = int(data.get("packet_size", 1024))
        entropy     = float(data.get("entropy", 7.0))

        # Строим flow-вектор из двух признаков
        flow_np = packet_to_flow_features(packet_size, entropy)
        x_orig  = torch.tensor(flow_np, dtype=torch.float32).unsqueeze(0).to(device)

        with lock:
            with torch.no_grad():
                x_mod = model(x_orig)   # (1, 10)

        params = extract_transform_params(x_orig, x_mod, max_delta=model_max_delta)

        ms = (time.time() - t0) * 1000
        stats["total_requests"] += 1
        stats["avg_latency_ms"]  = stats["avg_latency_ms"] * 0.95 + ms * 0.05

        return jsonify(params)

    except Exception as e:
        stats["errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/predict/flow", methods=["POST"])
def predict_flow():
    """
    Полный режим: принимает все 10 признаков потока.
    Возвращает параметры трансформации + модифицированные признаки.
    Используется в test_e2e.py для детального анализа.
    """
    try:
        data     = request.get_json(force=True)
        features = data.get("features")

        if not features or len(features) != 10:
            return jsonify({"error": "нужно ровно 10 признаков"}), 400

        x_orig = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with lock:
            with torch.no_grad():
                x_mod = model(x_orig)

        params = extract_transform_params(x_orig, x_mod, max_delta=model_max_delta)
        params["modified_features"] = x_mod.squeeze(0).cpu().tolist()
        params["original_features"] = x_orig.squeeze(0).cpu().tolist()

        return jsonify(params)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": stats["model_loaded"]})


@app.route("/stats")
def get_stats():
    return jsonify(stats)


# ── Запуск ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="127.0.0.1")
    p.add_argument("--port",  default=8000, type=int)
    p.add_argument("--model", default="saved_models/transformer.pt")
    args = p.parse_args()

    load_model(args.model)
    print(f"\nML агент запущен: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
