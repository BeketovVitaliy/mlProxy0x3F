"""
Flask-сервер ML-агента.

Go прокси отправляет POST /predict с признаками пакета,
получает параметры трансформации.

Запуск:
  python server/app.py

Эндпоинты:
  POST /predict  — основной, для Go прокси
  GET  /health   — для мониторинга
  GET  /stats    — статистика инференсов
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import torch
import numpy as np
from flask import Flask, request, jsonify

from models.transformer import TrafficTransformer
from utils.features import (
    extract_packet_features, scale_transformer_output
)

app = Flask(__name__)

# ── Глобальное состояние ─────────────────────────────────────────────────────
model:  TrafficTransformer | None = None
device: torch.device               = torch.device("cpu")
lock = threading.Lock()  # модель не thread-safe, нужен lock

# Статистика
stats = {
    "total_requests":  0,
    "total_errors":    0,
    "avg_latency_ms":  0.0,
    "model_loaded":    False,
}


def load_model(path: str = "saved_models/transformer.pt") -> bool:
    global model, device, stats

    if not os.path.exists(path):
        print(f"WARN: Модель не найдена по пути {path}, использую случайные веса")
        print("      Сначала запусти: python train/train_transformer.py")
        model = TrafficTransformer()
        model.eval()
        stats["model_loaded"] = False
        return False

    ckpt  = torch.load(path, map_location=device)
    model = TrafficTransformer()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    val_conf = ckpt.get("val_conf", "?")
    print(f"Модель загружена: {path} (DPI confidence при обучении={val_conf})")
    stats["model_loaded"] = True
    return True


# ── Эндпоинты ────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Принимает признаки пакета, возвращает параметры трансформации.

    Request JSON:
      {"packet_size": 1400, "entropy": 7.8}

    Response JSON:
      {"padding_bytes": 512, "delay_ms": 15, "chunk_size": 1024}
    """
    global stats
    start = time.time()

    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "invalid json"}), 400

        packet_size = int(data.get("packet_size", 0))
        entropy     = float(data.get("entropy", 0.0))

        # Нормализуем входные признаки
        features = extract_packet_features(packet_size, entropy)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with lock:
            with torch.no_grad():
                raw_output = model(x).squeeze(0).cpu().numpy()

        # Масштабируем [0,1] → реальные параметры
        params = scale_transformer_output(raw_output)

        # Обновляем статистику
        elapsed = (time.time() - start) * 1000
        stats["total_requests"] += 1
        stats["avg_latency_ms"] = (
            stats["avg_latency_ms"] * 0.95 + elapsed * 0.05  # скользящее среднее
        )

        return jsonify(params)

    except Exception as e:
        stats["total_errors"] += 1
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": stats["model_loaded"],
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify(stats)


# ── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1")  # только localhost!
    parser.add_argument("--port",  default=8000, type=int)
    parser.add_argument("--model", default="saved_models/transformer.pt")
    args = parser.parse_args()

    load_model(args.model)

    print(f"ML-агент запущен на {args.host}:{args.port}")
    # use_reloader=False важно — иначе модель загрузится дважды
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
