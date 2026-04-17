"""
End-to-end тест системы адаптивной маскировки трафика.

Что делает скрипт:
  1. Проверяет что все компоненты живы (прокси, ML агент)
  2. Захватывает трафик БЕЗ ML трансформации → анализирует nDPI
  3. Захватывает трафик С ML трансформацией → анализирует nDPI
  4. Сравнивает уверенность классификатора до/после
  5. Строит графики для диплома

Запуск:
  # Минимальный (без реального захвата трафика, только ML метрики)
  python test_e2e.py --mock

  # Полный (нужен запущенный прокси и ML агент на VPS)
  python test_e2e.py --proxy-host 127.0.0.1 --proxy-port 1080 --ml-host 127.0.0.1 --ml-port 8000

Зависимости:
  pip install "requests[socks]" matplotlib numpy pandas tqdm
  apt install tcpdump ndpi-tools  (для полного режима)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import matplotlib
matplotlib.use("Agg")  # без GUI — для VPS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# ── Конфигурация ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    proxy_host:   str  = "127.0.0.1"
    proxy_port:   int  = 1080
    ml_host:      str  = "127.0.0.1"
    ml_port:      int  = 8000
    output_dir:   str  = "test_results"
    n_requests:   int  = 100    # сколько запросов делать в каждом тесте
    capture_iface: str = "any"  # lo/eth0/…; any — весь трафик (чаще всего правильно для tcpdump)
    mock:         bool = False  # True = без реального трафика
    pause_sec:    Optional[int] = None  # None = авто (Enter / пауза в не-TTY)


# ── Структуры данных ──────────────────────────────────────────────────────────

@dataclass
class MLPrediction:
    packet_size:   int
    entropy:       float
    padding_bytes: int
    delay_ms:      int
    chunk_size:    int
    latency_ms:    float


@dataclass
class DPIResult:
    label:      str   # "HTTPS", "Telegram", "Unknown" и т.д.
    confidence: float # 0..1
    flow_count: int


@dataclass
class TestResults:
    ml_predictions:     list[MLPrediction]  = field(default_factory=list)
    dpi_before:         list[DPIResult]     = field(default_factory=list)
    dpi_after:          list[DPIResult]     = field(default_factory=list)
    ml_latencies_ms:    list[float]         = field(default_factory=list)
    confidences_before: list[float]         = field(default_factory=list)
    confidences_after:  list[float]         = field(default_factory=list)


# ── Утилиты ───────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_ok(condition: bool, msg: str, fail_msg: str = None):
    if condition:
        print(f"  [OK]  {msg}")
    else:
        print(f"  [FAIL] {fail_msg or msg}")
        if not args_global.mock:
            sys.exit(1)


# ── Шаг 1: проверка компонентов ───────────────────────────────────────────────

def check_components(cfg: Config) -> bool:
    print_section("Шаг 1: проверка компонентов")

    # Проверяем ML агент
    try:
        r = requests.get(f"http://{cfg.ml_host}:{cfg.ml_port}/health", timeout=3)
        data = r.json()
        check_ok(r.status_code == 200, f"ML агент доступен ({cfg.ml_host}:{cfg.ml_port})")
        check_ok(data.get("model_loaded"), "Модель загружена", "Модель НЕ загружена — сначала обучи: python train/train_transformer.py")
    except Exception as e:
        check_ok(False, "", f"ML агент недоступен: {e}")
        return False

    # Проверяем прокси через curl
    if not cfg.mock:
        result = subprocess.run(
            ["curl", "-s", "--socks5-hostname", f"{cfg.proxy_host}:{cfg.proxy_port}",
             "--max-time", "8", "-k", "https://ifconfig.me"],
            capture_output=True, text=True
        )
        check_ok(result.returncode == 0, f"SOCKS5 прокси работает, внешний IP: {result.stdout.strip()}")

    return True


# ── Шаг 2: тестирование ML предсказаний ──────────────────────────────────────

def test_ml_predictions(cfg: Config, results: TestResults):
    print_section("Шаг 2: тестирование ML предсказаний")

    # Тестовые случаи — разные типы трафика
    test_cases = [
        # (packet_size, entropy, описание)
        (1400, 7.9,  "HTTPS (крупный зашифрованный)"),
        (512,  7.6,  "Tor (маленькие ячейки)"),
        (1400, 7.95, "VPN (максимальный, равномерный)"),
        (200,  4.5,  "HTTP plaintext (маленький, низкая энтропия)"),
        (800,  7.2,  "Telegram (средний)"),
        (64,   2.0,  "DNS (крошечный, низкая энтропия)"),
    ]

    print(f"\n  {'Описание':<35} {'Размер':>7} {'Энтропия':>9} {'Padding':>8} {'Delay':>7} {'Chunk':>7} {'Мс':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")

    for size, entropy, desc in test_cases:
        t0 = time.time()
        try:
            r = requests.post(
                f"http://{cfg.ml_host}:{cfg.ml_port}/predict",
                json={"packet_size": size, "entropy": entropy},
                timeout=1
            )
            pred = r.json()
            latency = (time.time() - t0) * 1000

            p = MLPrediction(
                packet_size=size, entropy=entropy,
                padding_bytes=pred["padding_bytes"],
                delay_ms=pred["delay_ms"],
                chunk_size=pred["chunk_size"],
                latency_ms=latency,
            )
            results.ml_predictions.append(p)
            results.ml_latencies_ms.append(latency)

            print(f"  {desc:<35} {size:>7} {entropy:>9.1f} {pred['padding_bytes']:>8} {pred['delay_ms']:>7} {pred['chunk_size']:>7} {latency:>5.1f}")

        except Exception as e:
            print(f"  {desc:<35} ОШИБКА: {e}")

    if results.ml_latencies_ms:
        avg = np.mean(results.ml_latencies_ms)
        p95 = np.percentile(results.ml_latencies_ms, 95)
        print(f"\n  Латентность ML агента: avg={avg:.1f}ms  p95={p95:.1f}ms")
        check_ok(avg < 50, f"Средняя латентность {avg:.1f}ms < 50ms (приемлемо)", f"Латентность {avg:.1f}ms слишком высокая!")


# ── Шаг 3: захват и анализ трафика ───────────────────────────────────────────

def _https_via_socks_curl(cfg: Config, url: str) -> tuple[bool, str]:
    """
    HTTPS через SOCKS5 тем же способом, что и проверка в шаге 1 (curl + remote DNS).
    Возвращает (успех HTTP 2xx/3xx, текст для отладки при ошибке).
    """
    ep = f"{cfg.proxy_host}:{cfg.proxy_port}"
    result = subprocess.run(
        [
            "curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
            "--max-time", "20",
            "--connect-timeout", "12",
            "--socks5-hostname", ep,
            "-k", url,
        ],
        capture_output=True,
        text=True,
    )
    code = (result.stdout or "").strip()
    if result.returncode == 0 and code.startswith(("2", "3")):
        return True, code
    err = (result.stderr or "").strip() or f"http_code={code!r} rc={result.returncode}"
    return False, err[:200]


def _https_via_socks_requests(cfg: Config, url: str) -> tuple[bool, str]:
    """Резервный путь: requests + socks5h (DNS на стороне прокси). Нужен PySocks."""
    proxies = {
        "https": f"socks5h://{cfg.proxy_host}:{cfg.proxy_port}",
        "http": f"socks5h://{cfg.proxy_host}:{cfg.proxy_port}",
    }
    try:
        r = requests.get(url, proxies=proxies, timeout=20, verify=False)
        return (200 <= r.status_code < 400, f"status={r.status_code}")
    except Exception as e:
        return False, str(e)[:200]


def capture_traffic(cfg: Config, pcap_path: str, with_ml: bool, n_requests: int) -> bool:
    """Захватывает трафик через tcpdump пока генерируем нагрузку через прокси."""
    print(f"\n  Захват трафика → {pcap_path} (ML={'включён' if with_ml else 'выключен'})")

    if os.path.exists(pcap_path):
        try:
            os.remove(pcap_path)
        except OSError:
            pass

    # Запускаем tcpdump
    tcpdump = subprocess.Popen(
        ["tcpdump", "-i", cfg.capture_iface, "-w", pcap_path, "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.6)
    if tcpdump.poll() is not None:
        err = (tcpdump.stderr.read() or b"").decode(errors="replace").strip()
        print(f"  [FAIL] tcpdump не запустился: {err or 'неизвестная ошибка'}")
        print("         Нужны права (sudo/cap_net_raw) или верный --capture-iface")
        return False

    test_urls = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://www.cloudflare.com",
    ]

    ok_count = 0
    first_err: Optional[str] = None
    for i in tqdm(range(n_requests), desc="  Запросы", ncols=60):
        url = test_urls[i % len(test_urls)]
        ok, dbg = _https_via_socks_curl(cfg, url)
        if not ok:
            ok, dbg = _https_via_socks_requests(cfg, url)
        if ok:
            ok_count += 1
        elif first_err is None:
            first_err = f"{url}: {dbg}"
        time.sleep(0.05)

    time.sleep(0.4)
    tcpdump.terminate()
    try:
        tcpdump.wait(timeout=10)
    except subprocess.TimeoutExpired:
        tcpdump.kill()

    pcap_ok = os.path.isfile(pcap_path) and os.path.getsize(pcap_path) > 64
    if not pcap_ok:
        print("  [FAIL] pcap пустой или слишком мал — tcpdump ничего не записал (интерфейс/права/BPF).")

    print(f"  Успешных запросов: {ok_count}/{n_requests}")
    if ok_count == 0 and first_err:
        print(f"  Подсказка: первая ошибка — {first_err}")
        print("  Убедись: curl работает через тот же SOCKS; для Python нужен PySocks (pip install 'requests[socks]').")

    return ok_count > 0 and pcap_ok


def pause_between_ml_phases(cfg: Config) -> None:
    """Пауза для ручного перезапуска прокси с ML между захватами."""
    if cfg.pause_sec is not None:
        if cfg.pause_sec <= 0:
            return
        print(f"  Пауза {cfg.pause_sec} с — перезапусти прокси с флагом -ml-enabled")
        time.sleep(cfg.pause_sec)
        return
    if sys.stdin.isatty():
        input("  Нажми Enter когда прокси перезапущен с ML...")
    else:
        print("  Поток не TTY: ждём 15 с — перезапусти прокси с -ml-enabled (или задай --pause-sec 0 и сделай это заранее)")
        time.sleep(15)


def run_ndpi(pcap_path: str) -> list[DPIResult]:
    """Запускает ndpiReader на pcap файле и парсит результаты."""
    try:
        result = subprocess.run(
            ["ndpiReader", "-i", pcap_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        print("  [FAIL] ndpiReader не найден. Установи ndpi-tools (или nDPI с ndpiReader).")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  [FAIL] ndpiReader превысил таймаут.")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] nDPI: {e}")
        sys.exit(1)

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    debug_path = pcap_path.replace(".pcap", "_ndpi_debug.txt")
    with open(debug_path, "w") as f:
        f.write(combined)
    print(f"  [DBG] Полный вывод ndpiReader → {debug_path}")
    parsed = parse_ndpi_output(combined)
    if not parsed:
        snippet = combined[:500].replace("\n", " ")
        print("  [WARN] Не удалось разобрать сводку nDPI (или в pcap нет потоков).")
        print(f"  Фрагмент вывода: {snippet}")
    return parsed


def _ndpi_detected_block(text: str) -> str:
    """Ограничиваем разбор блоком «Detected protocols», чтобы не спутать со «Protocol statistics»."""
    key = "Detected protocols:"
    start = text.find(key)
    if start == -1:
        return text
    start = start + len(key)
    end = text.find("Protocol statistics:", start)
    if end == -1:
        end = len(text)
    return text[start:end]


def parse_ndpi_output(output: str) -> list[DPIResult]:
    """Парсит вывод ndpiReader (nDPI 4.x и старый формат)."""
    block = _ndpi_detected_block(output)
    flow_by_label: dict[str, int] = {}

    # nDPI 4.x: "  TLS    packets:  12  bytes: 1234  flows: 2"
    pat_ndpi4 = re.compile(
        r"^\s*(\S+.*?)\s+packets:\s+\d+\s+bytes:\s+\d+\s+flows:\s+(\d+)\s*$",
        re.MULTILINE,
    )
    for m in pat_ndpi4.finditer(block):
        raw = m.group(1).strip()
        if not raw:
            continue
        label = raw.split()[0]   # берём первое слово: "TLS.HTTPS" → "TLS.HTTPS"
        # Нормализуем составные имена: "TLS.HTTPS" → "HTTPS", "DNS.Google" → "DNS"
        if "." in label:
            label = label.split(".")[-1]
        n = int(m.group(2))
        flow_by_label[label] = flow_by_label.get(label, 0) + n

    # Старый формат: "TLS.HTTPS   12 flows" (без bytes)
    if not flow_by_label:
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[-1] == "flows":
                try:
                    raw_label = parts[0]
                    label = raw_label.split(".")[-1]
                    n = int(parts[-2])
                    flow_by_label[label] = flow_by_label.get(label, 0) + n
                except (ValueError, IndexError):
                    continue

    # Ещё fallback: "  10 TLS flows" (встречается в нестандартных сборках)
    if not flow_by_label:
        pat_alt = re.compile(r"(\d+)\s+(\w+)\s+flows", re.IGNORECASE)
        for m in pat_alt.finditer(output):
            label = m.group(2)
            n = int(m.group(1))
            flow_by_label[label] = flow_by_label.get(label, 0) + n

    if not flow_by_label:
        return []

    results = [
        DPIResult(label=k, confidence=0.0, flow_count=v)
        for k, v in sorted(flow_by_label.items(), key=lambda x: -x[1])
    ]
    tot = sum(r.flow_count for r in results) or 1
    for r in results:
        r.confidence = r.flow_count / tot
    return results


def generate_synthetic_dpi(pcap_path: str) -> list[DPIResult]:
    """
    Синтетические данные DPI когда ndpiReader не установлен.
    Имитирует реалистичное поведение — до ML DPI уверен, после ML — нет.
    """
    is_after = "after" in pcap_path
    if is_after:
        # После ML: больше Unknown, меньше конкретных классов
        return [
            DPIResult("Unknown",  0.55, 55),
            DPIResult("HTTPS",    0.25, 25),
            DPIResult("Telegram", 0.10, 10),
            DPIResult("VPN",      0.10, 10),
        ]
    else:
        # До ML: DPI уверенно классифицирует
        return [
            DPIResult("HTTPS",    0.60, 60),
            DPIResult("Telegram", 0.25, 25),
            DPIResult("VPN",      0.10, 10),
            DPIResult("Unknown",  0.05,  5),
        ]


def run_dpi_tests(cfg: Config, results: TestResults):
    print_section("Шаг 3: DPI анализ трафика до/после ML")

    os.makedirs(cfg.output_dir, exist_ok=True)
    pcap_before = f"{cfg.output_dir}/before_ml.pcap"
    pcap_after  = f"{cfg.output_dir}/after_ml.pcap"

    if cfg.mock:
        print("  [mock] Используем синтетические DPI данные")
        results.dpi_before = generate_synthetic_dpi("before")
        results.dpi_after  = generate_synthetic_dpi("after")
    else:
        # Трафик БЕЗ ML
        print("\n  --- БЕЗ ML трансформации ---")
        capture_traffic(cfg, pcap_before, with_ml=False, n_requests=cfg.n_requests)
        results.dpi_before = run_ndpi(pcap_before)
        if not results.dpi_before:
            print("  [FAIL] Нет корректной сводки nDPI до ML — проверь pcap и ndpiReader.")
            sys.exit(1)

        # Трафик С ML (предполагаем что прокси уже запущен с -ml-enabled)
        print("\n  --- С ML трансформацией ---")
        print("  (убедись что прокси запущен с флагом -ml-enabled)")
        pause_between_ml_phases(cfg)
        capture_traffic(cfg, pcap_after, with_ml=True, n_requests=cfg.n_requests)
        results.dpi_after = run_ndpi(pcap_after)
        if not results.dpi_after:
            print("  [FAIL] Нет корректной сводки nDPI после ML — проверь pcap и ndpiReader.")
            sys.exit(1)

    # Выводим таблицу сравнения
    print(f"\n  {'Класс':<15} {'До ML':>12} {'После ML':>12} {'Изменение':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    before_map = {r.label: r.confidence for r in results.dpi_before}
    after_map  = {r.label: r.confidence for r in results.dpi_after}
    all_labels = sorted(set(list(before_map) + list(after_map)))

    for label in all_labels:
        b = before_map.get(label, 0.0)
        a = after_map.get(label, 0.0)
        delta = a - b
        arrow = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else "  ")
        print(f"  {label:<15} {b:>11.1%} {a:>11.1%} {arrow} {delta:>+9.1%}")

    # Ключевая метрика: изменение доли Unknown
    unknown_before = before_map.get("Unknown", 0.0)
    unknown_after  = after_map.get("Unknown", 0.0)
    improvement    = unknown_after - unknown_before
    print(f"\n  Доля неклассифицированного трафика: {unknown_before:.0%} → {unknown_after:.0%} (+{improvement:.0%})")
    check_ok(improvement > 0.1, f"ML улучшил маскировку на {improvement:.0%}", f"Слабый эффект: +{improvement:.0%}")

    # Заполняем confidences для графиков
    results.confidences_before = [r.confidence for r in results.dpi_before if r.label != "Unknown"]
    results.confidences_after  = [r.confidence for r in results.dpi_after  if r.label != "Unknown"]


# ── Шаг 4: графики ────────────────────────────────────────────────────────────

def build_plots(cfg: Config, results: TestResults):
    print_section("Шаг 4: построение графиков")

    os.makedirs(cfg.output_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Адаптивная маскировка сетевого трафика — результаты тестирования",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── График 1: DPI классификация до ML ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if results.dpi_before:
        labels  = [r.label for r in results.dpi_before]
        sizes   = [r.confidence for r in results.dpi_before]
        colors  = ["#E24B4A" if l != "Unknown" else "#888780" for l in labels]
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=colors, startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"}
        )
        for t in autotexts:
            t.set_fontsize(9)
    ax1.set_title("DPI до ML", fontweight="bold")

    # ── График 2: DPI классификация после ML ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if results.dpi_after:
        labels  = [r.label for r in results.dpi_after]
        sizes   = [r.confidence for r in results.dpi_after]
        colors  = ["#E24B4A" if l != "Unknown" else "#1D9E75" for l in labels]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=colors, startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"}
        )
        for t in autotexts:
            t.set_fontsize(9)
    ax2.set_title("DPI после ML", fontweight="bold")

    # ── График 3: сравнение уверенности классификатора ────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ["До ML", "После ML"]

    known_before = sum(r.confidence for r in results.dpi_before if r.label != "Unknown")
    known_after  = sum(r.confidence for r in results.dpi_after  if r.label != "Unknown")

    bars = ax3.bar(categories, [known_before, known_after],
                   color=["#E24B4A", "#1D9E75"], width=0.5,
                   edgecolor="white", linewidth=0.5)
    ax3.set_ylabel("Доля классифицированного трафика")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Эффективность маскировки", fontweight="bold")
    for bar, val in zip(bars, [known_before, known_after]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.0%}", ha="center", va="bottom", fontweight="bold")

    # ── График 4: латентность ML агента ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if results.ml_latencies_ms:
        ax4.hist(results.ml_latencies_ms, bins=20, color="#7F77DD",
                 edgecolor="white", linewidth=0.5)
        ax4.axvline(np.mean(results.ml_latencies_ms), color="#3C3489",
                    linestyle="--", linewidth=1.5,
                    label=f"avg={np.mean(results.ml_latencies_ms):.1f}ms")
        ax4.set_xlabel("Латентность (мс)")
        ax4.set_ylabel("Количество запросов")
        ax4.set_title("Латентность ML агента", fontweight="bold")
        ax4.legend(fontsize=9)

    # ── График 5: параметры трансформации от размера пакета ───────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if results.ml_predictions:
        sizes    = [p.packet_size  for p in results.ml_predictions]
        paddings = [p.padding_bytes for p in results.ml_predictions]
        delays   = [p.delay_ms      for p in results.ml_predictions]
        chunks   = [p.chunk_size    for p in results.ml_predictions]

        x = range(len(sizes))
        ax5.bar([i - 0.25 for i in x], [p / 1024 * 100 for p in paddings],
                width=0.25, label="Padding (% от max)", color="#7F77DD", alpha=0.8)
        ax5.bar([i        for i in x], [d / 200  * 100 for d in delays],
                width=0.25, label="Delay (% от max)",   color="#EF9F27", alpha=0.8)
        ax5.bar([i + 0.25 for i in x], [c / 8192 * 100 for c in chunks],
                width=0.25, label="Chunk (% от max)",   color="#1D9E75", alpha=0.8)

        ax5.set_xticks(list(x))
        ax5.set_xticklabels([f"{s}B" for s in sizes], rotation=45, fontsize=8)
        ax5.set_ylabel("% от максимума")
        ax5.set_title("Параметры трансформации", fontweight="bold")
        ax5.legend(fontsize=8)

    # ── График 6: энтропия → padding (ключевая зависимость) ──────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if results.ml_predictions:
        entropies = [p.entropy       for p in results.ml_predictions]
        paddings  = [p.padding_bytes for p in results.ml_predictions]
        scatter = ax6.scatter(entropies, paddings, c=[p.delay_ms for p in results.ml_predictions],
                              cmap="YlOrRd", s=80, edgecolors="white", linewidth=0.5)
        plt.colorbar(scatter, ax=ax6, label="delay_ms")
        ax6.set_xlabel("Энтропия пакета")
        ax6.set_ylabel("Padding (байт)")
        ax6.set_title("Энтропия → стратегия ML", fontweight="bold")

    out_path = f"{cfg.output_dir}/results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Графики сохранены: {out_path}")


# ── Шаг 5: итоговый отчёт ────────────────────────────────────────────────────

def print_summary(results: TestResults):
    print_section("Итог")

    if results.ml_latencies_ms:
        print(f"  ML агент:          avg={np.mean(results.ml_latencies_ms):.1f}ms  "
              f"p95={np.percentile(results.ml_latencies_ms, 95):.1f}ms")

    if results.dpi_before and results.dpi_after:
        before_map = {r.label: r.confidence for r in results.dpi_before}
        after_map  = {r.label: r.confidence for r in results.dpi_after}

        unknown_before = before_map.get("Unknown", 0.0)
        unknown_after  = after_map.get("Unknown", 0.0)
        improvement    = unknown_after - unknown_before

        known_before = 1.0 - unknown_before
        known_after  = 1.0 - unknown_after
        reduction    = (known_before - known_after) / max(known_before, 0.001) * 100

        print(f"  Неклассифицировано: {unknown_before:.0%} → {unknown_after:.0%} "
              f"(+{improvement:.0%})")
        print(f"  Уверенность DPI:    снизилась на {reduction:.0f}%")

        if improvement > 0.2:
            verdict = "ОТЛИЧНО — ML существенно снизил точность DPI"
        elif improvement > 0.1:
            verdict = "ХОРОШО — ML заметно влияет на классификацию"
        else:
            verdict = "СЛАБО — нужно доообучить модель"
        print(f"\n  Вердикт: {verdict}")

    print(f"\n  Результаты сохранены в: {args_global.output_dir}/")
    print(f"  Главный график:         {args_global.output_dir}/results.png")


# ── Main ──────────────────────────────────────────────────────────────────────

args_global: Config = None

def main():
    global args_global

    parser = argparse.ArgumentParser(description="E2E тест адаптивной маскировки трафика")
    parser.add_argument("--proxy-host",    default="127.0.0.1")
    parser.add_argument("--proxy-port",    default=1080, type=int)
    parser.add_argument("--ml-host",       default="127.0.0.1")
    parser.add_argument("--ml-port",       default=8000, type=int)
    parser.add_argument("--output-dir",    default="test_results")
    parser.add_argument("--n-requests",    default=100, type=int)
    parser.add_argument(
        "--capture-iface",
        default="any",
        help="Интерфейс tcpdump (по умолчанию any — весь трафик; на VPS часто eth0).",
    )
    parser.add_argument(
        "--pause-sec",
        type=int,
        default=None,
        help="Пауза между фазами DPI (перезапуск прокси с -ml-enabled). "
        "По умолчанию: Enter в интерактиве, иначе 15 с.",
    )
    parser.add_argument("--mock",          action="store_true",
                        help="Без реального трафика — только ML метрики")
    args = parser.parse_args()

    cfg = Config(
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        ml_host=args.ml_host,
        ml_port=args.ml_port,
        output_dir=args.output_dir,
        n_requests=args.n_requests,
        capture_iface=args.capture_iface,
        mock=args.mock,
        pause_sec=args.pause_sec,
    )
    args_global = cfg

    print("\n" + "=" * 60)
    print("  Тест адаптивной маскировки сетевого трафика")
    print("=" * 60)

    results = TestResults()

    check_components(cfg)
    test_ml_predictions(cfg, results)
    run_dpi_tests(cfg, results)
    build_plots(cfg, results)
    print_summary(results)


if __name__ == "__main__":
    main()
