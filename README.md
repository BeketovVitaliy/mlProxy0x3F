# Адаптивная маскировка сетевого трафика с использованием ML

> Дипломный проект. Тема: *Методы адаптивной трансформации статистических характеристик сетевого трафика с использованием моделей машинного обучения*

---

## Содержание

1. [Что это и зачем](#1-что-это-и-зачем)
2. [Текущее состояние](#2-текущее-состояние)
3. [Архитектура](#3-архитектура)
4. [Структура проекта](#4-структура-проекта)
5. [Как запустить](#5-как-запустить)
6. [ML часть (supervised)](#6-ml-часть-supervised)
7. [Прокси](#7-прокси)
8. [Тестирование](#8-тестирование)
9. [Результаты и выводы](#9-результаты-и-выводы)
10. [Roadmap: переход к RL](#10-roadmap-переход-к-rl)

---

## 1. Что это и зачем

**DPI (Deep Packet Inspection)** — технология классификации сетевого трафика.
Современные DPI (nDPI, ТСПУ) определяют протокол по:

- Структуре протокола (TLS ClientHello, JA3 fingerprint, SNI)
- Статистике потока (размеры пакетов, IAT, энтропия)
- Паттернам поведения (burst size, byte rate)

**Задача проекта** — построить прокси-сервер с ML-агентом, который
адаптивно трансформирует трафик для снижения точности DPI-классификации.

---

## 2. Текущее состояние

### Что работает

- **SOCKS5 прокси** (Go) — полнофункциональный, с поддержкой IPv4/IPv6/domain
- **ML агент** (Python/Flask) — принимает фичи пакета, возвращает параметры трансформации
- **Supervised трансформер** — обучен через adversarial training, снижает confidence
  внутреннего DPI-классификатора с 1.0 до 0.50 (случайное угадывание)
- **Трансформации** — TCP фрагментация + timing jitter (TLS-безопасные)
- **First-packet fragmentation** — агрессивная фрагментация TLS ClientHello (100 байт/фрагмент)
- **E2E тест** — автоматический захват трафика + анализ через nDPI

### Результаты тестирования

| Метрика | Значение |
|---------|----------|
| Classifier confidence (внутренний) | 1.0 → **0.50** (успех) |
| CW margin (logit gap top1-top2) | ∞ → **0.0** (успех) |
| nDPI TLS classification | 99.3% → 98.7% (**минимальный эффект**) |
| ML agent latency | avg **17ms**, p95 **18ms** |

### Ключевой вывод

Supervised трансформер **успешно обманывает внутренний классификатор**, но
**не влияет на nDPI**, потому что:

1. nDPI классифицирует TLS по **парсингу протокола** (ClientHello, JA3, SNI),
   а не по flow-статистике
2. nDPI делает **TCP reassembly** — фрагментация TCP сегментов не мешает
   парсингу протокола
3. Supervised модель тренируется на **неправильном таргете** — внутренний
   классификатор не представляет реальный DPI

**Вывод: необходим переход к RL** с обратной связью от реального DPI.

---

## 3. Архитектура

```
┌──────────┐        ┌────────────────────────────────┐        ┌──────────┐
│  Клиент  │──:1080─▶  Go прокси (SOCKS5)            │──────▶ │  Сервер  │
└──────────┘        │    ├── Transform Engine         │        └──────────┘
                    │    │     ├── first-packet frag   │
                    │    │     ├── TCP chunking         │
                    │    │     └── timing jitter        │
                    │    └── ML Client (HTTP/JSON)     │
                    │         ↕                        │
                    │    Python ML Agent (:8000)       │
                    │    ├── /predict (inference)      │
                    │    ├── /health                   │
                    │    └── TrafficTransformer (PT)   │
                    └────────────────────────────────┘
```

### Трансформации (TLS-безопасные)

| Трансформация | Как работает | Эффект |
|---------------|-------------|--------|
| First-packet fragmentation | ClientHello → куски по 100 байт с 5ms паузой | Ломает DPI без TCP reassembly |
| TCP chunking | Данные дробятся на куски по chunk_size | Меняет паттерн размеров пакетов |
| Timing jitter | Задержка delay_ms + рандом между фрагментами | Меняет IAT |

Padding (добавление мусорных байт) **убран** — он ломает TLS, т.к. получатель
не знает о дополнительных байтах в зашифрованном потоке.

---

## 4. Структура проекта

```
.
├── proxy/                          # Go прокси-сервер
│   ├── cmd/proxy/
│   │   └── main.go                 # точка входа, флаги
│   ├── internal/
│   │   ├── socks5/
│   │   │   ├── server.go           # TCP listener
│   │   │   └── handler.go          # SOCKS5 handshake + tunnel + TCP_NODELAY
│   │   ├── transform/
│   │   │   └── engine.go           # фрагментация + jitter + first-packet logic
│   │   └── ml/
│   │       └── client.go           # HTTP клиент к ML агенту + Shannon entropy
│   └── go.mod
│
├── ml/                             # Python ML агент
│   ├── data/
│   │   └── prepare.py              # синтетический датасет (5 классов)
│   ├── models/
│   │   ├── classifier.py           # DPI-имитатор (10 фич → 5 классов)
│   │   └── transformer.py          # adversarial трансформер + CW loss
│   ├── train/
│   │   ├── train_classifier.py     # обучение классификатора
│   │   └── train_transformer.py    # adversarial обучение (CW + temperature)
│   ├── server/
│   │   └── app.py                  # Flask API
│   ├── utils/
│   │   └── features.py             # признаки трафика
│   ├── saved_models/               # веса (создаётся при обучении)
│   └── requirements.txt
│
├── test_e2e.py                     # E2E тест с nDPI
└── test_results/                   # pcap + графики
```

---

## 5. Как запустить

### Требования

- Go 1.22+
- Python 3.11+ с venv
- `apt install tcpdump ndpi-tools` (для тестов)

### Быстрый старт

```bash
# 1. Python зависимости
cd ml && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Обучение моделей
python train/train_classifier.py     # ~2 мин
python train/train_transformer.py    # ~5 мин

# 3. ML агент
python server/app.py &

# 4. Go прокси (новый терминал)
cd proxy && go run ./cmd/proxy -addr 0.0.0.0:1080 -ml-enabled

# 5. Проверка
curl --socks5 127.0.0.1:1080 https://ifconfig.me
```

---

## 6. ML часть (supervised)

### Классификатор (DPI-имитатор)

- Input: 10 flow-фич (mean/std/min/max размеров и IAT, packet_count, bytes_total)
- Output: 5 классов (HTTPS, Telegram, VPN, Tor, Unknown)
- Архитектура: 3× (Linear + BatchNorm + ReLU + Dropout) → Linear(5)
- Val accuracy: ~98.6%
- Поддерживает **temperature scaling** для adversarial обучения

### Трансформер

- Residual: `out = clamp(x + tanh(delta_net(x)) * max_delta, 0, 1)`
- max_delta=0.5 — ограничивает максимальное изменение каждого признака
- 7594 параметра

### Adversarial loss (CW + entropy)

```
L = γ * L_cw + (1-γ) * L_entropy + α * L_utility + β * L_boundary
```

- **L_cw** (Carlini-Wagner): `max(logit_top1 - logit_top2, 0)` — работает на логитах,
  градиент всегда ненулевой
- **L_entropy**: `log(K) - H(softmax(logits/T))` — с temperature T=5.0
- **L_utility**: L2 штраф за большие дельты (α=0.1)
- **L_boundary**: штраф за выход за [0.05, 0.95] (β=0.1)

### Ключевые фиксы (история)

| Версия | Проблема | Решение |
|--------|----------|---------|
| v1-v2 | Mode collapse | Residual connection + noise |
| v3 | Градиенты = 0 (softmax saturation) | Temperature scaling T=5.0 |
| v4 | Entropy loss не работает при conf=1.0 | CW logit-space loss |

---

## 7. Прокси

### Go → ML коммуникация

```
Go: POST /predict {"packet_size": 1400, "entropy": 7.9}
ML: {"delay_ms": 7, "chunk_size": 712, "padding_bytes": 0}
```

- Таймаут 50ms — при недоступности ML прокси работает без трансформации
- Entropy считается через `math.Log2` (исправлен баг с кастомным log2)
- Адрес ML агента по умолчанию: `localhost:8000` (исправлен баг с портом 50051)

### First-packet fragmentation

Первые 4KB соединения (TLS ClientHello) фрагментируются на куски по 100 байт
с 5ms паузой между ними. TCP_NODELAY на remote connection гарантирует
отправку каждого фрагмента отдельным TCP сегментом.

---

## 8. Тестирование

```bash
# Полный E2E тест (нужен запущенный ML агент + прокси)
python test_e2e.py --capture-iface eth0

# Шаги теста:
# 1. Проверка компонентов (ML агент, прокси)
# 2. Тест ML предсказаний (6 типов трафика, латентность)
# 3. DPI анализ: 100 запросов без ML → pcap → nDPI
# 4. DPI анализ: 100 запросов с ML → pcap → nDPI
# 5. Сравнение + графики
```

---

## 9. Результаты и выводы

### Supervised подход

**Успех:** трансформер научился обманывать внутренний DPI-классификатор
(confidence 1.0 → 0.50, margin → 0.0).

**Ограничение:** нет эффекта на реальный DPI (nDPI), потому что:
- nDPI парсит протокол (TLS handshake), не считает flow-статистику
- nDPI делает TCP reassembly — фрагментация TCP не помогает
- Тренировка на синтетическом классификаторе ≠ тренировка против реального DPI

### Вывод

Supervised adversarial подход ограничен тем, что **суррогатный классификатор
не представляет реальную DPI-систему**. Для адаптивного обхода DPI необходим
**Reinforcement Learning** с обратной связью от реального DPI (nDPI).

---

## 10. Roadmap: переход к RL

### Архитектура

```
                    ┌─────────────────────────────────────┐
                    │          RL Training Loop            │
                    │                                     │
  ┌──────┐  action  │  ┌───────┐  traffic  ┌──────────┐  │
  │  RL  │─────────▶│  │ Proxy │──────────▶│ tcpdump  │  │
  │Agent │  {chunk, │  │ (Go)  │           │ + nDPI   │  │
  │(PPO) │  delay}  │  └───────┘           └────┬─────┘  │
  │      │◀─────────│                           │         │
  └──────┘  reward  │   reward = f(nDPI_class)  │         │
                    └───────────────────────────┘
```

### RL формулировка

- **State**: `[packet_size, entropy, bytes_sent, conn_age, prev_class]`
- **Action**: `[chunk_size ∈ {64..1460}, delay_ms ∈ {0..50}, first_frag_size ∈ {50..500}]`
- **Reward**: `+1` если nDPI → Unknown, `-1` если nDPI → classified, `-0.1 * delay/50` (штраф за latency)
- **Алгоритм**: PPO (Proximal Policy Optimization) или SAC (Soft Actor-Critic)

### Что нужно реализовать

1. **RL Environment** (`ml/rl/env.py`) — обёртка gymnasium: отправляет трафик через прокси → захватывает pcap → nDPI → reward
2. **RL Agent** (`ml/rl/agent.py`) — PPO policy network
3. **Go proxy: расширение API** — `/action` endpoint для получения RL-действий (вместо `/predict`)
4. **Reward shaping** — баланс между маскировкой и latency

### Преимущества RL над supervised

| | Supervised | RL |
|---|---|---|
| Таргет | Внутренний классификатор | **Реальный nDPI** |
| Адаптивность | Нет | **Переобучается при обновлении DPI** |
| Feedback loop | Нет | **Замкнутый: action → DPI → reward** |
| Action space | Фиксированный маппинг | **Исследует новые стратегии** |
