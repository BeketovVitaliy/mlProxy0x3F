# Промт для следующего чата: реализация RL агента

Скопируй текст ниже и вставь в новый чат.

---

## Контекст проекта

У меня есть проект `mlProxy0x3F` — SOCKS5 прокси (Go) с ML-агентом (Python/Flask),
который адаптивно трансформирует трафик для обхода DPI.

### Что уже готово и работает

**Go прокси** (`proxy/`):
- SOCKS5 сервер с двунаправленным туннелированием
- Transform Engine: TCP фрагментация + timing jitter (TLS-безопасные, без padding)
- First-packet fragmentation: первые 4KB (TLS ClientHello) дробятся на 100-байтовые куски с 5ms паузой
- TCP_NODELAY на remote connection
- ML Client: HTTP/JSON к Python агенту, таймаут 50ms, fallback на DefaultParams
- Shannon entropy считается через math.Log2

**Python ML** (`ml/`):
- DPIClassifier: 10 flow-фич → 5 классов (HTTPS/Telegram/VPN/Tor/Unknown), val_acc=98.6%
  - Поддерживает temperature scaling и get_logits()
- TrafficTransformer: residual (out = clamp(x + tanh(delta) * max_delta)), 7594 параметров
- CW + Entropy adversarial loss (работает на логитах, не на softmax)
- Flask сервер: POST /predict принимает {packet_size, entropy} → возвращает {delay_ms, chunk_size}
- Синтетический датасет: 50k потоков из статистических профилей

**E2E тест** (`test_e2e.py`):
- Проверка компонентов → ML предсказания → tcpdump + nDPI до/после ML → графики

### Результаты supervised подхода (почему переходим к RL)

- Внутренний классификатор: confidence 1.0 → 0.50 ✅ (модель обманывает суррогат)
- Реальный nDPI: TLS 99.3% → 98.7% ❌ (минимальный эффект)

**Причины неудачи с nDPI:**
1. nDPI парсит протокол (TLS ClientHello, JA3, SNI), а не flow-статистику
2. nDPI делает TCP reassembly — фрагментация TCP сегментов не помогает
3. Supervised модель тренируется против суррогатного классификатора, не против реального DPI

### Протокол Go ↔ Python

```
Go:     POST http://localhost:8000/predict
        Body: {"packet_size": 1400, "entropy": 7.9}

Python: Response: {"padding_bytes": 0, "delay_ms": 7, "chunk_size": 712}
```

Go применяет: sleep(delay_ms) → fragment(data, chunk_size) → write каждый фрагмент.
Первый пакет каждого соединения: принудительно chunk=100, delay=5ms.

---

## Задача: реализовать RL агент

Мне нужно заменить supervised трансформер на RL агент (PPO или SAC),
который учится через обратную связь от реального nDPI.

### Архитектура RL

```
┌──────────┐  action   ┌───────┐  traffic  ┌──────────┐
│ RL Agent │──────────▶│ Proxy │──────────▶│ tcpdump  │
│ (Python) │           │ (Go)  │           │ + nDPI   │
│ PPO/SAC  │◀──────────│       │◀──────────│          │
└──────────┘  reward   └───────┘  classify └──────────┘
```

### RL формулировка (моё предложение, можно улучшить)

**State:**
- packet_size (normalized), entropy (normalized)
- bytes_sent_so_far (connection context)
- connection_age_ms
- previous_dpi_classification (one-hot или embedding)

**Action space (continuous):**
- chunk_size: 64..1460
- delay_ms: 0..50
- first_frag_size: 50..500 (только для первого пакета)

**Reward:**
- +1.0 если nDPI классифицировал поток как Unknown
- -1.0 если nDPI уверенно классифицировал протокол
- -0.1 * (delay_ms / 50) — штраф за latency
- бонус за разнообразие стратегий (entropy regularization)

### Что нужно реализовать

1. **`ml/rl/env.py`** — Gymnasium environment:
   - reset(): запускает новое соединение через прокси
   - step(action): применяет action к трансформации, отправляет трафик,
     захватывает pcap, анализирует nDPI, возвращает reward
   - Проблема: nDPI анализ медленный (~сек), нужна стратегия batch-оценки

2. **`ml/rl/agent.py`** — PPO или SAC policy:
   - Actor: state → action distribution
   - Critic: state → value estimate
   - Можно использовать stable-baselines3

3. **`ml/rl/train.py`** — тренировочный цикл:
   - Собирает trajectories через environment
   - Обновляет policy
   - Логирует reward, classification rates

4. **Обновить Go прокси** (если нужно):
   - Возможно новый endpoint для RL-действий
   - Или переиспользовать существующий /predict с расширенным action space

5. **`ml/rl/reward.py`** — вычисление reward:
   - Обёртка над ndpiReader
   - Парсинг вывода nDPI
   - Batch-обработка pcap файлов

### Ограничения и вопросы

- nDPI анализ занимает ~1 секунду на pcap — как ускорить training loop?
  Может быть: batch episodes, параллельные workers, simulated environment для warmup
- Нужно ли переделывать Go API или хватит текущего /predict?
- Стоит ли использовать stable-baselines3 или написать PPO с нуля?
- Как балансировать exploration vs exploitation — начать с агрессивных действий
  или с минимальных?

### Дополнительный контекст

- VPS Ubuntu, Go 1.22+, Python 3.13, PyTorch
- GPU: CUDA драйвер старый, тренировка на CPU (нейросети маленькие, это ок)
- nDPI установлен: `ndpiReader` доступен из CLI
- tcpdump требует sudo или capabilities

Помоги спроектировать и реализовать RL framework. Начни с архитектуры
Environment и Agent, потом переходи к коду. Посмотри текущий код в проекте
чтобы понять структуру и переиспользовать максимум.
