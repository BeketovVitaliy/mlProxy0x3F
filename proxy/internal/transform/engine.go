package transform

import (
	"context"
	"io"
	"log"
	"math/rand"
	"time"

	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/ml"
)

const (
	// Размер фрагмента для TLS ClientHello.
	// 100 байт — достаточно мало чтобы DPI не смог распарсить
	// ClientHello из одного TCP сегмента, но достаточно чтобы
	// не сломать TCP стек.
	firstPacketChunkSize = 100

	// Пауза между фрагментами первого пакета.
	// Нужна чтобы ОС отправила каждый фрагмент отдельным TCP сегментом
	// (иначе Nagle может склеить).
	firstPacketDelayMs = 5

	// Сколько байт от начала соединения считаем «первым пакетом».
	// TLS ClientHello обычно 200-500 байт, но может быть до 4KB
	// с расширениями.
	firstPacketThreshold = 4096
)

// Params хранит параметры одной трансформации.
type Params struct {
	DelayMs   int
	ChunkSize int
}

var DefaultParams = Params{
	DelayMs:   0,
	ChunkSize: 0,
}

// Engine применяет трансформации к потоку данных.
type Engine struct {
	ml *ml.Client
}

func NewEngine(mlClient *ml.Client) *Engine {
	return &Engine{ml: mlClient}
}

// Copy читает из src, применяет трансформацию и пишет в dst.
func (e *Engine) Copy(ctx context.Context, dst io.Writer, src io.Reader) error {
	buf := make([]byte, 32*1024)
	bytesSent := 0

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, err := src.Read(buf)
		if n > 0 {
			chunk := buf[:n]

			params := e.getParams(chunk)

			// Первые байты соединения (TLS ClientHello) — фрагментируем
			// агрессивно независимо от ML, чтобы сломать DPI парсинг.
			if bytesSent < firstPacketThreshold && e.ml != nil {
				params.ChunkSize = firstPacketChunkSize
				if params.DelayMs < firstPacketDelayMs {
					params.DelayMs = firstPacketDelayMs
				}
				log.Printf("[ML] first-packet frag: %d bytes → chunks of %d", n, params.ChunkSize)
			}
			bytesSent += n

			if writeErr := e.applyAndSend(ctx, dst, chunk, params); writeErr != nil {
				return writeErr
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

// getParams запрашивает ML-агента или возвращает дефолты.
func (e *Engine) getParams(data []byte) Params {
	if e.ml == nil {
		return DefaultParams
	}

	p, err := e.ml.GetParams(data)
	if err != nil {
		// ML недоступен — продолжаем без трансформации
		return DefaultParams
	}
	return Params{
		DelayMs:   p.DelayMs,
		ChunkSize: p.ChunkSize,
	}
}

// applyAndSend применяет задержку, фрагментацию и отправляет данные.
func (e *Engine) applyAndSend(ctx context.Context, dst io.Writer, data []byte, p Params) error {
	// 1. Задержка (имитирует другой паттерн межпакетных задержек)
	if p.DelayMs > 0 {
		select {
		case <-time.After(time.Duration(p.DelayMs) * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	// 2. Фрагментация — дробим буфер на чанки
	chunks := fragment(data, p.ChunkSize)

	for _, chunk := range chunks {
		if _, err := dst.Write(chunk); err != nil {
			return err
		}

		// Небольшая пауза между фрагментами если есть фрагментация
		if p.ChunkSize > 0 && p.DelayMs > 0 {
			jitter := time.Duration(rand.Intn(p.DelayMs/2+1)) * time.Millisecond
			select {
			case <-time.After(jitter):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	return nil
}

// fragment делит slice на части по chunkSize байт.
// Если chunkSize <= 0 — возвращает один чанк.
func fragment(data []byte, chunkSize int) [][]byte {
	if chunkSize <= 0 || chunkSize >= len(data) {
		return [][]byte{data}
	}

	var chunks [][]byte
	for len(data) > 0 {
		end := chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunks = append(chunks, data[:end])
		data = data[end:]
	}
	return chunks
}

