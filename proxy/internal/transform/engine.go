package transform

import (
	"context"
	"io"
	"math/rand"
	"time"

	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/ml"
)

// Params хранит параметры одной трансформации.
// ML-агент возвращает эти значения; без агента используются дефолты.
type Params struct {
	// PaddingBytes — сколько мусорных байт добавить после полезной нагрузки.
	// DPI видит увеличенный нерегулярный размер пакета.
	PaddingBytes int

	// DelayMs — задержка перед отправкой чанка (имитирует другой паттерн IAT).
	DelayMs int

	// ChunkSize — на сколько байт дробить исходный буфер (фрагментация).
	// 0 = не дробить.
	ChunkSize int
}

// DefaultParams — безопасные дефолты когда ML выключен.
var DefaultParams = Params{
	PaddingBytes: 0,
	DelayMs:      0,
	ChunkSize:    0,
}

// Engine применяет трансформации к потоку данных.
type Engine struct {
	ml *ml.Client // может быть nil
}

func NewEngine(mlClient *ml.Client) *Engine {
	return &Engine{ml: mlClient}
}

// Copy читает из src, применяет трансформацию и пишет в dst.
// Это основная точка где ML влияет на трафик.
func (e *Engine) Copy(ctx context.Context, dst io.Writer, src io.Reader) error {
	buf := make([]byte, 32*1024) // 32KB буфер

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, err := src.Read(buf)
		if n > 0 {
			chunk := buf[:n]

			// Получаем параметры трансформации от ML или дефолты
			params := e.getParams(chunk)

			// Применяем трансформацию
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
		PaddingBytes: p.PaddingBytes,
		DelayMs:      p.DelayMs,
		ChunkSize:    p.ChunkSize,
	}
}

// applyAndSend применяет padding, задержку, фрагментацию и отправляет данные.
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
		// 3. Padding — добавляем случайные байты в конец чанка
		payload := addPadding(chunk, p.PaddingBytes)

		if _, err := dst.Write(payload); err != nil {
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

// addPadding добавляет n случайных байт в конец данных.
// Получатель должен знать как их убрать — для диплома это фиксированный маркер.
func addPadding(data []byte, n int) []byte {
	if n <= 0 {
		return data
	}
	padding := make([]byte, n)
	rand.Read(padding) // crypto/rand для production, math/rand достаточно для диплома
	return append(data, padding...)
}
