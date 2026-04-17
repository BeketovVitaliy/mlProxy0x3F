package ml

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"
)

// Client общается с Python ML-агентом.
// Протокол: простой HTTP/JSON (gRPC можно добавить позже,
// для диплома JSON проще отлаживать и объяснять).
type Client struct {
	baseURL    string
	httpClient *http.Client
}

type Params struct {
	DelayMs   int
	ChunkSize int
}

var DefaultParams = Params{
	DelayMs:   0,
	ChunkSize: 0,
}

func NewClient(addr string) (*Client, error) {
	return &Client{
		baseURL: "http://" + addr,
		httpClient: &http.Client{
			Timeout: 50 * time.Millisecond, // жёсткий таймаут — ML не должен тормозить трафик
		},
	}, nil
}

func (c *Client) Close() {}

// featuresRequest — фичи которые отправляем ML-агенту.
type featuresRequest struct {
	PacketSize int     `json:"packet_size"`
	Entropy    float64 `json:"entropy"`
}

// paramsResponse — ответ агента с параметрами трансформации.
type paramsResponse struct {
	PaddingBytes int `json:"padding_bytes"`
	DelayMs      int `json:"delay_ms"`
	ChunkSize    int `json:"chunk_size"`
}

// GetParams отправляет фичи пакета ML-агенту и получает параметры трансформации.
func (c *Client) GetParams(data []byte) (Params, error) {
	req := featuresRequest{
		PacketSize: len(data),
		Entropy:    calcEntropy(data),
	}

	body, err := json.Marshal(req)
	if err != nil {
		return DefaultParams, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		c.baseURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return DefaultParams, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return DefaultParams, fmt.Errorf("ml agent unavailable: %w", err)
	}
	defer resp.Body.Close()

	var result paramsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return DefaultParams, err
	}

	return Params{
		DelayMs:   clamp(result.DelayMs, 0, 200),
		ChunkSize: clamp(result.ChunkSize, 0, 8192),
	}, nil
}

// calcEntropy считает энтропию Шеннона для байтового массива.
func calcEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}

	var freq [256]int
	for _, b := range data {
		freq[b]++
	}

	var entropy float64
	n := float64(len(data))
	for _, count := range freq {
		if count == 0 {
			continue
		}
		p := float64(count) / n
		entropy -= p * math.Log2(p)
	}
	return entropy
}

func clamp(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
