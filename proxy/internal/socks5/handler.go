package socks5

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"time"

	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/transform"
)

const (
	socks5Version = 0x05

	// Auth методы
	authNone     = 0x00
	authNoAccept = 0xFF

	// Команды
	cmdConnect = 0x01

	// Типы адресов
	atypIPv4   = 0x01
	atypDomain = 0x03
	atypIPv6   = 0x04

	// Ответы
	repSuccess = 0x00
	repFailure = 0x01
)

type handler struct {
	client net.Conn
	engine *transform.Engine
}

func newHandler(client net.Conn, engine *transform.Engine) *handler {
	return &handler{client: client, engine: engine}
}

// Handle выполняет SOCKS5 handshake и запускает туннель.
func (h *handler) Handle(ctx context.Context) error {
	defer h.client.Close()
	h.client.SetDeadline(time.Now().Add(30 * time.Second))

	// 1. Согласование метода аутентификации
	if err := h.negotiateAuth(); err != nil {
		return fmt.Errorf("auth negotiation: %w", err)
	}

	// 2. Читаем запрос
	target, err := h.readRequest()
	if err != nil {
		h.sendReply(repFailure)
		return fmt.Errorf("read request: %w", err)
	}

	// 3. Подключаемся к целевому серверу
	remote, err := net.DialTimeout("tcp", target, 15*time.Second)
	if err != nil {
		h.sendReply(repFailure)
		return fmt.Errorf("dial %s: %w", target, err)
	}
	defer remote.Close()

	// TCP_NODELAY гарантирует что каждый Write уйдёт отдельным
	// TCP сегментом. Без этого Nagle может склеить мелкие фрагменты
	// обратно в один пакет, и DPI увидит целый ClientHello.
	if tc, ok := remote.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
	}

	// 4. Отправляем клиенту успешный ответ
	if err := h.sendReply(repSuccess); err != nil {
		return err
	}

	// 5. Снимаем дедлайн — теперь работает туннель
	h.client.SetDeadline(time.Time{})
	remote.SetDeadline(time.Time{})

	// 6. Запускаем двунаправленный туннель с трансформацией
	return h.tunnel(ctx, remote)
}

// negotiateAuth — согласование: поддерживаем только «без аутентификации».
func (h *handler) negotiateAuth() error {
	// VER | NMETHODS
	header := make([]byte, 2)
	if _, err := io.ReadFull(h.client, header); err != nil {
		return err
	}
	if header[0] != socks5Version {
		return errors.New("not SOCKS5")
	}

	// METHODS
	methods := make([]byte, header[1])
	if _, err := io.ReadFull(h.client, methods); err != nil {
		return err
	}

	for _, m := range methods {
		if m == authNone {
			_, err := h.client.Write([]byte{socks5Version, authNone})
			return err
		}
	}

	h.client.Write([]byte{socks5Version, authNoAccept})
	return errors.New("no supported auth method")
}

// readRequest парсит SOCKS5 CONNECT запрос и возвращает "host:port".
func (h *handler) readRequest() (string, error) {
	// VER | CMD | RSV | ATYP
	header := make([]byte, 4)
	if _, err := io.ReadFull(h.client, header); err != nil {
		return "", err
	}
	if header[0] != socks5Version {
		return "", errors.New("not SOCKS5")
	}
	if header[1] != cmdConnect {
		return "", fmt.Errorf("unsupported command: %d", header[1])
	}

	var host string
	switch header[3] {
	case atypIPv4:
		addr := make([]byte, 4)
		if _, err := io.ReadFull(h.client, addr); err != nil {
			return "", err
		}
		host = net.IP(addr).String()

	case atypDomain:
		lenBuf := make([]byte, 1)
		if _, err := io.ReadFull(h.client, lenBuf); err != nil {
			return "", err
		}
		domain := make([]byte, lenBuf[0])
		if _, err := io.ReadFull(h.client, domain); err != nil {
			return "", err
		}
		host = string(domain)

	case atypIPv6:
		addr := make([]byte, 16)
		if _, err := io.ReadFull(h.client, addr); err != nil {
			return "", err
		}
		host = net.IP(addr).String()

	default:
		return "", fmt.Errorf("unknown address type: %d", header[3])
	}

	// PORT (2 байта big-endian)
	portBuf := make([]byte, 2)
	if _, err := io.ReadFull(h.client, portBuf); err != nil {
		return "", err
	}
	port := binary.BigEndian.Uint16(portBuf)

	return net.JoinHostPort(host, strconv.Itoa(int(port))), nil
}

// sendReply отправляет SOCKS5 ответ клиенту.
func (h *handler) sendReply(rep byte) error {
	// VER REP RSV ATYP BND.ADDR(4) BND.PORT(2)
	reply := []byte{
		socks5Version, rep, 0x00,
		atypIPv4, 0, 0, 0, 0,
		0, 0,
	}
	_, err := h.client.Write(reply)
	return err
}

// tunnel запускает двунаправленный поток данных с трансформацией исходящего трафика.
func (h *handler) tunnel(ctx context.Context, remote net.Conn) error {
	errCh := make(chan error, 2)

	// Клиент → Удалённый (трансформируем)
	go func() {
		err := h.engine.Copy(ctx, remote, h.client)
		errCh <- err
	}()

	// Удалённый → Клиент (без трансформации — ответы сервера не трогаем)
	go func() {
		_, err := io.Copy(h.client, remote)
		errCh <- err
	}()

	// Ждём завершения любого из направлений
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}
