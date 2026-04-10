package socks5

import (
	"context"
	"log"
	"net"
	"sync"

	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/transform"
)

// Server принимает входящие SOCKS5 соединения.
type Server struct {
	addr    string
	engine  *transform.Engine
	ln      net.Listener
	wg      sync.WaitGroup
	cancel  context.CancelFunc
}

func NewServer(addr string, engine *transform.Engine) *Server {
	return &Server{addr: addr, engine: engine}
}

func (s *Server) ListenAndServe() error {
	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return err
	}
	s.ln = ln

	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel

	for {
		conn, err := ln.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				// нормальное завершение
				s.wg.Wait()
				return nil
			default:
				log.Printf("Accept error: %v", err)
				continue
			}
		}

		s.wg.Add(1)
		go func(c net.Conn) {
			defer s.wg.Done()
			h := newHandler(c, s.engine)
			if err := h.Handle(ctx); err != nil {
				log.Printf("[%s] handler error: %v", c.RemoteAddr(), err)
			}
		}(conn)
	}
}

func (s *Server) Stop() {
	if s.cancel != nil {
		s.cancel()
	}
	if s.ln != nil {
		s.ln.Close()
	}
}
