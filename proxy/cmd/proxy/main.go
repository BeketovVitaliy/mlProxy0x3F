package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/ml"
	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/socks5"
	"github.com/BeketovVitaliy/mlProxy0x3F/proxy/internal/transform"
)

func main() {
	addr := flag.String("addr", "0.0.0.0:1080", "SOCKS5 listen address")
	mlAddr := flag.String("ml", "localhost:8000", "ML agent HTTP address")
	mlEnabled := flag.Bool("ml-enabled", false, "Enable ML transform")
	flag.Parse()

	// ML client (опциональный)
	var mlClient *ml.Client
	if *mlEnabled {
		var err error
		mlClient, err = ml.NewClient(*mlAddr)
		if err != nil {
			log.Fatalf("ML client error: %v", err)
		}
		defer mlClient.Close()
		log.Printf("ML agent connected: %s", *mlAddr)
	}

	// Transform engine
	engine := transform.NewEngine(mlClient)

	// SOCKS5 сервер
	srv := socks5.NewServer(*addr, engine)

	// Graceful shutdown
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Println("Shutting down...")
		srv.Stop()
	}()

	log.Printf("SOCKS5 proxy listening on %s (ml=%v)", *addr, *mlEnabled)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
