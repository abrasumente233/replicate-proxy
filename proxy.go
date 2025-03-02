package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

const (
	targetURL = "https://fast-api.snova.ai/v1/chat/completions"
)

var (
	port = flag.Int("port", 9876, "Port to run the proxy server on")
)

func main() {
	flag.Parse()
	proxyAddr := fmt.Sprintf(":%d", *port)
	http.HandleFunc("/v1/chat/completions", proxyHandler)
	log.Printf("Proxy server listening on %s", proxyAddr)
	log.Fatal(http.ListenAndServe(proxyAddr, nil))
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	// Check for Bearer token
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		http.Error(w, "Unauthorized: Bearer token required", http.StatusUnauthorized)
		return
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	// Create a new request to the target URL
	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "Error creating proxy request", http.StatusInternalServerError)
		return
	}

	// Set headers for the proxy request
	proxyReq.Header.Set("Content-Type", "application/json")
	proxyReq.Header.Set("Authorization", fmt.Sprintf("Basic %s", token))

	// Send the request to the target server
	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, "Error sending request to target server", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Check if the response is streamed
	isStreaming := isStreamingResponse(body)

	// Set response headers
	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	// Handle streaming or non-streaming response
	if isStreaming {
		handleStreamingResponse(w, resp.Body)
	} else {
		io.Copy(w, resp.Body)
	}
}

func isStreamingResponse(body []byte) bool {
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		return false
	}
	return requestData["stream"] == true
}

func handleStreamingResponse(w http.ResponseWriter, body io.Reader) {
	scanner := bufio.NewScanner(body)
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) != "" {
			fmt.Fprintf(w, "%s\n\n", line)
			flusher.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading stream: %v", err)
	}
}
