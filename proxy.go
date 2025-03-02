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
	"time"
)

const (
	replicateAPIURL = "https://api.replicate.com/v1/models/anthropic/claude-3.7-sonnet/predictions"
)

var (
	port = flag.Int("port", 9876, "Port to run the proxy server on")
)

// OpenAI-compatible request structure
type OpenAIRequest struct {
	Messages          []Message              `json:"messages,omitempty"`
	Prompt            string                 `json:"prompt,omitempty"`
	Model             string                 `json:"model,omitempty"`
	ResponseFormat    map[string]string      `json:"response_format,omitempty"`
	Stop              interface{}            `json:"stop,omitempty"`
	Stream            bool                   `json:"stream,omitempty"`
	MaxTokens         int                    `json:"max_tokens,omitempty"`
	Temperature       float64                `json:"temperature,omitempty"`
	Tools             []interface{}          `json:"tools,omitempty"`
	ToolChoice        interface{}            `json:"tool_choice,omitempty"`
	Seed              int                    `json:"seed,omitempty"`
	TopP              float64                `json:"top_p,omitempty"`
	TopK              int                    `json:"top_k,omitempty"`
	FrequencyPenalty  float64                `json:"frequency_penalty,omitempty"`
	PresencePenalty   float64                `json:"presence_penalty,omitempty"`
	RepetitionPenalty float64                `json:"repetition_penalty,omitempty"`
	LogitBias         map[int]float64        `json:"logit_bias,omitempty"`
	TopLogprobs       int                    `json:"top_logprobs,omitempty"`
	MinP              float64                `json:"min_p,omitempty"`
	TopA              float64                `json:"top_a,omitempty"`
	Prediction        map[string]string      `json:"prediction,omitempty"`
	Transforms        []string               `json:"transforms,omitempty"`
	Models            []string               `json:"models,omitempty"`
	Route             string                 `json:"route,omitempty"`
	Provider          map[string]interface{} `json:"provider,omitempty"`
}

// Message structure for OpenAI format
type Message struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"`
	Name       string      `json:"name,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// Replicate API request structure
type ReplicateRequest struct {
	Stream bool                   `json:"stream"`
	Input  map[string]interface{} `json:"input"`
}

// Replicate API response structure for prediction creation
type ReplicatePredictionResponse struct {
	URLs struct {
		Stream string `json:"stream"`
		Get    string `json:"get"`
	} `json:"urls"`
	ID     string `json:"id"`
	Status string `json:"status"`
}

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

	// Parse the OpenAI-compatible request
	var openAIReq OpenAIRequest
	if err := json.Unmarshal(body, &openAIReq); err != nil {
		http.Error(w, "Error parsing request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Convert to Replicate request format
	replicateReq := convertToReplicateRequest(openAIReq)
	replicateReqBody, err := json.Marshal(replicateReq)
	if err != nil {
		http.Error(w, "Error creating Replicate request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Create a new request to the Replicate API
	proxyReq, err := http.NewRequest("POST", replicateAPIURL, bytes.NewReader(replicateReqBody))
	if err != nil {
		http.Error(w, "Error creating proxy request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Set headers for the proxy request
	proxyReq.Header.Set("Content-Type", "application/json")
	proxyReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	// Send the request to Replicate API
	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, "Error sending request to Replicate API: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// If response is not successful, forward the error
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	// Read the full response body first
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Error reading Replicate response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Parse the raw response to get both the stream URL and prediction ID
	var rawResponse map[string]interface{}
	if err := json.Unmarshal(respBody, &rawResponse); err != nil {
		http.Error(w, "Error parsing Replicate response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Extract prediction ID
	predictionID, ok := rawResponse["id"].(string)
	if !ok {
		http.Error(w, "No prediction ID found in response", http.StatusInternalServerError)
		return
	}

	// Log the complete response for debugging
	log.Printf("Replicate full response: %s", string(respBody))

	// If streaming is requested
	if openAIReq.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Get the stream URL
		urls, ok := rawResponse["urls"].(map[string]interface{})
		if !ok {
			http.Error(w, "URLs field not found in response", http.StatusInternalServerError)
			return
		}

		streamURL, ok := urls["stream"].(string)
		if !ok || streamURL == "" {
			http.Error(w, "No stream URL provided in Replicate response", http.StatusInternalServerError)
			return
		}

		log.Printf("Connecting to stream URL: %s", streamURL)

		// Stream the response from Replicate
		streamReq, err := http.NewRequest("GET", streamURL, nil)
		if err != nil {
			http.Error(w, "Error creating stream request: "+err.Error(), http.StatusInternalServerError)
			return
		}

		streamReq.Header.Set("Accept", "text/event-stream")
		streamReq.Header.Set("Cache-Control", "no-store")
		streamReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

		streamResp, err := client.Do(streamReq)
		if err != nil {
			http.Error(w, "Error connecting to Replicate stream: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer streamResp.Body.Close()

		// Check if the stream response is successful
		if streamResp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(streamResp.Body)
			log.Printf("Stream response error: %d - %s", streamResp.StatusCode, string(respBody))
			http.Error(w, fmt.Sprintf("Stream error: %d", streamResp.StatusCode), streamResp.StatusCode)
			return
		}

		log.Printf("Stream connection established, starting to handle stream")

		// Stream the response back to the client in OpenAI compatible format
		handleReplicateStream(w, streamResp.Body)
	} else {
		// For non-streaming, we need to poll until the prediction is complete
		pollAndReturnPrediction(w, predictionID, token)
	}
}

func convertToReplicateRequest(req OpenAIRequest) ReplicateRequest {
	input := make(map[string]interface{})

	// Handle either messages or prompt
	if len(req.Messages) > 0 {
		// Convert messages to a prompt string for Claude
		prompt := formatMessagesAsPrompt(req.Messages)
		input["prompt"] = prompt
	} else if req.Prompt != "" {
		input["prompt"] = req.Prompt
	}

	// Add additional parameters that Claude supports
	if req.MaxTokens > 0 {
		input["max_tokens"] = req.MaxTokens
	}

	if req.Temperature > 0 {
		input["temperature"] = req.Temperature
	}

	// Handle stop tokens if provided
	if req.Stop != nil {
		input["stop_sequences"] = req.Stop
	}

	return ReplicateRequest{
		Stream: req.Stream,
		Input:  input,
	}
}

func formatMessagesAsPrompt(messages []Message) string {
	var prompt strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			content := getMessageContent(msg.Content)
			prompt.WriteString(fmt.Sprintf("System: %s\n\n", content))
		case "user":
			content := getMessageContent(msg.Content)
			if msg.Name != "" {
				prompt.WriteString(fmt.Sprintf("User %s: %s\n\n", msg.Name, content))
			} else {
				prompt.WriteString(fmt.Sprintf("Human: %s\n\n", content))
			}
		case "assistant":
			content := getMessageContent(msg.Content)
			if msg.Name != "" {
				prompt.WriteString(fmt.Sprintf("Assistant %s: %s\n\n", msg.Name, content))
			} else {
				prompt.WriteString(fmt.Sprintf("Assistant: %s\n\n", content))
			}
		case "tool":
			content := getMessageContent(msg.Content)
			prompt.WriteString(fmt.Sprintf("Tool Response (%s): %s\n\n", msg.ToolCallID, content))
		}
	}

	// Add the final assistant prompt
	prompt.WriteString("Assistant: ")

	return prompt.String()
}

func getMessageContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		// Handle content parts (text or image_url)
		var result strings.Builder
		for _, part := range v {
			if contentMap, ok := part.(map[string]interface{}); ok {
				if contentType, ok := contentMap["type"].(string); ok {
					if contentType == "text" {
						if text, ok := contentMap["text"].(string); ok {
							result.WriteString(text)
						}
					} else if contentType == "image_url" {
						result.WriteString("[Image attached]")
					}
				}
			}
		}
		return result.String()
	default:
		jsonContent, _ := json.Marshal(content)
		return string(jsonContent)
	}
}

func handleReplicateStream(w http.ResponseWriter, body io.Reader) {
	scanner := bufio.NewScanner(body)
	// Increase the buffer size to handle larger chunks
	const maxScanTokenSize = 1024 * 1024 // 1MB
	buf := make([]byte, maxScanTokenSize)
	scanner.Buffer(buf, maxScanTokenSize)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Use a simple counter for the message chunks
	chunkIndex := 0

	// Debug: Log that we started streaming
	log.Printf("Starting to handle Replicate stream")

	var currentEvent string
	var currentData string
	var streamDone bool = false

	for scanner.Scan() {
		line := scanner.Text()
		log.Printf("Received line from stream: %s", line)

		// Skip empty lines
		if strings.TrimSpace(line) == "" {
			continue
		}

		// Parse the SSE event
		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			log.Printf("Found event type: %s", currentEvent)
		} else if strings.HasPrefix(line, "data:") {
			currentData = strings.TrimPrefix(line, "data:")
			// Trim only one leading space if it exists
			if len(currentData) > 0 && currentData[0] == ' ' {
				currentData = currentData[1:]
			}
			log.Printf("Found data: %s", currentData)

			// Process the event and data
			if currentEvent != "" && currentData != "" {
				if currentEvent == "done" {
					processEvent(w, flusher, currentEvent, currentData, &chunkIndex)
					streamDone = true
					break // Exit the loop after processing "done" event
				} else {
					processEvent(w, flusher, currentEvent, currentData, &chunkIndex)
				}

				// Reset for next event
				currentEvent = ""
				currentData = ""
			}
		} else {
			log.Printf("Unrecognized line format: %s", line)
		}
	}

	if !streamDone && scanner.Err() != nil {
		log.Printf("Error reading stream: %v", scanner.Err())
	} else if !streamDone {
		log.Printf("Scanner completed without receiving 'done' event")
	}

	log.Printf("Exiting handleReplicateStream function")
}

// Helper function to process SSE events
func processEvent(w http.ResponseWriter, flusher http.Flusher, eventType string, data string, chunkIndex *int) {
	if eventType == "output" {
		// Skip " pending.*" or similar pending messages
		if !strings.Contains(data, "pending") {
			// Format as OpenAI compatible streaming format
			chunk := map[string]interface{}{
				"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   "claude-3.7-sonnet",
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": map[string]interface{}{
							"content": data,
						},
						"finish_reason": nil,
					},
				},
			}

			jsonChunk, _ := json.Marshal(chunk)
			log.Printf("Sending chunk to client: %s", jsonChunk)
			fmt.Fprintf(w, "data: %s\n\n", jsonChunk)
			flusher.Flush()
			*chunkIndex++
		} else {
			log.Printf("Skipping pending message: %s", data)
		}
	} else if eventType == "done" {
		// Send final chunk with finish_reason
		chunk := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   "claude-3.7-sonnet",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": "stop",
				},
			},
		}

		jsonChunk, _ := json.Marshal(chunk)
		log.Printf("Sending final chunk to client: %s", jsonChunk)
		fmt.Fprintf(w, "data: %s\n\n", jsonChunk)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		log.Printf("Stream completed")
	} else {
		log.Printf("Unhandled event type: %s", eventType)
	}
}

func pollAndReturnPrediction(w http.ResponseWriter, predictionID string, token string) {
	// For non-streaming responses, we'd need to poll the prediction until it's complete
	// This is a simplified version - in a real implementation, you'd want to add timeouts and error handling
	client := &http.Client{}

	log.Printf("[NON-STREAMING] Starting polling for prediction ID: %s", predictionID)

	// Get the initial prediction to get the "get" URL
	initialPollURL := fmt.Sprintf("https://api.replicate.com/v1/predictions/%s", predictionID)
	log.Printf("[NON-STREAMING] Initial poll URL: %s", initialPollURL)

	pollReq, err := http.NewRequest("GET", initialPollURL, nil)
	if err != nil {
		log.Printf("[NON-STREAMING] Error creating initial poll request: %v", err)
		http.Error(w, "Error creating poll request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	pollReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	pollResp, err := client.Do(pollReq)
	if err != nil {
		log.Printf("[NON-STREAMING] Error making initial poll request: %v", err)
		http.Error(w, "Error polling prediction: "+err.Error(), http.StatusInternalServerError)
		return
	}

	respBody, _ := io.ReadAll(pollResp.Body)
	pollResp.Body.Close()
	log.Printf("[NON-STREAMING] Initial poll response: %s", string(respBody))

	var initialPollResult map[string]interface{}
	if err := json.Unmarshal(respBody, &initialPollResult); err != nil {
		log.Printf("[NON-STREAMING] Error parsing initial poll response: %v", err)
		http.Error(w, "Error parsing poll response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Extract the "get" URL from the response
	urls, ok := initialPollResult["urls"].(map[string]interface{})
	if !ok {
		log.Printf("[NON-STREAMING] Error: 'urls' field not found or not a map in initial poll response")
		http.Error(w, "Error extracting URLs from prediction response", http.StatusInternalServerError)
		return
	}

	getURL, ok := urls["get"].(string)
	if !ok || getURL == "" {
		log.Printf("[NON-STREAMING] Warning: 'get' URL not found in initial poll response, falling back to constructed URL")
		// Fall back to constructed URL if "get" URL is not available
		getURL = initialPollURL
	}

	log.Printf("[NON-STREAMING] Using get URL for polling: %s", getURL)

	pollCount := 0
	for {
		pollCount++
		time.Sleep(1 * time.Second)
		log.Printf("[NON-STREAMING] Poll attempt #%d for prediction %s", pollCount, predictionID)

		pollReq, err := http.NewRequest("GET", getURL, nil)
		if err != nil {
			log.Printf("[NON-STREAMING] Error creating poll request: %v", err)
			http.Error(w, "Error creating poll request: "+err.Error(), http.StatusInternalServerError)
			return
		}

		pollReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

		pollResp, err := client.Do(pollReq)
		if err != nil {
			log.Printf("[NON-STREAMING] Error making poll request: %v", err)
			http.Error(w, "Error polling prediction: "+err.Error(), http.StatusInternalServerError)
			return
		}

		respBody, _ := io.ReadAll(pollResp.Body)
		pollResp.Body.Close()
		log.Printf("[NON-STREAMING] Poll response (attempt #%d): %s", pollCount, string(respBody))

		var pollResult map[string]interface{}
		if err := json.Unmarshal(respBody, &pollResult); err != nil {
			log.Printf("[NON-STREAMING] Error parsing poll response: %v", err)
			http.Error(w, "Error parsing poll response: "+err.Error(), http.StatusInternalServerError)
			return
		}

		status, _ := pollResult["status"].(string)
		log.Printf("[NON-STREAMING] Poll status (attempt #%d): %s", pollCount, status)

		if status == "succeeded" {
			// Extract the output, which could be a string or an array of strings
			var output string

			outputVal := pollResult["output"]
			log.Printf("[NON-STREAMING] Output type: %T", outputVal)

			switch val := outputVal.(type) {
			case string:
				// Direct string output
				output = val
				log.Printf("[NON-STREAMING] Output is a string, length: %d", len(output))
			case []interface{}:
				// Array of string chunks that need to be concatenated
				log.Printf("[NON-STREAMING] Output is an array with %d elements", len(val))
				var builder strings.Builder
				for _, chunk := range val {
					if strChunk, ok := chunk.(string); ok {
						builder.WriteString(strChunk)
					}
				}
				output = builder.String()
				log.Printf("[NON-STREAMING] Concatenated output string, length: %d", len(output))
			default:
				log.Printf("[NON-STREAMING] Unexpected output type: %T", outputVal)
				outputJSON, _ := json.Marshal(outputVal)
				log.Printf("[NON-STREAMING] Raw output value: %s", string(outputJSON))
				http.Error(w, "Unexpected output format in prediction response", http.StatusInternalServerError)
				return
			}

			// Check if we got any output content
			if len(output) == 0 {
				log.Printf("[NON-STREAMING] Warning: Empty output content")
			}

			log.Printf("[NON-STREAMING] Prediction succeeded! Output length: %d", len(output))
			if len(output) > 100 {
				log.Printf("[NON-STREAMING] Output preview (first 100 chars): %s...", output[:100])
			} else {
				log.Printf("[NON-STREAMING] Output: %s", output)
			}

			response := map[string]interface{}{
				"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
				"object":  "chat.completion",
				"created": time.Now().Unix(),
				"model":   "claude-3.7-sonnet",
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": output,
						},
						"finish_reason": "stop",
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     0, // We don't have this information
					"completion_tokens": 0, // We don't have this information
					"total_tokens":      0, // We don't have this information
				},
			}

			responseJSON, _ := json.Marshal(response)
			log.Printf("[NON-STREAMING] Sending response to client: %s", string(responseJSON))

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			log.Printf("[NON-STREAMING] Response sent successfully")
			return
		} else if status == "failed" || status == "canceled" {
			error, _ := pollResult["error"].(string)
			log.Printf("[NON-STREAMING] Prediction failed: %s", error)
			http.Error(w, fmt.Sprintf("Prediction failed: %s", error), http.StatusInternalServerError)
			return
		}

		// Continue polling for other statuses like "starting", "processing"
		log.Printf("[NON-STREAMING] Waiting for prediction to complete, current status: %s", status)
	}
}
