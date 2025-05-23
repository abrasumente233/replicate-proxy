package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/r3labs/sse/v2"
	"github.com/sirupsen/logrus"
)

// ReplicateModel contains information about a Replicate model
type ReplicateModel struct {
	ModelData ModelData
}

// ModelMap maps OpenAI model IDs to Replicate model information
var ModelMap = map[string]ReplicateModel{
	"openai/gpt-4.1": {
		ModelData: ModelData{
			ID:          "openai/gpt-4.1",
			Name:        "GPT-4.1",
			Description: "OpenAI's Flagship GPT model for complex tasks.",
			Pricing: ModelPricing{
				Prompt:     "0.000002",
				Completion: "0.000008",
				Image:      "0",
				Request:    "0",
			},
			ContextLength: 128000,
			Architecture: ModelArchitecture{
				Modality:     "text+image->text",
				Tokenizer:    "GPT",
				InstructType: nil,
			},
		},
	},
	"openai/gpt-4o": {
		ModelData: ModelData{
			ID:          "openai/gpt-4o",
			Name:        "GPT-4o",
			Description: "OpenAI's high-intelligence chat model",
			Pricing: ModelPricing{
				Prompt:     "0.0000025",
				Completion: "0.00001",
				Image:      "0.003613",
				Request:    "0",
			},
			ContextLength: 128000,
			Architecture: ModelArchitecture{
				Modality:     "text+image->text",
				Tokenizer:    "GPT",
				InstructType: nil,
			},
		},
	},
	"anthropic/claude-sonnet-4": {
		ModelData: ModelData{
			ID:          "anthropic/claude-4-sonnet",
			Name:        "Claude Sonnet 4",
			Description: "Claude Sonnet 4 is a significant upgrade to 3.7, delivering superior coding and reasoning while responding more precisely to your instructions",
			Pricing: ModelPricing{
				Prompt:     "0.000003",
				Completion: "0.000015",
				Image:      "0.0048",
				Request:    "0",
			},
			ContextLength: 200000,
			Architecture: ModelArchitecture{
				Modality:     "text+image->text",
				Tokenizer:    "Claude",
				InstructType: nil,
			},
		},
	},
	"anthropic/claude-3.7-sonnet": {
		ModelData: ModelData{
			ID:          "anthropic/claude-3.7-sonnet",
			Name:        "Claude 3.7 Sonnet",
			Description: "Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. \n\nClaude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.\n\nRead more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet)",
			Pricing: ModelPricing{
				Prompt:     "0.000003",
				Completion: "0.000015",
				Image:      "0.0048",
				Request:    "0",
			},
			ContextLength: 200000,
			Architecture: ModelArchitecture{
				Modality:     "text+image->text",
				Tokenizer:    "Claude",
				InstructType: nil,
			},
		},
	},
	"anthropic/claude-3.5-sonnet": {
		ModelData: ModelData{
			ID:            "anthropic/claude-3.5-sonnet",
			Name:          "Claude 3.5 Sonnet",
			Description:   "New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal",
			ContextLength: 200000,
			Architecture: ModelArchitecture{
				Modality:     "text+image->text",
				Tokenizer:    "Claude",
				InstructType: nil,
			},
			Pricing: ModelPricing{
				Prompt:     "0.000003",
				Completion: "0.000015",
				Image:      "0.0048",
				Request:    "0",
			},
		},
	},
}

var (
	port     = flag.Int("port", 9876, "Port to run the proxy server on")
	logLevel = flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	log      = logrus.New()
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

// ModelData represents a single model in the response
type ModelData struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	Pricing       ModelPricing      `json:"pricing"`
	ContextLength int               `json:"context_length"`
	Architecture  ModelArchitecture `json:"architecture"`
}

// ModelPricing represents the pricing information for a model
type ModelPricing struct {
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
	Image      string `json:"image"`
	Request    string `json:"request"`
}

// ModelArchitecture represents the architecture information for a model
type ModelArchitecture struct {
	Modality     string      `json:"modality"`
	Tokenizer    string      `json:"tokenizer"`
	InstructType interface{} `json:"instruct_type"`
}

// ModelsResponse represents the response for the /v1/models endpoint
type ModelsResponse struct {
	Data []ModelData `json:"data"`
}

// CORSMiddleware adds CORS headers to allow cross-origin requests
func CORSMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Content-Length, X-Requested-With")

		// Handle preflight requests
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Call the next handler
		next(w, r)
	}
}

func main() {
	flag.Parse()
	proxyAddr := fmt.Sprintf(":%d", *port)

	// Set up logrus
	setLogLevel(*logLevel)

	// Configure logrus output format
	log.SetFormatter(&logrus.TextFormatter{
		TimestampFormat: "2006/01/02 15:04:05",
		FullTimestamp:   true,
	})

	// Register routes with CORS middleware
	http.HandleFunc("/v1/chat/completions", CORSMiddleware(proxyHandler))
	http.HandleFunc("/v1/models", CORSMiddleware(modelsHandler))

	// Server startup logs are always shown (Info level)
	log.WithFields(logrus.Fields{
		"port":    *port,
		"address": fmt.Sprintf("http://localhost%s", proxyAddr),
	}).Info("🚀 Replicate Proxy Server started")
	log.Info("📋 Endpoints available: /v1/chat/completions, /v1/models")
	log.Info("🌐 CORS enabled: All origins allowed")

	log.Fatal(http.ListenAndServe(proxyAddr, nil))
}

// Set log level based on flag
func setLogLevel(level string) {
	switch strings.ToLower(level) {
	case "debug":
		log.SetLevel(logrus.DebugLevel)
	case "info":
		log.SetLevel(logrus.InfoLevel)
	case "warn", "warning":
		log.SetLevel(logrus.WarnLevel)
	case "error":
		log.SetLevel(logrus.ErrorLevel)
	default:
		log.SetLevel(logrus.InfoLevel)
	}
	log.Infof("Log level set to: %s", log.GetLevel())
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())

	// Create logger fields for this request
	reqLogger := log.WithFields(logrus.Fields{
		"request_id": requestID,
		"client_ip":  r.RemoteAddr,
		"method":     r.Method,
		"path":       r.URL.Path,
	})

	// Check for Bearer token
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		reqLogger.Error("❌ Unauthorized: Bearer token required")
		http.Error(w, "Unauthorized: Bearer token required", http.StatusUnauthorized)
		return
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		reqLogger.WithError(err).Error("❌ Error reading request body")
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	// Parse the OpenAI-compatible request
	var openAIReq OpenAIRequest
	if err := json.Unmarshal(body, &openAIReq); err != nil {
		reqLogger.WithError(err).Error("❌ Error parsing request body")
		http.Error(w, "Error parsing request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Determine which model to use and get model information
	modelInfo, err := getModelInfo(openAIReq.Model)
	if err != nil {
		reqLogger.WithError(err).Error("❌ Error getting model information")
		http.Error(w, "Error getting model information: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Log request details (Info level)
	reqLogger.WithFields(logrus.Fields{
		"model":      openAIReq.Model,
		"stream":     openAIReq.Stream,
		"max_tokens": openAIReq.MaxTokens,
	}).Info("📥 Incoming request")

	// More detailed logs (Debug level)
	reqLogger.WithField("messages_count", len(openAIReq.Messages)).Debug("📨 Messages count")

	// Convert to Replicate request format
	replicateReq := convertToReplicateRequest(openAIReq)
	replicateReqBody, err := json.Marshal(replicateReq)
	if err != nil {
		reqLogger.WithError(err).Error("❌ Error creating Replicate request")
		http.Error(w, "Error creating Replicate request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Create a new request to the Replicate API
	replicateAPIURL := fmt.Sprintf("https://api.replicate.com/v1/models/%s/predictions", modelInfo.ModelData.ID)
	reqLogger.WithField("url", replicateAPIURL).Debug("🔄 Forwarding request to Replicate API")
	proxyReq, err := http.NewRequest("POST", replicateAPIURL, bytes.NewReader(replicateReqBody))
	if err != nil {
		reqLogger.WithError(err).Error("❌ Error creating proxy request")
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
		reqLogger.WithError(err).Error("❌ Error sending request to Replicate API")
		http.Error(w, "Error sending request to Replicate API: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// If response is not successful, forward the error
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		reqLogger.WithField("status_code", resp.StatusCode).Warn("⚠️ Replicate API error")
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	// Read the full response body first
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		reqLogger.WithError(err).Error("❌ Error reading Replicate response")
		http.Error(w, "Error reading Replicate response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Parse the raw response to get both the stream URL and prediction ID
	var rawResponse map[string]interface{}
	if err := json.Unmarshal(respBody, &rawResponse); err != nil {
		reqLogger.WithError(err).Error("❌ Error parsing Replicate response")
		http.Error(w, "Error parsing Replicate response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Extract prediction ID
	predictionID, ok := rawResponse["id"].(string)
	if !ok {
		reqLogger.Error("❌ No prediction ID found in response")
		http.Error(w, "No prediction ID found in response", http.StatusInternalServerError)
		return
	}

	reqLogger.WithField("prediction_id", predictionID).Debug("📝 Received prediction ID")

	// If streaming is requested
	if openAIReq.Stream {
		reqLogger.Debug("📺 Processing stream request")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Get the stream URL
		urls, ok := rawResponse["urls"].(map[string]interface{})
		if !ok {
			reqLogger.Error("❌ URLs field not found in response")
			http.Error(w, "URLs field not found in response", http.StatusInternalServerError)
			return
		}

		streamURL, ok := urls["stream"].(string)
		if !ok || streamURL == "" {
			reqLogger.Error("❌ No stream URL provided in Replicate response")
			http.Error(w, "No stream URL provided in Replicate response", http.StatusInternalServerError)
			return
		}

		// Stream the response from Replicate
		reqLogger.Debug("🚿 Starting to stream response from Replicate")
		handleReplicateStream(w, streamURL, token, reqLogger, modelInfo.ModelData.ID)
	} else {
		// For non-streaming, we need to poll until the prediction is complete
		reqLogger.Debug("🔄 Starting to poll for prediction results")
		pollAndReturnPrediction(w, predictionID, token, reqLogger, modelInfo.ModelData.ID)
	}

	reqLogger.WithField("duration", time.Since(startTime).String()).Info("✅ Completed request")
}

// getModelInfo returns the model information for the specified model ID,
// or returns an error if the specified model is not found
func getModelInfo(modelID string) (ReplicateModel, error) {
	if modelID == "" {
		return ReplicateModel{}, fmt.Errorf("no model specified")
	}

	if modelInfo, ok := ModelMap[modelID]; ok {
		return modelInfo, nil
	}

	return ReplicateModel{}, fmt.Errorf("model %s not found", modelID)
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
	lastMessageIsAssistant := false

	for i, msg := range messages {
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
				// If it's the last message, don't add newlines
				if i == len(messages)-1 {
					prompt.WriteString(fmt.Sprintf("Assistant %s: %s", msg.Name, content))
					lastMessageIsAssistant = true
				} else {
					prompt.WriteString(fmt.Sprintf("Assistant %s: %s\n\n", msg.Name, content))
				}
			} else {
				// If it's the last message, don't add newlines
				if i == len(messages)-1 {
					prompt.WriteString(fmt.Sprintf("Assistant: %s", content))
					lastMessageIsAssistant = true
				} else {
					prompt.WriteString(fmt.Sprintf("Assistant: %s\n\n", content))
				}
			}
		case "tool":
			content := getMessageContent(msg.Content)
			prompt.WriteString(fmt.Sprintf("Tool Response (%s): %s\n\n", msg.ToolCallID, content))
		}
	}

	// Add the final assistant prompt only if the last message is not from assistant
	if !lastMessageIsAssistant {
		prompt.WriteString("Assistant: ")
	}

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

func handleReplicateStream(w http.ResponseWriter, streamURL string, token string, logger *logrus.Entry, modelID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		logger.Error("❌ Streaming unsupported")
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// Set up event channel
	events := make(chan *sse.Event)

	// Create a new SSE client
	client := sse.NewClient(streamURL)

	// Set authorization header
	client.Headers = map[string]string{
		"Authorization": fmt.Sprintf("Bearer %s", token),
		"Accept":        "text/event-stream",
		"Cache-Control": "no-store",
	}

	// Use a simple counter for the message chunks
	chunkIndex := 0
	totalChunks := 0

	// Start subscription in a goroutine
	go func() {
		err := client.SubscribeChan("", events)
		if err != nil {
			logger.WithError(err).Error("❌ Error subscribing to SSE events")
		}
	}()

	// Process events as they come in
	for event := range events {
		// Handle different event types
		switch string(event.Event) {
		case "output":
			data := string(event.Data)
			// Skip " pending.*" or similar pending messages
			if !strings.Contains(data, "pending") {
				totalChunks++
				// Every 10 chunks, log progress (debug level)
				if totalChunks%10 == 0 {
					logger.WithField("chunks", totalChunks).Debug("🔄 Streaming progress")
				}

				// Format as OpenAI compatible streaming format
				chunk := map[string]interface{}{
					"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
					"object":  "chat.completion.chunk",
					"created": time.Now().Unix(),
					"model":   modelID,
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
				fmt.Fprintf(w, "data: %s\n\n", jsonChunk)
				flusher.Flush()
				chunkIndex++
			}

		case "done":
			logger.WithField("total_chunks", totalChunks).Debug("✅ Stream completed")
			// Send final chunk with finish_reason
			chunk := map[string]interface{}{
				"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   modelID,
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"delta":         map[string]interface{}{},
						"finish_reason": "stop",
					},
				},
			}

			jsonChunk, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", jsonChunk)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return

		default:
			// Log unhandled event types (debug level)
			logger.WithField("event_type", string(event.Event)).Debug("ℹ️ Unhandled event type")
		}
	}
}

func pollAndReturnPrediction(w http.ResponseWriter, predictionID string, token string, logger *logrus.Entry, modelID string) {
	// For non-streaming responses, we'd need to poll the prediction until it's complete
	client := &http.Client{}

	// Get the initial prediction to get the "get" URL
	initialPollURL := fmt.Sprintf("https://api.replicate.com/v1/predictions/%s", predictionID)

	pollReq, err := http.NewRequest("GET", initialPollURL, nil)
	if err != nil {
		logger.WithError(err).Error("❌ Error creating poll request")
		http.Error(w, "Error creating poll request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	pollReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	pollResp, err := client.Do(pollReq)
	if err != nil {
		logger.WithError(err).Error("❌ Error polling prediction")
		http.Error(w, "Error polling prediction: "+err.Error(), http.StatusInternalServerError)
		return
	}

	respBody, _ := io.ReadAll(pollResp.Body)
	pollResp.Body.Close()

	var initialPollResult map[string]interface{}
	if err := json.Unmarshal(respBody, &initialPollResult); err != nil {
		logger.WithError(err).Error("❌ Error parsing poll response")
		http.Error(w, "Error parsing poll response: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Extract the "get" URL from the response
	urls, ok := initialPollResult["urls"].(map[string]interface{})
	if !ok {
		logger.Error("❌ Error extracting URLs from prediction response")
		http.Error(w, "Error extracting URLs from prediction response", http.StatusInternalServerError)
		return
	}

	getURL, ok := urls["get"].(string)
	if !ok || getURL == "" {
		// Fall back to constructed URL if "get" URL is not available
		logger.Debug("⚠️ No 'get' URL found, falling back to constructed URL")
		getURL = initialPollURL
	}

	pollCount := 0
	for {
		pollCount++
		logger.WithField("attempt", pollCount).Debug("🔄 Polling prediction")
		time.Sleep(1 * time.Second)

		pollReq, err := http.NewRequest("GET", getURL, nil)
		if err != nil {
			logger.WithError(err).Error("❌ Error creating poll request")
			http.Error(w, "Error creating poll request: "+err.Error(), http.StatusInternalServerError)
			return
		}

		pollReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

		pollResp, err := client.Do(pollReq)
		if err != nil {
			logger.WithError(err).Error("❌ Error polling prediction")
			http.Error(w, "Error polling prediction: "+err.Error(), http.StatusInternalServerError)
			return
		}

		respBody, _ := io.ReadAll(pollResp.Body)
		pollResp.Body.Close()

		var pollResult map[string]interface{}
		if err := json.Unmarshal(respBody, &pollResult); err != nil {
			logger.WithError(err).Error("❌ Error parsing poll response")
			http.Error(w, "Error parsing poll response: "+err.Error(), http.StatusInternalServerError)
			return
		}

		status, _ := pollResult["status"].(string)
		logger.WithField("status", status).Debug("📊 Prediction status")

		if status == "succeeded" {
			logger.Debug("✅ Prediction completed successfully")
			// Extract the output, which could be a string or an array of strings
			var output string

			outputVal := pollResult["output"]

			switch val := outputVal.(type) {
			case string:
				// Direct string output
				output = val
			case []interface{}:
				// Array of string chunks that need to be concatenated
				var builder strings.Builder
				for _, chunk := range val {
					if strChunk, ok := chunk.(string); ok {
						builder.WriteString(strChunk)
					}
				}
				output = builder.String()
			default:
				logger.Error("❌ Unexpected output format in prediction response")
				http.Error(w, "Unexpected output format in prediction response", http.StatusInternalServerError)
				return
			}

			response := map[string]interface{}{
				"id":      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
				"object":  "chat.completion",
				"created": time.Now().Unix(),
				"model":   modelID,
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

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			logger.Debug("📤 Response sent to client")
			return
		} else if status == "failed" || status == "canceled" {
			error, _ := pollResult["error"].(string)
			logger.WithField("error", error).Error("❌ Prediction failed")
			http.Error(w, fmt.Sprintf("Prediction failed: %s", error), http.StatusInternalServerError)
			return
		}

		// Continue polling for other statuses like "starting", "processing"
	}
}

// modelsHandler handles the /v1/models endpoint
func modelsHandler(w http.ResponseWriter, r *http.Request) {
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())
	logger := log.WithFields(logrus.Fields{
		"request_id": requestID,
		"endpoint":   "/v1/models",
		"method":     r.Method,
	})

	logger.Debug("Received request for models")

	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		logger.Warnf("Method not allowed: %s", r.Method)
		json.NewEncoder(w).Encode(map[string]string{"error": "Method not allowed"})
		return
	}

	// Extract model data from ModelMap
	modelData := make([]ModelData, 0, len(ModelMap))
	for _, model := range ModelMap {
		modelData = append(modelData, model.ModelData)
	}

	// Model information
	response := ModelsResponse{
		Data: modelData,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		logger.Errorf("Error encoding response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "Internal server error"})
		return
	}

	logger.Infof("Returned list of models")
}
