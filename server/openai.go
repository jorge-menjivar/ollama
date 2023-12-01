package server

import (
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jmorganca/ollama/api"
)

type OpenAIError struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   interface{} `json:"param"`
	Code    *string     `json:"code"`
}

type OpenAIErrorResponse struct {
	Error OpenAIError `json:"error"`
}

type OpenAIChatCompletionRequest struct {
	Model    string
	Messages []OpenAIMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (m *OpenAIMessage) toMessage() api.Message {
	return api.Message{
		Role:    m.Role,
		Content: m.Content,
	}
}

// non-streaming response

type OpenAIChatCompletionResponseChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason *string       `json:"finish_reason"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIChatCompletionResponse struct {
	ID                string                               `json:"id"`
	Object            string                               `json:"object"`
	Created           int64                                `json:"created"`
	Model             string                               `json:"model"`
	SystemFingerprint string                               `json:"system_fingerprint"`
	Choices           []OpenAIChatCompletionResponseChoice `json:"choices"`
	Usage             OpenAIUsage                          `json:"usage,omitempty"`
}

// streaming response

type OpenAIChatCompletionResponseChoiceStream struct {
	Index        int           `json:"index"`
	Delta        OpenAIMessage `json:"delta"`
	FinishReason *string       `json:"finish_reason"`
}

type OpenAIChatCompletionResponseStream struct {
	ID                string                                     `json:"id"`
	Object            string                                     `json:"object"`
	Created           int64                                      `json:"created"`
	Model             string                                     `json:"model"`
	SystemFingerprint string                                     `json:"system_fingerprint"`
	Choices           []OpenAIChatCompletionResponseChoiceStream `json:"choices"`
}

func ChatCompletions(c *gin.Context) {
	var req OpenAIChatCompletionRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, OpenAIErrorResponse{
			OpenAIError{
				Message: "missing request body",
				Type:    "invalid_request_error",
			},
		})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, OpenAIErrorResponse{
			OpenAIError{
				Message: err.Error(),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	// Call generate and receive the channel with the responses
	chatReq := api.ChatRequest{
		Model:  req.Model,
		Stream: &req.Stream,
	}
	for _, m := range req.Messages {
		chatReq.Messages = append(chatReq.Messages, m.toMessage())
	}
	ch, generated := chat(c, chatReq, time.Now())

	if !req.Stream {
		// Wait for the channel to close
		var chatResponse api.ChatResponse
		for val := range ch {
			var ok bool
			chatResponse, ok = val.(api.ChatResponse)
			if !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, OpenAIErrorResponse{
					OpenAIError{
						Message: err.Error(),
						Type:    "internal_server_error",
					},
				})
				return
			}
			if chatResponse.Done {
				chatResponse.Message = &api.Message{Role: "assistant", Content: generated.String()}
			}
		}
		// Send a single response with accumulated content
		id := fmt.Sprintf("chatcmpl-%d", rand.Intn(999))
		chatCompletionResponse := OpenAIChatCompletionResponse{
			ID:      id,
			Object:  "chat.completion",
			Created: chatResponse.CreatedAt.Unix(),
			Model:   req.Model,
			Choices: []OpenAIChatCompletionResponseChoice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: chatResponse.Message.Content,
					},
					FinishReason: func(done bool) *string {
						if done {
							reason := "stop"
							return &reason
						}
						return nil
					}(chatResponse.Done),
				},
			},
		}
		c.JSON(http.StatusOK, chatCompletionResponse)
		return
	}

	// Now, create the intermediate channel and transformation goroutine
	transformedCh := make(chan any)

	go func() {
		defer close(transformedCh)
		id := fmt.Sprintf("chatcmpl-%d", rand.Intn(999)) // TODO: validate that this does not change with each chunk
		predefinedResponse := OpenAIChatCompletionResponseStream{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []OpenAIChatCompletionResponseChoiceStream{
				{
					Index: 0,
					Delta: OpenAIMessage{
						Role: "assistant",
					},
				},
			},
		}
		transformedCh <- predefinedResponse
		for val := range ch {
			resp, ok := val.(api.ChatResponse)
			if !ok {
				// If val is not of type ChatResponse, send an error down the channel and exit
				transformedCh <- OpenAIErrorResponse{
					OpenAIError{
						Message: err.Error(),
						Type:    "internal_server_error",
					},
				}
				return
			}

			// TODO: handle errors

			// Transform the ChatResponse into OpenAIChatCompletionResponse
			chatCompletionResponse := OpenAIChatCompletionResponseStream{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: resp.CreatedAt.Unix(),
				Model:   resp.Model,
				Choices: []OpenAIChatCompletionResponseChoiceStream{
					{
						Index: 0,
						Delta: OpenAIMessage{
							Content: resp.Message.Content,
						},
						FinishReason: func(done bool) *string {
							if done {
								reason := "stop"
								return &reason
							}
							return nil
						}(resp.Done),
					},
				},
			}
			transformedCh <- chatCompletionResponse
		}
	}()

	// Pass the transformed channel to streamResponse
	streamResponse(c, transformedCh)
}
