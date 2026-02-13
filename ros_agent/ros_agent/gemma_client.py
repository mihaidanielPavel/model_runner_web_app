#!/usr/bin/env python3
"""
Gemma3 API Client for ROS2 Agent

This module provides an interface to communicate with Gemma3 running in Docker
via Ollama-compatible API.
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class ChatMessage:
    """Represents a chat message for the agent."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class GemmaClient:
    """Client for communicating with Gemma3 via Ollama API."""
    
    def __init__(self, api_url: str = "http://model-runner.docker.internal", 
                 model_name: str = "ai/gemma3", 
                 timeout: int = 30,
                 max_retries: int = 3,
                 logger=None):
        """
        Initialize the Gemma3 client.
        
        Args:
            api_url: Base URL for the Ollama API
            model_name: Name of the Gemma model to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            logger: ROS2 logger instance (from self.get_logger())
        """
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger
        self.conversation_history: List[ChatMessage] = []
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to the Docker Model Runner API."""
        try:
            # Docker Model Runner uses OpenAI-compatible endpoints
            response = requests.get(f"{self.api_url}/engines/llama.cpp/v1/models", timeout=5)
            if response.status_code == 200:
                # models_data = response.json()
                # available_models = []
                
                # Extract model IDs from the response
                # if 'data' in models_data:
                #     available_models = [model['id'] for model in models_data['data']]
                
                # if self.model_name not in available_models:
                #     logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                #     # Try to use the first available gemma model
                #     gemma_models = [m for m in available_models if 'gemma' in m.lower()]
                #     if gemma_models:
                #         self.model_name = gemma_models[0]
                #         logger.info(f"Using model: {self.model_name}")
                
                if self.logger:
                    self.logger.info(f"Successfully connected to Docker Model Runner API at {self.api_url}")
                return True
            else:
                if self.logger:
                    self.logger.error(f"Failed to connect to API: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"Connection test failed: {e}")
            return False
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a request to the Docker Model Runner API with retry logic."""
        # Docker Model Runner uses OpenAI-compatible endpoints
        url = f"{self.api_url}/engines/llama.cpp/v1/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    json=data, 
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    if self.logger:
                        self.logger.warning(f"API request failed (attempt {attempt + 1}): HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                if self.logger:
                    self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        if self.logger:
            self.logger.error(f"All {self.max_retries} attempts failed for endpoint: {endpoint}")
        return None
    
    def generate_response(self, prompt: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Generate a response from Gemma3 with metadata extraction.
        
        Args:
            prompt: User's question or prompt
            context: Additional context (e.g., sensor data)
            
        Returns:
            Dictionary with 'response', 'usage', 'timings', and 'metrics' or None if failed
        """
        # Prepare the full prompt with context
        full_prompt = self._prepare_prompt(prompt, context)
        
        # Estimate token usage (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(full_prompt) // 4
        if self.logger and estimated_tokens > 3000:  # Warn if approaching limit
            self.logger.warning(f"High token usage estimated: {estimated_tokens} tokens")
        
        # Add user message to conversation history
        self.conversation_history.append(ChatMessage(role="user", content=full_prompt))
        
        # Prepare API request
        request_data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in self.conversation_history[-5:]  # Keep only last 5 messages to save tokens
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500  # Limit response tokens to save budget
            }
        }
        
        # Make the request
        response_data = self._make_request("chat/completions", request_data)
        
        if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
            assistant_response = response_data['choices'][0]['message']['content']
            
            # Add assistant response to conversation history
            self.conversation_history.append(
                ChatMessage(role="assistant", content=assistant_response)
            )
            
            # Extract metadata
            usage_data = response_data.get('usage', {})
            timings_data = response_data.get('timings', {})
            
            # Calculate derived metrics
            metrics = self._calculate_metrics(usage_data, timings_data)
            
            return {
                'response': assistant_response,
                'usage': usage_data,
                'timings': timings_data,
                'metrics': metrics
            }
        
        return None
    
    def _calculate_metrics(self, usage_data: Dict[str, Any], timings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived metrics from usage and timing data."""
        metrics = {}
        
        # Token metrics
        completion_tokens = usage_data.get('completion_tokens', 0)
        prompt_tokens = usage_data.get('prompt_tokens', 0)
        total_tokens = usage_data.get('total_tokens', 0)
        
        # Timing metrics
        prompt_ms = timings_data.get('prompt_ms', 0)
        predicted_ms = timings_data.get('predicted_ms', 0)
        
        # Calculate derived metrics
        if prompt_ms > 0:
            metrics['prompt_tokens_per_second'] = (prompt_tokens * 1000) / prompt_ms
        else:
            metrics['prompt_tokens_per_second'] = 0
            
        if predicted_ms > 0:
            metrics['completion_tokens_per_second'] = (completion_tokens * 1000) / predicted_ms
        else:
            metrics['completion_tokens_per_second'] = 0
        
        # Average tokens per second
        total_ms = prompt_ms + predicted_ms
        if total_ms > 0:
            metrics['overall_tokens_per_second'] = (total_tokens * 1000) / total_ms
        else:
            metrics['overall_tokens_per_second'] = 0
        
        # Efficiency score (higher is better)
        if completion_tokens > 0 and predicted_ms > 0:
            metrics['efficiency_score'] = completion_tokens / (predicted_ms / 1000)  # tokens per second
        else:
            metrics['efficiency_score'] = 0
        
        # Cost estimate (rough approximation: $0.0005 per 1K tokens)
        metrics['estimated_cost'] = (total_tokens / 1000) * 0.0005
        
        return metrics
    
    def _prepare_prompt(self, prompt: str, context: str) -> str:
        """Prepare the full prompt with system context and sensor data."""
        system_prompt = """You are a robot AI assistant. You can call functions to interact with the robot.

IMPORTANT FUNCTION USAGE:
- For "where is robot" or "position" questions: use get_navigation_status()
- For sensor data: use get_latest_topic_data(/topic_name) with actual topic names like /amcl_pose, /scan
- For movement: use move_to_position(x,y,yaw) OR move_to_position(label) for named locations
- For stopping: use stop_navigation()
- For location management: use add_location_label(label,x,y,yaw), list_location_labels(), remove_location_label(label)

Functions: get_navigation_status(), get_latest_topic_data(topic), move_to_position(x,y,yaw), stop_navigation(), call_service(name,data), add_location_label(label,x,y,yaw), list_location_labels(), remove_location_label(label)

When asked to perform actions, call the appropriate functions. Be helpful and prioritize safety.

Context:"""
        
        if context:
            return f"{system_prompt}\n{context}\n\nUser: {prompt}"
        else:
            return f"{system_prompt}\nNo context available.\n\nUser: {prompt}"
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        if self.logger:
            self.logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        if not self.conversation_history:
            return "No conversation history"
        
        summary = f"Conversation with {len(self.conversation_history)} messages:\n"
        for i, msg in enumerate(self.conversation_history[-5:], 1):  # Last 5 messages
            role_emoji = "👤" if msg.role == "user" else "🤖"
            summary += f"{i}. {role_emoji} {msg.role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\n"
        
        return summary
    
    def health_check(self) -> bool:
        """Check if the Docker Model Runner API is healthy and responsive."""
        try:
            response = requests.get(f"{self.api_url}/engines/llama.cpp/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
