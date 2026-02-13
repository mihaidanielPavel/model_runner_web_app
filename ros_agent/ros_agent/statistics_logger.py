#!/usr/bin/env python3
"""
Statistics Logger for ROS2 Agent

Logs interaction statistics to CSV files for research purposes.
"""

import csv
import time
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class InteractionStats:
    """Statistics for a single interaction."""
    timestamp: float
    question: str
    response: str
    response_time: float  # Time to get response from agent
    processing_time: float  # Time for agent to process
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    prompt_ms: Optional[float] = None
    predicted_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    efficiency_score: Optional[float] = None
    estimated_cost: Optional[float] = None
    navigation_command: bool = False
    service_call: bool = False
    topic_data_request: bool = False
    success: bool = True
    error_message: Optional[str] = None

class StatisticsLogger:
    """Logs interaction statistics to CSV files."""
    
    def __init__(self, log_dir: str = "logs", filename_prefix: str = "interaction_stats"):
        """
        Initialize the statistics logger.
        
        Args:
            log_dir: Directory to store log files
            filename_prefix: Prefix for log filenames
        """
        self.log_dir = Path(log_dir)
        self.filename_prefix = filename_prefix
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"{filename_prefix}_{timestamp}.csv"
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
        self.interaction_count = 0
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'question', 'response', 'response_time', 
                    'processing_time', 'tokens_input', 'tokens_output',
                    'prompt_ms', 'predicted_ms', 'tokens_per_second',
                    'efficiency_score', 'estimated_cost',
                    'navigation_command', 'service_call', 'topic_data_request',
                    'success', 'error_message'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        except Exception as e:
            print(f"Failed to initialize CSV file: {e}")
    
    def log_interaction(self, stats: InteractionStats):
        """
        Log interaction statistics to CSV.
        
        Args:
            stats: Interaction statistics to log
        """
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'question', 'response', 'response_time', 
                    'processing_time', 'tokens_input', 'tokens_output',
                    'prompt_ms', 'predicted_ms', 'tokens_per_second',
                    'efficiency_score', 'estimated_cost',
                    'navigation_command', 'service_call', 'topic_data_request',
                    'success', 'error_message'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Convert stats to dict and write
                stats_dict = asdict(stats)
                writer.writerow(stats_dict)
                
            self.interaction_count += 1
            
        except Exception as e:
            print(f"Failed to log interaction statistics: {e}")
    
    def log_agent_interaction(self, question: str, response: str, 
                            response_time: float, processing_time: float,
                            navigation_command: bool = False,
                            service_call: bool = False,
                            topic_data_request: bool = False,
                            success: bool = True,
                            error_message: Optional[str] = None,
                            tokens_input: Optional[int] = None,
                            tokens_output: Optional[int] = None,
                            prompt_ms: Optional[float] = None,
                            predicted_ms: Optional[float] = None,
                            tokens_per_second: Optional[float] = None,
                            efficiency_score: Optional[float] = None,
                            estimated_cost: Optional[float] = None):
        """
        Log an agent interaction with automatic timestamp.
        
        Args:
            question: User's question
            response: Agent's response
            response_time: Total response time in seconds
            processing_time: Agent processing time in seconds
            navigation_command: Whether this was a navigation command
            service_call: Whether this was a service call
            topic_data_request: Whether this was a topic data request
            success: Whether the interaction was successful
            error_message: Error message if any
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            prompt_ms: Prompt processing time in milliseconds
            predicted_ms: Prediction time in milliseconds
            tokens_per_second: Tokens generated per second
            efficiency_score: Efficiency score (tokens/second)
            estimated_cost: Estimated cost in USD
        """
        stats = InteractionStats(
            timestamp=time.time(),
            question=question,
            response=response,
            response_time=response_time,
            processing_time=processing_time,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            prompt_ms=prompt_ms,
            predicted_ms=predicted_ms,
            tokens_per_second=tokens_per_second,
            efficiency_score=efficiency_score,
            estimated_cost=estimated_cost,
            navigation_command=navigation_command,
            service_call=service_call,
            topic_data_request=topic_data_request,
            success=success,
            error_message=error_message
        )
        
        self.log_interaction(stats)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of logged statistics."""
        return {
            'log_file': str(self.csv_file),
            'interaction_count': self.interaction_count,
            'log_dir': str(self.log_dir)
        }
    
    def close(self):
        """Close the logger (no-op for CSV logging)."""
        pass
