"""
ROS2 Agent Package

AI agent interface for mobile robot interaction using ROS2 and Gemma3.
"""

__version__ = "1.0.0"
__author__ = "User"
__email__ = "user@example.com"

from .agent_node import AgentNode
from .gemma_client import GemmaClient, ChatMessage
from .data_bridge import DataBridge, SensorData
from .service_handler import ServiceHandler
from .cli_interface import AgentCLI

__all__ = [
    'AgentNode',
    'GemmaClient', 
    'ChatMessage',
    'DataBridge',
    'SensorData',
    'ServiceHandler',
    'AgentCLI'
]




