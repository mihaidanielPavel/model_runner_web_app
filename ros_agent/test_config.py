#!/usr/bin/env python3
"""
Test script to verify configuration loading works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ros_agent'))

from ros_agent.agent_node import AgentNode
import yaml

def test_config_loading():
    """Test that configuration loading works without ROS2."""
    print("Testing configuration loading...")
    
    # Test default config
    node = AgentNode()
    print(f"✓ AgentNode initialized successfully")
    print(f"✓ Configuration loaded: {len(node.config)} sections")
    
    # Test specific config sections
    if 'gemma' in node.config:
        print(f"✓ Gemma config: {node.config['gemma']}")
    
    if 'web_server' in node.config:
        print(f"✓ Web server config: {node.config['web_server']}")
    
    if 'sensors' in node.config:
        print(f"✓ Sensors config: {len(node.config['sensors'])} sensor types")
    
    if 'navigation' in node.config:
        print(f"✓ Navigation config: {node.config['navigation']}")
    
    print("✓ All configuration tests passed!")

if __name__ == '__main__':
    test_config_loading()
