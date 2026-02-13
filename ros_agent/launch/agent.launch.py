#!/usr/bin/env python3
"""
Launch file for ROS2 Agent

This launch file starts the ROS2 agent node with proper configuration.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """Generate launch description for the ROS2 agent."""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to configuration file'
    )
    
    # Get package share directory
    pkg_share = FindPackageShare(package='ros_agent')
    
    # Default config file path
    default_config = PathJoinSubstitution([
        pkg_share,
        'config',
        'config.yaml'
    ])
    
    # Configuration file path
    config_file = LaunchConfiguration('config_file')
    
    # Agent node
    agent_node = Node(
        package='ros_agent',
        executable='agent_node',
        name='ros_agent',
        output='screen',
        parameters=[{
            'config_file': config_file
        }],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    # Log info
    log_info = LogInfo(
        msg="Starting ROS2 Agent Node"
    )
    
    return LaunchDescription([
        config_file_arg,
        log_info,
        agent_node
    ])

