#!/usr/bin/env python3
"""
Launch file for ROS2 Agent CLI

This launch file starts the ROS2 agent CLI interface.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Generate launch description for the ROS2 agent CLI."""
    
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
    
    # Log info
    log_info = LogInfo(
        msg="Starting ROS2 Agent CLI"
    )
    
    return LaunchDescription([
        config_file_arg,
        log_info
    ])




