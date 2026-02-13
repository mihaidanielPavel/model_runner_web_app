#!/usr/bin/env python3
"""
Launch file for ROS2 Agent Web Dashboard

Launches both the ROS2 agent node and Flask web server.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """Generate launch description for the web dashboard."""
    
    # Get package share directory
    pkg_share = get_package_share_directory('ros_agent')
    
    # Launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_share, 'config', 'config.yaml'),
        description='Path to configuration file'
    )
    
    web_port_arg = DeclareLaunchArgument(
        'web_port',
        default_value='5000',
        description='Port for the web server'
    )
    
    web_host_arg = DeclareLaunchArgument(
        'web_host',
        default_value='0.0.0.0',
        description='Host for the web server'
    )
    
    web_debug_arg = DeclareLaunchArgument(
        'web_debug',
        default_value='false',
        description='Enable debug mode for web server'
    )
    
    # ROS2 Agent Node
    agent_node = Node(
        package='ros_agent',
        executable='agent_node',
        name='ros_agent',
        output='screen',
        emulate_tty=True
    )
    
    # Flask Web Server
    web_server = ExecuteProcess(
        cmd=[
            'python3', '-m', 'ros_agent.web_server',
            '--config', LaunchConfiguration('config_file'),
            '--port', LaunchConfiguration('web_port'),
            '--host', LaunchConfiguration('web_host'),
            '--debug', LaunchConfiguration('web_debug')
        ],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(PythonExpression([
            "'true' == '", LaunchConfiguration('web_debug'), "' or 'false' == '", LaunchConfiguration('web_debug'), "'"
        ]))
    )
    
    # Alternative web server command without debug flag
    web_server_no_debug = ExecuteProcess(
        cmd=[
            'python3', '-m', 'ros_agent.web_server',
            '--config', LaunchConfiguration('config_file'),
            '--port', LaunchConfiguration('web_port'),
            '--host', LaunchConfiguration('web_host')
        ],
        output='screen',
        emulate_tty=True,
        condition=IfCondition(PythonExpression([
            "'true' != '", LaunchConfiguration('web_debug'), "' and 'false' != '", LaunchConfiguration('web_debug'), "'"
        ]))
    )
    
    # Log information
    log_info = LogInfo(
        msg=[
            'ROS2 Agent Web Dashboard starting...\n',
            'Web server will be available at: http://',
            LaunchConfiguration('web_host'),
            ':',
            LaunchConfiguration('web_port'),
            '\n',
            'Configuration file: ',
            LaunchConfiguration('config_file')
        ]
    )
    
    return LaunchDescription([
        config_file_arg,
        web_port_arg,
        web_host_arg,
        web_debug_arg,
        log_info,
        agent_node,
        web_server,
        web_server_no_debug
    ])
