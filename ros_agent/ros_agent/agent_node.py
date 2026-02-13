#!/usr/bin/env python3
"""
Main ROS2 Agent Node

This is the core ROS2 node that subscribes to sensor topics,
manages data collection, and provides the interface for the AI agent.
"""

import logging
import yaml
import time
import math
from typing import Dict, List, Optional, Any
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

# ROS2 message imports
from sensor_msgs.msg import Image, LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap, GetCostmap, LoadMap, SaveMap
from std_srvs.srv import Empty

# Local imports
from .data_bridge import DataBridge, SensorData
from .service_handler import ServiceHandler
from .gemma_client import GemmaClient
from .statistics_logger import StatisticsLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This outputs to console
    ]
)

logger = logging.getLogger(__name__)

class NavigationHandler:
    """Handles navigation2 stack integration for the agent."""
    
    def __init__(self, node: Node, config: Dict[str, Any] = None):
        """Initialize navigation handler with ROS2 node."""
        self.node = node
        self.config = config or {}
        self.navigation_client = None
        self.current_goal_handle = None
        self.navigation_status = "idle"  # idle, navigating, completed, failed
        self.current_pose = None
        self.current_pose_msg = None  # Store full pose message for latest data access
        
        # Get navigation settings from config
        nav_config = self.config.get('navigation', {})
        self.enabled = nav_config.get('enabled', True)
        self.action_server = nav_config.get('action_server', 'navigate_to_pose')
        self.pose_topic = nav_config.get('pose_topic', '/amcl_pose')
        self.timeout = nav_config.get('timeout', 30.0)
        
        if self.enabled:
            # Initialize navigation action client
            self._setup_navigation_client()
            
            # Setup pose subscriber for current position
            self._setup_pose_subscriber()
    
    def _setup_navigation_client(self):
        """Setup navigation action client."""
        try:
            self.navigation_client = ActionClient(
                self.node, 
                NavigateToPose, 
                self.action_server
            )
            logger.info(f"Navigation action client initialized for {self.action_server}")
        except Exception as e:
            logger.error(f"Failed to initialize navigation client: {e}")
    
    def _setup_pose_subscriber(self):
        """Setup subscriber for current robot pose."""
        try:
            self.pose_subscriber = self.node.create_subscription(
                PoseWithCovarianceStamped,
                self.pose_topic,
                self._pose_callback,
                QoSProfile(depth=10)
            )
            logger.info(f"Pose subscriber initialized for {self.pose_topic}")
        except Exception as e:
            logger.error(f"Failed to setup pose subscriber: {e}")
    
    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        """Handle pose updates."""
        self.current_pose = msg.pose.pose
        self.current_pose_msg = msg  # Store the full message for latest data access
    
    def move_to_position(self, x: float, y: float, yaw: float = 0.0) -> bool:
        """
        Move robot to specified position.
        
        Args:
            x: Target x coordinate
            y: Target y coordinate  
            yaw: Target orientation (optional, defaults to 0.0)
            
        Returns:
            True if goal was sent successfully, False otherwise
        """
        if not self.navigation_client:
            logger.error("Navigation client not available")
            return False
        
        try:
            # Wait for action server
            if not self.navigation_client.wait_for_server(timeout_sec=5.0):
                logger.error("Navigation action server not available")
                return False
            
            # Create goal
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
            
            # Set position
            goal_msg.pose.pose.position.x = float(x)
            goal_msg.pose.pose.position.y = float(y)
            goal_msg.pose.pose.position.z = 0.0
            
            # Set orientation (convert yaw to quaternion)
            import math
            goal_msg.pose.pose.orientation.x = 0.0
            goal_msg.pose.pose.orientation.y = 0.0
            goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            
            # Send goal
            self.current_goal_handle = self.navigation_client.send_goal_async(
                goal_msg,
                feedback_callback=self._navigation_feedback_callback
            )
            
            self.navigation_status = "navigating"
            logger.info(f"Navigation goal sent to position ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send navigation goal: {e}")
            return False
    
    def _navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        try:
            # You can add feedback processing here if needed
            pass
        except Exception as e:
            logger.error(f"Navigation feedback error: {e}")
    
    def stop_navigation(self) -> bool:
        """
        Stop current navigation.
        
        Returns:
            True if stop command was sent successfully
        """
        try:
            if self.current_goal_handle and self.navigation_status == "navigating":
                # Cancel the current goal
                cancel_future = self.current_goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=5.0)
                
                if cancel_future.done():
                    cancel_response = cancel_future.result()
                    if cancel_response.status == 1:  # SUCCESS
                        self.navigation_status = "idle"
                        logger.info("Navigation stopped successfully")
                        return True
                    else:
                        logger.warning(f"Navigation cancel response: {cancel_response.status}")
                        self.navigation_status = "idle"  # Still mark as idle
                        return True
                else:
                    logger.warning("Navigation cancel timeout")
                    self.navigation_status = "idle"  # Still mark as idle
                    return True
            else:
                logger.info("No active navigation to stop")
                return True
        except Exception as e:
            logger.error(f"Failed to stop navigation: {e}")
            return False
    
    def get_navigation_status(self) -> Dict[str, Any]:
        """Get current navigation status."""
        status = {
            'status': self.navigation_status,
            'current_pose': None,
            'goal_active': self.navigation_status == "navigating"
        }
        
        if self.current_pose_msg:
            # Use the latest AMCL pose data (map coordinates)
            pose = self.current_pose_msg.pose.pose
            # Convert ROS2 Time to serializable format
            stamp = self.current_pose_msg.header.stamp
            timestamp_serializable = {
                'sec': stamp.sec,
                'nanosec': stamp.nanosec
            }
            status['current_pose'] = {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': pose.position.z,
                'frame_id': self.current_pose_msg.header.frame_id,
                'timestamp': timestamp_serializable
            }
        
        return status
    
    def parse_navigation_command(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Parse navigation commands from natural language.
        
        Args:
            command: Natural language command
            
        Returns:
            Parsed command data or None if not a navigation command
        """
        command_lower = command.lower().strip()
        
        # Parse "move to position (x,y)" commands
        import re
        
        # Pattern for "move to position (x,y)" or "move to (x,y)"
        move_pattern = r'move\s+to\s+(?:position\s+)?\(?\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)?'
        match = re.search(move_pattern, command_lower)
        
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                return {
                    'action': 'move_to_position',
                    'x': x,
                    'y': y,
                    'yaw': 0.0
                }
            except ValueError:
                logger.error(f"Invalid coordinates in command: {command}")
                return None
        
        # Parse "stop" commands
        if any(word in command_lower for word in ['stop', 'halt', 'cancel']):
            return {
                'action': 'stop_navigation'
            }
        
        # Parse "status" commands
        if any(word in command_lower for word in ['status', 'where', 'position']):
            return {
                'action': 'get_status'
            }
        
        return None
    
    def execute_navigation_command(self, command_data: Dict[str, Any]) -> str:
        """
        Execute parsed navigation command.
        
        Args:
            command_data: Parsed command data
            
        Returns:
            Status message
        """
        action = command_data.get('action')
        
        if action == 'move_to_position':
            x = command_data.get('x')
            y = command_data.get('y')
            yaw = command_data.get('yaw', 0.0)
            
            if self.move_to_position(x, y, yaw):
                return f"Moving to position ({x}, {y})"
            else:
                return "Failed to start navigation"
        
        elif action == 'stop_navigation':
            if self.stop_navigation():
                return "Navigation stopped"
            else:
                return "Failed to stop navigation"
        
        elif action == 'get_status':
            status = self.get_navigation_status()
            if status['current_pose']:
                pose = status['current_pose']
                return f"Robot is at position ({pose['x']:.2f}, {pose['y']:.2f}). Navigation status: {status['status']}"
            else:
                return f"Navigation status: {status['status']} (pose unknown)"
        
        return "Unknown navigation command"

class AgentNode(Node):
    """Main ROS2 node for the agent system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the agent node."""
        super().__init__('ros_agent')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_bridge = DataBridge()
        self.service_handler = ServiceHandler(self)
        self.navigation_handler = NavigationHandler(self, self.config)
        self.gemma_client = None
        
        # Initialize statistics logger
        log_config = self.config.get('logging', {})
        log_dir = log_config.get('log_dir', 'logs')
        self.stats_logger = StatisticsLogger(log_dir=log_dir)
        
        # Initialize Gemma client if configured
        if self.config.get('gemma', {}).get('api_url'):
            try:
                self.gemma_client = GemmaClient(
                    api_url=self.config['gemma']['api_url'],
                    model_name=self.config['gemma']['model_name'],
                    timeout=self.config['gemma']['timeout'],
                    max_retries=self.config['gemma']['max_retries'],
                    logger=self.get_logger()
                )
                logger.info("Gemma client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemma client: {e}")
        
        # Topic subscribers
        self.subscribers: Dict[str, Any] = {}
        
        # Latest message cache for on-demand topic reading
        self.latest_messages: Dict[str, Any] = {}
        
        # Location labels memory for AI agent
        self.location_labels: Dict[str, Dict[str, float]] = {}
        
        # Load location labels from YAML file
        self._load_location_labels()
        
        # Auto-discovery settings
        self.discovery_enabled = self.config.get('discovery', {}).get('enabled', True)
        self.scan_interval = self.config.get('discovery', {}).get('scan_interval', 5.0)
        self.last_discovery_time = 0.0
        
        # Setup QoS profile
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Initialize sensor subscribers
        self._setup_sensor_subscribers()
        
        # Setup discovery timer
        if self.discovery_enabled:
            self.discovery_timer = self.create_timer(
                self.scan_interval, 
                self._discover_topics
            )
        
        # Setup cleanup timer
        self.cleanup_timer = self.create_timer(30.0, self._cleanup_old_data)
        
        logger.info("Agent node initialized successfully")
    
    def _load_location_labels(self):
        """Load location labels from YAML file."""
        try:
            labels_file = Path(__file__).parent.parent / 'config' / 'location_labels.yaml'
            if labels_file.exists():
                with open(labels_file, 'r') as f:
                    labels_data = yaml.safe_load(f) or {}
                    self.location_labels = labels_data.get('locations', {})
                logger.info(f"Loaded {len(self.location_labels)} location labels from {labels_file}")
            else:
                logger.info("No location labels file found, starting with empty labels")
                self.location_labels = {}
        except Exception as e:
            logger.error(f"Failed to load location labels: {e}")
            self.location_labels = {}
    
    def _save_location_labels(self):
        """Save location labels to YAML file."""
        try:
            labels_file = Path(__file__).parent.parent / 'config' / 'location_labels.yaml'
            labels_file.parent.mkdir(parents=True, exist_ok=True)
            
            labels_data = {
                'locations': self.location_labels,
                'last_updated': time.time()
            }
            
            with open(labels_file, 'w') as f:
                yaml.dump(labels_data, f, default_flow_style=False)
            
            logger.info(f"Saved {len(self.location_labels)} location labels to {labels_file}")
        except Exception as e:
            logger.error(f"Failed to save location labels: {e}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try multiple possible locations for config file
            possible_paths = [
                Path(__file__).parent.parent / 'config' / 'config.yaml',  # Development
                Path('/opt/ros/humble/share/ros_agent/config/config.yaml'),  # ROS2 install
                Path('/root/ros_ws/install/ros_agent/share/ros_agent/config/config.yaml'),  # Colcon install
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path is None:
                logger.warning("Config file not found in any standard location, using defaults")
                return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'gemma': {
                'api_url': 'http://model-runner.docker.internal',
                'model_name': 'ai/gemma3',
                'timeout': 30,
                'max_retries': 3
            },
            'sensors': {
                'camera': {'topics': ['/camera/image_raw'], 'enabled': True},
                'lidar': {'topics': ['/scan'], 'enabled': True},
                'odometry': {'topics': ['/odom'], 'enabled': True},
                'imu': {'topics': ['/imu'], 'enabled': True}
            },
            'discovery': {
                'enabled': True,
                'scan_interval': 5.0,
                'exclude_patterns': ['/rosout', '/tf', '/tf_static']
            }
        }
    
    def _setup_sensor_subscribers(self):
        """Setup subscribers for configured sensor topics."""
        sensors_config = self.config.get('sensors', {})
        
        for sensor_type, config in sensors_config.items():
            if not config.get('enabled', True):
                continue
            
            topics = config.get('topics', [])
            for topic in topics:
                self._create_subscriber(sensor_type, topic)
    
    def _create_subscriber(self, sensor_type: str, topic_name: str):
        """Create a subscriber for a specific sensor topic."""
        try:
            if sensor_type == 'camera':
                subscriber = self.create_subscription(
                    Image,
                    topic_name,
                    lambda msg: self._camera_callback(msg, topic_name),
                    self.qos_profile
                )
            elif sensor_type == 'lidar':
                subscriber = self.create_subscription(
                    LaserScan,
                    topic_name,
                    lambda msg: self._lidar_callback(msg, topic_name),
                    self.qos_profile
                )
            elif sensor_type == 'odometry':
                subscriber = self.create_subscription(
                    Odometry,
                    topic_name,
                    lambda msg: self._odometry_callback(msg, topic_name),
                    self.qos_profile
                )
            elif sensor_type == 'imu':
                subscriber = self.create_subscription(
                    Imu,
                    topic_name,
                    lambda msg: self._imu_callback(msg, topic_name),
                    self.qos_profile
                )
            else:
                logger.warning(f"Unknown sensor type: {sensor_type}")
                return
            
            self.subscribers[topic_name] = subscriber
            logger.info(f"Created subscriber for {sensor_type} topic: {topic_name}")
            
        except Exception as e:
            logger.error(f"Failed to create subscriber for {topic_name}: {e}")
    
    def _camera_callback(self, msg: Image, topic_name: str):
        """Handle camera image messages."""
        try:
            # Store latest message for on-demand access
            self.latest_messages[topic_name] = msg
            
            sensor_data = self.data_bridge.process_image_message(msg, topic_name)
            self.data_bridge.update_sensor_data(sensor_data)
        except Exception as e:
            logger.error(f"Error processing camera message from {topic_name}: {e}")
    
    def _lidar_callback(self, msg: LaserScan, topic_name: str):
        """Handle LiDAR scan messages."""
        try:
            # Store latest message for on-demand access
            self.latest_messages[topic_name] = msg
            
            sensor_data = self.data_bridge.process_laser_scan_message(msg, topic_name)
            self.data_bridge.update_sensor_data(sensor_data)
        except Exception as e:
            logger.error(f"Error processing LiDAR message from {topic_name}: {e}")
    
    def _odometry_callback(self, msg: Odometry, topic_name: str):
        """Handle odometry messages."""
        try:
            # Store latest message for on-demand access
            self.latest_messages[topic_name] = msg
            
            sensor_data = self.data_bridge.process_odometry_message(msg, topic_name)
            self.data_bridge.update_sensor_data(sensor_data)
        except Exception as e:
            logger.error(f"Error processing odometry message from {topic_name}: {e}")
    
    def _imu_callback(self, msg: Imu, topic_name: str):
        """Handle IMU messages."""
        try:
            # Store latest message for on-demand access
            self.latest_messages[topic_name] = msg
            
            sensor_data = self.data_bridge.process_imu_message(msg, topic_name)
            self.data_bridge.update_sensor_data(sensor_data)
        except Exception as e:
            logger.error(f"Error processing IMU message from {topic_name}: {e}")
    
    def _discover_topics(self):
        """Discover and subscribe to new sensor topics."""
        try:
            current_time = time.time()
            if current_time - self.last_discovery_time < self.scan_interval:
                return
            
            # Get all available topics
            topic_names_and_types = self.get_topic_names_and_types()
            exclude_patterns = self.config.get('discovery', {}).get('exclude_patterns', [])
            
            # Find new sensor topics
            for topic_name, topic_types in topic_names_and_types:
                # Skip excluded topics
                if any(pattern in topic_name for pattern in exclude_patterns):
                    continue
                
                # Skip if already subscribed
                if topic_name in self.subscribers:
                    continue
                
                # Determine sensor type based on message type
                sensor_type = self._determine_sensor_type(topic_types)
                if sensor_type:
                    self._create_subscriber(sensor_type, topic_name)
            
            self.last_discovery_time = current_time
            
        except Exception as e:
            logger.error(f"Topic discovery failed: {e}")
    
    def _determine_sensor_type(self, topic_types: List[str]) -> Optional[str]:
        """Determine sensor type from topic message types."""
        for topic_type in topic_types:
            if 'sensor_msgs/msg/Image' in topic_type:
                return 'camera'
            elif 'sensor_msgs/msg/LaserScan' in topic_type:
                return 'lidar'
            elif 'nav_msgs/msg/Odometry' in topic_type:
                return 'odometry'
            elif 'sensor_msgs/msg/Imu' in topic_type:
                return 'imu'
        return None
    
    def _cleanup_old_data(self):
        """Cleanup old sensor data."""
        try:
            removed_count = self.data_bridge.clear_old_data(max_age=60.0)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old sensor data entries")
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def get_sensor_summary(self) -> str:
        """Get current sensor data summary."""
        return self.data_bridge.get_sensor_summary()
    
    def get_services_summary(self) -> str:
        """Get available services summary."""
        return self.service_handler.get_services_summary()
    
    def call_service(self, service_name: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a ROS2 service."""
        return self.service_handler.call_service(service_name, request_data)
    
    def get_service_definition(self, service_name: str) -> Optional[str]:
        """Get service definition for agent to compose requests."""
        return self.service_handler.get_service_definition(service_name)
    
    def compose_service_request(self, service_name: str, natural_language_request: str) -> Optional[Dict[str, Any]]:
        """Compose service request from natural language."""
        return self.service_handler.compose_service_request(service_name, natural_language_request)
    
    def ask_agent(self, question: str) -> Optional[str]:
        """Ask the AI agent a question."""
        start_time = time.time()
        
        if not self.gemma_client:
            response = "AI agent not available (Gemma client not initialized)"
            response_time = time.time() - start_time
            self.stats_logger.log_agent_interaction(
                question=question,
                response=response,
                response_time=response_time,
                processing_time=response_time,
                success=False,
                error_message="Gemma client not initialized"
            )
            return response
        
        try:
            # Get current sensor context
            sensor_context = self.get_sensor_summary()
            
            # Get navigation status for context (using latest AMCL pose)
            nav_status = self.navigation_handler.get_navigation_status()
            nav_context = f"Navigation status: {nav_status['status']}"
            if nav_status['current_pose']:
                pose = nav_status['current_pose']
                nav_context += f", Current position (map frame): ({pose['x']:.3f}, {pose['y']:.3f})"
                nav_context += f", Frame: {pose['frame_id']}"
            
            # Add available topics context
            available_topics = self.get_available_topics()
            topics_context = f"Available topics with latest data: {', '.join(available_topics)}"
            
            # Add available services context
            services_summary = self.get_services_summary()
            
            # Create minimal context based on question type
            context_parts = []
            
            # Always include basic robot info
            context_parts.append(f"Robot: {nav_context}")
            
            # Add relevant context based on question content
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['sensor', 'data', 'scan', 'camera', 'lidar', 'imu', 'odom']):
                context_parts.append(f"Sensors: {sensor_context}")
            
            if any(word in question_lower for word in ['topic', 'data', 'latest', 'current']):
                context_parts.append(f"Topics: {topics_context}")
            
            if any(word in question_lower for word in ['service', 'call', 'clear', 'costmap']):
                context_parts.append(f"Services: {services_summary}")
            
            # Add location labels if relevant
            if any(word in question_lower for word in ['location', 'label', 'storage', 'home', 'kitchen', 'office', 'room']):
                location_labels_info = self.list_location_labels()
                context_parts.append(f"Location Labels: {location_labels_info}")
            
            # Combine minimal context
            full_context = "\n".join(context_parts)
            
            # Add function definitions (compact)
            functions_info = """Functions: 
- get_navigation_status(): Get robot position and navigation status
- get_latest_topic_data(topic): Get data from specific topic (use /amcl_pose for position, /scan for LiDAR)
- move_to_position(x,y,yaw): Move robot to coordinates OR location label (e.g., "storage", "home")
- stop_navigation(): Stop current navigation
- call_service(name,data): Call ROS2 service
- add_location_label(label,x,y,yaw): Add named location (e.g., "storage" at coordinates)
- list_location_labels(): Show all saved location labels
- remove_location_label(label): Remove a location label"""
            
            full_context = f"{full_context}\n{functions_info}"
            
            # Ask the agent
            agent_start_time = time.time()
            response_data = self.gemma_client.generate_response(question, full_context)
            processing_time = time.time() - agent_start_time
            
            if response_data:
                response = response_data.get('response', '')
                usage_data = response_data.get('usage', {})
                timings_data = response_data.get('timings', {})
                metrics_data = response_data.get('metrics', {})
            else:
                response = None
                usage_data = {}
                timings_data = {}
                metrics_data = {}
            
            # Check if the response contains function calls and execute them
            final_response = self._process_agent_response(response, question)
            
            response_time = time.time() - start_time
            
            # Log statistics with token and timing data
            self.stats_logger.log_agent_interaction(
                question=question,
                response=final_response or "No response",
                response_time=response_time,
                processing_time=processing_time,
                success=final_response is not None,
                tokens_input=usage_data.get('prompt_tokens'),
                tokens_output=usage_data.get('completion_tokens'),
                prompt_ms=timings_data.get('prompt_ms'),
                predicted_ms=timings_data.get('predicted_ms'),
                tokens_per_second=metrics_data.get('completion_tokens_per_second'),
                efficiency_score=metrics_data.get('efficiency_score'),
                estimated_cost=metrics_data.get('estimated_cost')
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            response = f"Error querying agent: {e}"
            response_time = time.time() - start_time
            
            # Log error statistics
            self.stats_logger.log_agent_interaction(
                question=question,
                response=response,
                response_time=response_time,
                processing_time=response_time,
                success=False,
                error_message=str(e),
                tokens_input=None,
                tokens_output=None,
                prompt_ms=None,
                predicted_ms=None,
                tokens_per_second=None,
                efficiency_score=None,
                estimated_cost=None
            )
            return response
    
    def ask_agent_with_metadata(self, question: str) -> Dict[str, Any]:
        """Ask the AI agent a question and return response with metadata."""
        start_time = time.time()
        
        if not self.gemma_client:
            response = "AI agent not available (Gemma client not initialized)"
            response_time = time.time() - start_time
            self.stats_logger.log_agent_interaction(
                question=question,
                response=response,
                response_time=response_time,
                processing_time=response_time,
                success=False,
                error_message="Gemma client not initialized"
            )
            return {
                'response': response,
                'success': False,
                'error': 'Gemma client not initialized',
                'response_time': response_time
            }
        
        try:
            # Get current sensor context
            sensor_context = self.get_sensor_summary()
            
            # Get navigation status for context (using latest AMCL pose)
            nav_status = self.navigation_handler.get_navigation_status()
            nav_context = f"Navigation status: {nav_status['status']}"
            if nav_status['current_pose']:
                pose = nav_status['current_pose']
                nav_context += f", Current position (map frame): ({pose['x']:.3f}, {pose['y']:.3f})"
                nav_context += f", Frame: {pose['frame_id']}"
            
            # Add available topics context
            available_topics = self.get_available_topics()
            topics_context = f"Available topics with latest data: {', '.join(available_topics)}"
            
            # Add available services context
            services_summary = self.get_services_summary()
            
            # Create minimal context based on question type
            context_parts = []
            
            # Always include basic robot info
            context_parts.append(f"Robot: {nav_context}")
            
            # Add relevant context based on question content
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['sensor', 'data', 'scan', 'camera', 'lidar', 'imu', 'odom']):
                context_parts.append(f"Sensors: {sensor_context}")
            
            if any(word in question_lower for word in ['topic', 'data', 'latest', 'current']):
                context_parts.append(f"Topics: {topics_context}")
            
            if any(word in question_lower for word in ['service', 'call', 'clear', 'costmap']):
                context_parts.append(f"Services: {services_summary}")
            
            # Add location labels if relevant
            if any(word in question_lower for word in ['location', 'label', 'storage', 'home', 'kitchen', 'office', 'room']):
                location_labels_info = self.list_location_labels()
                context_parts.append(f"Location Labels: {location_labels_info}")
            
            # Combine minimal context
            full_context = "\n".join(context_parts)
            
            # Add function definitions (compact)
            functions_info = """Functions: 
- get_navigation_status(): Get robot position and navigation status
- get_latest_topic_data(topic): Get data from specific topic (use /amcl_pose for position, /scan for LiDAR)
- move_to_position(x,y,yaw): Move robot to coordinates OR location label (e.g., "storage", "home")
- stop_navigation(): Stop current navigation
- call_service(name,data): Call ROS2 service
- add_location_label(label,x,y,yaw): Add named location (e.g., "storage" at coordinates)
- list_location_labels(): Show all saved location labels
- remove_location_label(label): Remove a location label"""
            
            full_context = f"{full_context}\n{functions_info}"
            
            # Ask the agent
            agent_start_time = time.time()
            response_data = self.gemma_client.generate_response(question, full_context)
            processing_time = time.time() - agent_start_time
            
            if response_data:
                response_text = response_data.get('response', 'No response')
                
                # Extract token and timing data directly from response_data
                usage_data = response_data.get('usage', {})
                timings_data = response_data.get('timings', {})
                metrics_data = response_data.get('metrics', {})
                
                # Check if the response contains function calls and execute them
                final_response = self._process_agent_response(response_text, question)
                
                response_time = time.time() - start_time
                
                # Log statistics with enhanced metadata
                self.stats_logger.log_agent_interaction(
                    question=question,
                    response=final_response or "No response",
                    response_time=response_time,
                    processing_time=processing_time,
                    tokens_input=usage_data.get('prompt_tokens'),
                    tokens_output=usage_data.get('completion_tokens'),
                    prompt_ms=timings_data.get('prompt_ms'),
                    predicted_ms=timings_data.get('predicted_ms'),
                    tokens_per_second=metrics_data.get('overall_tokens_per_second'),
                    efficiency_score=metrics_data.get('efficiency_score'),
                    estimated_cost=self._calculate_cost_estimate(usage_data),
                    success=final_response is not None
                )
                
                return {
                    'response': final_response,
                    'success': True,
                    'response_time': response_time,
                    'processing_time': processing_time,
                    'metadata': {
                        'usage': usage_data,
                        'timings': timings_data,
                        'metrics': metrics_data
                    }
                }
            else:
                response_time = time.time() - start_time
                error_msg = "Failed to get response from AI agent"
                
                # Log error statistics
                self.stats_logger.log_agent_interaction(
                    question=question,
                    response=error_msg,
                    response_time=response_time,
                    processing_time=processing_time,
                    success=False,
                    error_message="No response from AI agent"
                )
                
                return {
                    'response': error_msg,
                    'success': False,
                    'error': 'No response from AI agent',
                    'response_time': response_time
                }
                
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            response_time = time.time() - start_time
            
            # Log error statistics
            self.stats_logger.log_agent_interaction(
                question=question,
                response=f"Error: {str(e)}",
                response_time=response_time,
                processing_time=response_time,
                success=False,
                error_message=str(e)
            )
            
            return {
                'response': f"Error processing query: {str(e)}",
                'success': False,
                'error': str(e),
                'response_time': response_time
            }
    
    def _calculate_cost_estimate(self, usage_data: Dict[str, Any]) -> Optional[float]:
        """Calculate estimated cost based on token usage."""
        try:
            # Rough cost estimation (adjust based on actual pricing)
            # Assuming $0.0001 per 1K tokens for input and $0.0002 per 1K tokens for output
            prompt_tokens = usage_data.get('prompt_tokens', 0)
            completion_tokens = usage_data.get('completion_tokens', 0)
            
            input_cost = (prompt_tokens / 1000) * 0.0001
            output_cost = (completion_tokens / 1000) * 0.0002
            
            return input_cost + output_cost
        except:
            return None
    
    def _process_agent_response(self, response: str, original_question: str) -> str:
        """Process agent response and execute any function calls."""
        if not response:
            return response
        
        # Check for function calls in the response
        import re
        
        # Pattern to match function calls like: move_to_position(1.5, 2.0)
        function_pattern = r'(\w+)\(([^)]*)\)'
        matches = re.findall(function_pattern, response)
        
        if not matches:
            return response
        
        # Execute function calls
        execution_results = []
        for func_name, params_str in matches:
            try:
                result = self._execute_function_call(func_name, params_str)
                execution_results.append(f"Executed {func_name}({params_str}): {result}")
            except Exception as e:
                execution_results.append(f"Error executing {func_name}({params_str}): {str(e)}")
        
        # Combine original response with execution results
        if execution_results:
            return f"{response}\n\n[EXECUTION RESULTS]\n" + "\n".join(execution_results)
        
        return response
    
    def _execute_function_call(self, func_name: str, params_str: str) -> str:
        """Execute a function call based on the function name and parameters."""
        try:
            # Parse parameters (handle both positional and named parameters)
            params = []
            if params_str.strip():
                # Simple parameter parsing (handles basic types)
                param_parts = params_str.split(',')
                for part in param_parts:
                    part = part.strip()
                    
                    # Handle named parameters (e.g., "x=-1.5" or "label='storage'")
                    if '=' in part:
                        param_name, param_value = part.split('=', 1)
                        param_value = param_value.strip()
                        
                        # Parse the value part
                        if param_value.startswith('"') and param_value.endswith('"'):
                            params.append(param_value[1:-1])  # Remove quotes
                        elif param_value.startswith("'") and param_value.endswith("'"):
                            params.append(param_value[1:-1])  # Remove quotes
                        elif param_value.lower() in ['true', 'false']:
                            params.append(param_value.lower() == 'true')
                        elif '.' in param_value:
                            params.append(float(param_value))
                        elif param_value.isdigit() or (param_value.startswith('-') and param_value[1:].isdigit()):
                            params.append(int(param_value))
                        else:
                            params.append(param_value)  # Keep as string
                    else:
                        # Handle positional parameters
                        if part.startswith('"') and part.endswith('"'):
                            params.append(part[1:-1])  # Remove quotes
                        elif part.startswith("'") and part.endswith("'"):
                            params.append(part[1:-1])  # Remove quotes
                        elif part.lower() in ['true', 'false']:
                            params.append(part.lower() == 'true')
                        elif '.' in part:
                            params.append(float(part))
                        elif part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                            params.append(int(part))
                        else:
                            params.append(part)  # Keep as string
            
            # Execute the function
            if func_name == 'move_to_position':
                if len(params) >= 1:
                    # Check if first parameter is a location label
                    first_param = params[0]
                    if isinstance(first_param, str) and not first_param.replace('.', '').replace('-', '').isdigit():
                        # It's a location label, get coordinates
                        location_coords = self.get_location_label(first_param)
                        if location_coords:
                            return self.move_to_position(location_coords['x'], location_coords['y'], location_coords['yaw'])
                        else:
                            return f"Error: Location label '{first_param}' not found. Available labels: {', '.join(self.location_labels.keys())}"
                    elif len(params) >= 2:
                        # It's coordinates
                        x, y = float(params[0]), float(params[1])
                        yaw = float(params[2]) if len(params) > 2 else 0.0
                        return self.move_to_position(x, y, yaw)
                    else:
                        return "Error: move_to_position requires at least x and y coordinates or a location label"
                else:
                    return "Error: move_to_position requires at least x and y coordinates or a location label"
            
            elif func_name == 'stop_navigation':
                return self.stop_navigation()
            
            elif func_name == 'get_latest_topic_data':
                if len(params) >= 1:
                    topic_name = params[0]
                    
                    # Handle common topic name variations
                    if topic_name.lower() in ['position', 'pose', 'location', 'where']:
                        # Try common pose topics
                        pose_topics = ['/amcl_pose', '/robot_pose', '/pose', '/odom']
                        for pose_topic in pose_topics:
                            result = self.get_latest_topic_data(pose_topic)
                            if result:
                                return result
                        return "No pose data available. Try /amcl_pose or /odom topics."
                    
                    elif topic_name.lower() in ['scan', 'lidar', 'laser']:
                        # Try common LiDAR topics
                        lidar_topics = ['/scan', '/laser_scan', '/lidar/scan']
                        for lidar_topic in lidar_topics:
                            result = self.get_latest_topic_data(lidar_topic)
                            if result:
                                return result
                        return "No LiDAR data available. Try /scan topic."
                    
                    elif topic_name.lower() in ['camera', 'image', 'video']:
                        # Try common camera topics
                        camera_topics = ['/camera/image_raw', '/camera/rgb/image_raw', '/image_raw']
                        for camera_topic in camera_topics:
                            result = self.get_latest_topic_data(camera_topic)
                            if result:
                                return result
                        return "No camera data available. Try /camera/image_raw topic."
                    
                    else:
                        # Try the topic name as provided
                        result = self.get_latest_topic_data(topic_name)
                        if result:
                            return result
                        
                        # If not found, suggest available topics
                        available_topics = self.get_available_topics()
                        return f"No data available for topic {topic_name}. Available topics: {', '.join(available_topics[:5])}"
                else:
                    return "Error: get_latest_topic_data requires a topic name"
            
            elif func_name == 'call_service':
                if len(params) >= 2:
                    service_name = params[0]
                    request_data = {}  # Could be enhanced to parse JSON
                    result = self.call_service(service_name, request_data)
                    return f"Service call result: {result}" if result else "Service call failed"
                else:
                    return "Error: call_service requires service name and request data"
            
            elif func_name == 'get_navigation_status':
                status = self.get_navigation_status()
                if status['current_pose']:
                    pose = status['current_pose']
                    return f"Robot position: ({pose['x']:.3f}, {pose['y']:.3f}) in {pose['frame_id']} frame. Navigation status: {status['status']}"
                else:
                    return f"Navigation status: {status['status']} (position unknown)"
            
            elif func_name == 'add_location_label':
                if len(params) >= 3:
                    label = params[0]
                    x, y = float(params[1]), float(params[2])
                    yaw = float(params[3]) if len(params) > 3 else 0.0
                    return self.add_location_label(label, x, y, yaw)
                else:
                    return "Error: add_location_label requires label, x, y coordinates"
            
            elif func_name == 'list_location_labels':
                return self.list_location_labels()
            
            elif func_name == 'remove_location_label':
                if len(params) >= 1:
                    label = params[0]
                    return self.remove_location_label(label)
                else:
                    return "Error: remove_location_label requires a label name"
            
            else:
                return f"Unknown function: {func_name}"
                
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current node status."""
        nav_status = self.navigation_handler.get_navigation_status()
        return {
            'node_name': self.get_name(),
            'subscribers': len(self.subscribers),
            'sensor_data_count': len(self.data_bridge.data_cache),
            'services_discovered': len(self.service_handler.discovered_services),
            'gemma_client_available': self.gemma_client is not None,
            'discovery_enabled': self.discovery_enabled,
            'navigation_status': nav_status
        }
    
    def move_to_position(self, x: float, y: float, yaw: float = 0.0) -> str:
        """Move robot to specified position."""
        if self.navigation_handler.move_to_position(x, y, yaw):
            return f"Moving to position ({x}, {y})"
        else:
            return "Failed to start navigation"
    
    def stop_navigation(self) -> str:
        """Stop current navigation."""
        if self.navigation_handler.stop_navigation():
            return "Navigation stopped"
        else:
            return "Failed to stop navigation"
    
    def get_navigation_status(self) -> Dict[str, Any]:
        """Get navigation status."""
        return self.navigation_handler.get_navigation_status()
    
    def get_latest_topic_data(self, topic_name: str) -> Optional[str]:
        """
        Get the latest data from a specific topic.
        
        Args:
            topic_name: Name of the topic to read
            
        Returns:
            Formatted string with latest topic data or None if not available
        """
        if topic_name not in self.latest_messages:
            return None
        
        msg = self.latest_messages[topic_name]
        
        # Process the message based on its type
        if hasattr(msg, 'pose'):  # PoseWithCovarianceStamped
            pose = msg.pose.pose
            return f"Latest data from {topic_name}: Position ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}), Frame: {msg.header.frame_id}"
        elif hasattr(msg, 'ranges'):  # LaserScan
            ranges = [r for r in msg.ranges if not math.isnan(r) and r > 0]
            if ranges:
                min_range = min(ranges)
                max_range = max(ranges)
                avg_range = sum(ranges) / len(ranges)
                return f"Latest data from {topic_name}: {len(ranges)} valid ranges, min: {min_range:.3f}m, max: {max_range:.3f}m, avg: {avg_range:.3f}m"
            else:
                return f"Latest data from {topic_name}: No valid ranges"
        elif hasattr(msg, 'linear_acceleration'):  # Imu
            accel = msg.linear_acceleration
            gyro = msg.angular_velocity
            return f"Latest data from {topic_name}: Linear accel ({accel.x:.3f}, {accel.y:.3f}, {accel.z:.3f}), Angular vel ({gyro.x:.3f}, {gyro.y:.3f}, {gyro.z:.3f})"
        elif hasattr(msg, 'width'):  # Image
            return f"Latest data from {topic_name}: Image {msg.width}x{msg.height}, encoding: {msg.encoding}, size: {len(msg.data)} bytes"
        else:
            return f"Latest data from {topic_name}: Message type {type(msg).__name__} (raw data available)"
    
    def get_available_topics(self) -> List[str]:
        """Get list of topics with latest data available."""
        return list(self.latest_messages.keys())
    
    def add_location_label(self, label: str, x: float, y: float, yaw: float = 0.0) -> str:
        """
        Add a location label with coordinates.
        
        Args:
            label: Name of the location (e.g., "storage", "home")
            x: X coordinate
            y: Y coordinate
            yaw: Orientation (optional)
            
        Returns:
            Success message
        """
        self.location_labels[label.lower()] = {
            'x': float(x),
            'y': float(y),
            'yaw': float(yaw)
        }
        # Auto-save location labels
        self._save_location_labels()
        return f"Location '{label}' added at position ({x}, {y}, {yaw})"
    
    def get_location_label(self, label: str) -> Optional[Dict[str, float]]:
        """
        Get coordinates for a location label.
        
        Args:
            label: Name of the location
            
        Returns:
            Dictionary with x, y, yaw coordinates or None if not found
        """
        return self.location_labels.get(label.lower())
    
    def list_location_labels(self) -> str:
        """Get a formatted list of all location labels."""
        if not self.location_labels:
            return "No location labels defined"
        
        labels_list = []
        for label, coords in self.location_labels.items():
            labels_list.append(f"{label}: ({coords['x']}, {coords['y']}, {coords['yaw']})")
        
        return f"Location labels: {', '.join(labels_list)}"
    
    def remove_location_label(self, label: str) -> str:
        """
        Remove a location label.
        
        Args:
            label: Name of the location to remove
            
        Returns:
            Success message
        """
        if label.lower() in self.location_labels:
            del self.location_labels[label.lower()]
            # Auto-save location labels
            self._save_location_labels()
            return f"Location label '{label}' removed"
        else:
            return f"Location label '{label}' not found"
    
    def _load_location_labels(self):
        """Load location labels from YAML file."""
        try:
            # Try multiple possible locations for location labels file
            possible_paths = [
                Path(__file__).parent.parent / 'config' / 'location_labels.yaml',  # Development
                Path('/opt/ros/humble/share/ros_agent/config/location_labels.yaml'),  # ROS2 install
                Path('/root/ros_ws/install/ros_agent/share/ros_agent/config/location_labels.yaml'),  # Colcon install
            ]
            
            labels_file = None
            for path in possible_paths:
                if path.exists():
                    labels_file = path
                    break
            
            if labels_file is None:
                # Use the first writable location
                labels_file = possible_paths[0]
                logger.info(f"Location labels file not found, will create at: {labels_file}")
                self.location_labels = {}
            else:
                with open(labels_file, 'r') as f:
                    labels_data = yaml.safe_load(f) or {}
                    self.location_labels = labels_data.get('locations', {})
                logger.info(f"Loaded {len(self.location_labels)} location labels from {labels_file}")
        except Exception as e:
            logger.error(f"Failed to load location labels: {e}")
            self.location_labels = {}
    
    def _save_location_labels(self):
        """Save location labels to YAML file."""
        try:
            # Try multiple possible locations for location labels file
            possible_paths = [
                Path(__file__).parent.parent / 'config',  # Development
                Path('/opt/ros/humble/share/ros_agent/config'),  # ROS2 install
                Path('/root/ros_ws/install/ros_agent/share/ros_agent/config'),  # Colcon install
            ]
            
            config_dir = None
            for path in possible_paths:
                if path.exists() or path.parent.exists():
                    config_dir = path
                    break
            
            if config_dir is None:
                config_dir = possible_paths[0]  # Use development path as fallback
            
            config_dir.mkdir(parents=True, exist_ok=True)
            labels_file = config_dir / 'location_labels.yaml'
            
            labels_data = {
                'locations': self.location_labels,
                'last_updated': time.time(),
                'version': '1.0'
            }
            
            with open(labels_file, 'w') as f:
                yaml.dump(labels_data, f, default_flow_style=False, sort_keys=True)
            
            logger.info(f"Saved {len(self.location_labels)} location labels to {labels_file}")
        except Exception as e:
            logger.error(f"Failed to save location labels: {e}")
    
    def destroy_node(self):
        """Cleanup when destroying the node."""
        try:
            self.service_handler.cleanup()
            logger.info("Agent node destroyed successfully")
        except Exception as e:
            logger.error(f"Error during node cleanup: {e}")
        finally:
            super().destroy_node()

def main(args=None):
    """Main entry point for the agent node."""
    rclpy.init(args=args)
    
    try:
        node = AgentNode()
        
        # Use multi-threaded executor for better performance
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        logger.info("Starting ROS2 Agent Node...")
        executor.spin()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Node error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
