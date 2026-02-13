#!/usr/bin/env python3
"""
Web Bridge for ROS2-Flask Communication

Handles communication between the ROS2 agent node and Flask web server.
Manages real-time data streaming and event forwarding.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

class WebBridge:
    """Bridge between ROS2 AgentNode and Flask web server."""
    
    def __init__(self, agent_node, socketio: SocketIO):
        """
        Initialize the web bridge.
        
        Args:
            agent_node: ROS2 AgentNode instance
            socketio: Flask-SocketIO instance
        """
        self.agent_node = agent_node
        self.socketio = socketio
        
        # Thread-safe data cache
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        
        # Update intervals (seconds)
        self.status_update_interval = 2.0
        self.sensor_update_interval = 1.0
        self.navigation_update_interval = 0.5
        
        # Last update times
        self.last_status_update = 0.0
        self.last_sensor_update = 0.0
        self.last_navigation_update = 0.0
        
        # Start update threads
        self._start_update_threads()
        
        logger.info("Web bridge initialized")
    
    def _start_update_threads(self):
        """Start background threads for data updates."""
        # Status update thread
        status_thread = threading.Thread(target=self._status_update_worker, daemon=True)
        status_thread.start()
        
        # Sensor update thread
        sensor_thread = threading.Thread(target=self._sensor_update_worker, daemon=True)
        sensor_thread.start()
        
        # Navigation update thread
        nav_thread = threading.Thread(target=self._navigation_update_worker, daemon=True)
        nav_thread.start()
        
        logger.info("Web bridge update threads started")
    
    def _status_update_worker(self):
        """Background worker for status updates."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_status_update >= self.status_update_interval:
                    self._update_status()
                    self.last_status_update = current_time
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Status update worker error: {e}")
                time.sleep(1.0)
    
    def _sensor_update_worker(self):
        """Background worker for sensor updates."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_sensor_update >= self.sensor_update_interval:
                    self._update_sensors()
                    self.last_sensor_update = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Sensor update worker error: {e}")
                time.sleep(1.0)
    
    def _navigation_update_worker(self):
        """Background worker for navigation updates."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_navigation_update >= self.navigation_update_interval:
                    self._update_navigation()
                    self.last_navigation_update = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Navigation update worker error: {e}")
                time.sleep(1.0)
    
    def _update_status(self):
        """Update robot status and broadcast to clients."""
        try:
            status = self.agent_node.get_status()
            
            with self.cache_lock:
                self.data_cache['status'] = status
            
            # Broadcast to all connected clients
            self.socketio.emit('status_update', status, room='dashboard')
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def _update_sensors(self):
        """Update sensor data and broadcast to clients."""
        try:
            sensor_summary = self.agent_node.get_sensor_summary()
            available_topics = self.agent_node.get_available_topics()
            
            sensor_data = {
                'summary': sensor_summary,
                'topics': available_topics,
                'timestamp': time.time()
            }
            
            with self.cache_lock:
                self.data_cache['sensors'] = sensor_data
            
            # Broadcast to all connected clients
            self.socketio.emit('sensor_update', sensor_data, room='dashboard')
            
        except Exception as e:
            logger.error(f"Error updating sensors: {e}")
    
    def _update_navigation(self):
        """Update navigation status and broadcast to clients."""
        try:
            nav_status = self.agent_node.get_navigation_status()
            
            with self.cache_lock:
                self.data_cache['navigation'] = nav_status
            
            # Broadcast to all connected clients
            self.socketio.emit('navigation_update', nav_status, room='dashboard')
            
        except Exception as e:
            logger.error(f"Error updating navigation: {e}")
    
    def broadcast_log_entry(self, log_data: Dict[str, Any]):
        """Broadcast a new log entry to clients."""
        try:
            self.socketio.emit('log_entry', log_data, room='dashboard')
        except Exception as e:
            logger.error(f"Error broadcasting log entry: {e}")
    
    def broadcast_location_update(self, locations: Dict[str, Any]):
        """Broadcast location labels update to clients."""
        try:
            self.socketio.emit('location_update', {'locations': locations}, room='dashboard')
        except Exception as e:
            logger.error(f"Error broadcasting location update: {e}")
    
    def get_cached_data(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get cached data by type."""
        with self.cache_lock:
            return self.data_cache.get(data_type)
    
    def process_agent_query(self, question: str) -> Dict[str, Any]:
        """
        Process an agent query and return response with metadata.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            
            if not self.agent_node.gemma_client:
                return {
                    'response': 'AI agent not available (Gemma client not initialized)',
                    'success': False,
                    'error': 'Gemma client not initialized',
                    'response_time': time.time() - start_time
                }
            
            # Get current sensor context
            sensor_context = self.agent_node.get_sensor_summary()
            
            # Get navigation status for context
            nav_status = self.agent_node.get_navigation_status()
            nav_context = f"Navigation status: {nav_status['status']}"
            if nav_status['current_pose']:
                pose = nav_status['current_pose']
                nav_context += f", Current position (map frame): ({pose['x']:.3f}, {pose['y']:.3f})"
                nav_context += f", Frame: {pose['frame_id']}"
            
            # Add available topics context
            available_topics = self.agent_node.get_available_topics()
            topics_context = f"Available topics with latest data: {', '.join(available_topics)}"
            
            # Add available services context
            services_summary = self.agent_node.get_services_summary()
            
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
                location_labels_info = self.agent_node.list_location_labels()
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
            response_data = self.agent_node.gemma_client.generate_response(question, full_context)
            processing_time = time.time() - agent_start_time
            
            if response_data:
                response_text = response_data.get('response', 'No response')
                metadata = response_data.get('metadata', {})
                
                # Check if the response contains function calls and execute them
                final_response = self.agent_node._process_agent_response(response_text, question)
                
                response_time = time.time() - start_time
                
                # Extract token and timing data from metadata
                usage_data = metadata.get('usage', {})
                timings_data = metadata.get('timings', {})
                metrics_data = metadata.get('metrics', {})
                
                # Log statistics with enhanced metadata
                self.agent_node.stats_logger.log_agent_interaction(
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
                
                # Broadcast log entry to web clients
                log_entry = {
                    'timestamp': time.time(),
                    'question': question,
                    'response': final_response or "No response",
                    'response_time': response_time,
                    'processing_time': processing_time,
                    'tokens_input': usage_data.get('prompt_tokens'),
                    'tokens_output': usage_data.get('completion_tokens'),
                    'total_tokens': usage_data.get('total_tokens'),
                    'prompt_ms': timings_data.get('prompt_ms'),
                    'predicted_ms': timings_data.get('predicted_ms'),
                    'tokens_per_second': metrics_data.get('overall_tokens_per_second'),
                    'efficiency_score': metrics_data.get('efficiency_score'),
                    'success': final_response is not None
                }
                self.broadcast_log_entry(log_entry)
                
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
                self.agent_node.stats_logger.log_agent_interaction(
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
            logger.error(f"Agent query processing failed: {e}")
            response_time = time.time() - start_time
            
            # Log error statistics
            self.agent_node.stats_logger.log_agent_interaction(
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
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Web bridge cleanup completed")