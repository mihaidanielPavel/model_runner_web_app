#!/usr/bin/env python3
"""
Flask Web Server for ROS2 Agent Dashboard

Provides REST API endpoints and WebSocket communication for the web dashboard.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import yaml
from pathlib import Path

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# Local imports
from .agent_node import AgentNode
from .web_bridge import WebBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebServer:
    """Flask web server with SocketIO for ROS2 agent dashboard."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the web server."""
        # Find the correct static folder path
        static_folder = self._find_static_folder()
        
        self.app = Flask(__name__, 
                        static_folder=static_folder,
                        template_folder=None)  # We don't use templates
        
        # Log Flask configuration
        logger.info(f"Flask static folder: {self.app.static_folder}")
        logger.info(f"Flask static URL path: {self.app.static_url_path}")
        
        # Configure Flask
        self.app.config['SECRET_KEY'] = 'ros_agent_secret_key'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize ROS2 components
        self.agent_node = None
        self.web_bridge = None
        self.executor = None
        
        # Web server settings
        self.host = self.config.get('web_server', {}).get('host', '0.0.0.0')
        self.port = self.config.get('web_server', {}).get('port', 5000)
        self.debug = self.config.get('web_server', {}).get('debug', False)
        
        # Setup routes and socket events
        self._setup_routes()
        self._setup_socket_events()
        
        logger.info(f"Web server initialized on {self.host}:{self.port}")
    
    def _find_static_folder(self) -> str:
        """Find the correct static folder path."""
        # Try multiple possible locations for static folder
        possible_paths = [
            Path(__file__).parent / 'static',  # Development
            Path('/opt/ros/humble/share/ros_agent/ros_agent/static'),  # ROS2 install
            Path('/root/ros_ws/install/ros_agent/share/ros_agent/ros_agent/static'),  # Colcon install
            Path('/usr/local/share/ros_agent/ros_agent/static'),  # System install
        ]
        
        for path in possible_paths:
            if path.exists() and (path / 'index.html').exists():
                logger.info(f"Found static folder at: {path}")
                # List files in the static folder for debugging
                files = list(path.glob('*'))
                logger.info(f"Static files found: {[f.name for f in files]}")
                return str(path)
        
        # Fallback to relative path
        logger.warning("Static folder not found in any standard location, using relative path")
        logger.warning(f"Searched paths: {[str(p) for p in possible_paths]}")
        return 'static'
    
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
            'web_server': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            },
            'gemma': {
                'api_url': 'http://model-runner.docker.internal',
                'model_name': 'ai/gemma3',
                'timeout': 30,
                'max_retries': 3
            }
        }
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main dashboard page."""
            try:
                return self.app.send_static_file('index.html')
            except Exception as e:
                logger.error(f"Error serving index.html: {e}")
                # Return a simple HTML page as fallback
                return '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>ROS2 Agent Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .error { color: red; }
                        .info { color: blue; }
                    </style>
                </head>
                <body>
                    <h1>ROS2 Agent Dashboard</h1>
                    <p class="error">Error: Static files not found</p>
                    <p class="info">Debug info: <a href="/debug">/debug</a></p>
                    <p>API endpoints available:</p>
                    <ul>
                        <li><a href="/api/status">/api/status</a></li>
                        <li><a href="/api/sensors">/api/sensors</a></li>
                        <li><a href="/api/navigation">/api/navigation</a></li>
                        <li><a href="/api/locations">/api/locations</a></li>
                        <li><a href="/api/logs">/api/logs</a></li>
                    </ul>
                </body>
                </html>
                ''', 200
        
        @self.app.route('/debug')
        def debug():
            """Debug endpoint to check static file paths."""
            static_folder = self.app.static_folder
            index_path = Path(static_folder) / 'index.html' if static_folder else None
            app_js_path = Path(static_folder) / 'app.js' if static_folder else None
            style_css_path = Path(static_folder) / 'style.css' if static_folder else None
            
            debug_info = {
                'static_folder': static_folder,
                'index_exists': index_path.exists() if index_path else False,
                'index_path': str(index_path) if index_path else None,
                'app_js_exists': app_js_path.exists() if app_js_path else False,
                'app_js_path': str(app_js_path) if app_js_path else None,
                'style_css_exists': style_css_path.exists() if style_css_path else False,
                'style_css_path': str(style_css_path) if style_css_path else None,
                'current_dir': str(Path.cwd()),
                'script_dir': str(Path(__file__).parent),
                'static_files': list(Path(static_folder).glob('*')) if static_folder and Path(static_folder).exists() else [],
            }
            
            return jsonify(debug_info)
        
        @self.app.route('/<path:filename>')
        def static_files(filename):
            """Serve static files explicitly."""
            try:
                return self.app.send_static_file(filename)
            except Exception as e:
                logger.error(f"Error serving static file {filename}: {e}")
                return f"Static file not found: {filename}", 404
        
        @self.app.route('/api/status')
        def get_status():
            """Get current robot status."""
            try:
                if self.agent_node:
                    status = self.agent_node.get_status()
                    return jsonify(status)
                else:
                    return jsonify({'error': 'Agent node not available'}), 503
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/sensors')
        def get_sensors():
            """Get current sensor data."""
            try:
                if self.agent_node:
                    sensor_summary = self.agent_node.get_sensor_summary()
                    available_topics = self.agent_node.get_available_topics()
                    return jsonify({
                        'summary': sensor_summary,
                        'topics': available_topics
                    })
                else:
                    return jsonify({'error': 'Agent node not available'}), 503
            except Exception as e:
                logger.error(f"Error getting sensors: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/navigation')
        def get_navigation():
            """Get navigation status."""
            try:
                if self.agent_node:
                    nav_status = self.agent_node.get_navigation_status()
                    return jsonify(nav_status)
                else:
                    return jsonify({'error': 'Agent node not available'}), 503
            except Exception as e:
                logger.error(f"Error getting navigation: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/locations', methods=['GET'])
        def get_locations():
            """Get all location labels."""
            try:
                if self.agent_node:
                    locations = self.agent_node.location_labels
                    return jsonify({'locations': locations})
                else:
                    return jsonify({'error': 'Agent node not available'}), 503
            except Exception as e:
                logger.error(f"Error getting locations: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/locations', methods=['POST'])
        def add_location():
            """Add a new location label."""
            try:
                if not self.agent_node:
                    return jsonify({'error': 'Agent node not available'}), 503
                
                data = request.get_json()
                label = data.get('label')
                x = data.get('x')
                y = data.get('y')
                yaw = data.get('yaw', 0.0)
                
                if not all([label, x is not None, y is not None]):
                    return jsonify({'error': 'Missing required fields: label, x, y'}), 400
                
                result = self.agent_node.add_location_label(label, x, y, yaw)
                return jsonify({'message': result})
            except Exception as e:
                logger.error(f"Error adding location: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/locations/<label>', methods=['DELETE'])
        def remove_location(label):
            """Remove a location label."""
            try:
                if not self.agent_node:
                    return jsonify({'error': 'Agent node not available'}), 503
                
                result = self.agent_node.remove_location_label(label)
                return jsonify({'message': result})
            except Exception as e:
                logger.error(f"Error removing location: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/logs')
        def get_logs():
            """Get recent interaction logs."""
            try:
                if self.agent_node and self.agent_node.stats_logger:
                    # Read recent logs from CSV file
                    log_file = self.agent_node.stats_logger.csv_file
                    logs = self._read_recent_logs(log_file)
                    return jsonify({'logs': logs})
                else:
                    return jsonify({'logs': []})
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/query', methods=['POST'])
        def query_agent():
            """Send a query to the agent."""
            try:
                if not self.agent_node:
                    return jsonify({'error': 'Agent node not available'}), 503
                
                data = request.get_json()
                question = data.get('question')
                
                if not question:
                    return jsonify({'error': 'Missing question'}), 400
                
                # Process the query and get response with metadata
                response_data = self.agent_node.ask_agent_with_metadata(question)
                
                return jsonify(response_data)
            except Exception as e:
                logger.error(f"Error querying agent: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socket_events(self):
        """Setup SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info(f"Client connected: {request.sid}")
            join_room('dashboard')
            
            # Send initial data
            if self.agent_node:
                emit('status_update', self.agent_node.get_status())
                emit('navigation_update', self.agent_node.get_navigation_status())
                emit('location_update', {'locations': self.agent_node.location_labels})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info(f"Client disconnected: {request.sid}")
            leave_room('dashboard')
        
        @self.socketio.on('agent_query')
        def handle_agent_query(data):
            """Handle agent query from client."""
            try:
                question = data.get('question')
                if not question:
                    emit('error', {'message': 'No question provided'})
                    return
                
                if not self.agent_node:
                    emit('error', {'message': 'Agent node not available'})
                    return
                
                # Process query and emit response
                response_data = self.agent_node.ask_agent_with_metadata(question)
                emit('agent_response', response_data)
                
            except Exception as e:
                logger.error(f"Error handling agent query: {e}")
                emit('error', {'message': str(e)})
        
        @self.socketio.on('navigation_command')
        def handle_navigation_command(data):
            """Handle navigation command from client."""
            try:
                command = data.get('command')
                if not command:
                    emit('error', {'message': 'No command provided'})
                    return
                
                if not self.agent_node:
                    emit('error', {'message': 'Agent node not available'})
                    return
                
                # Process navigation command
                if command == 'stop':
                    result = self.agent_node.stop_navigation()
                elif command == 'status':
                    result = self.agent_node.get_navigation_status()
                else:
                    emit('error', {'message': 'Unknown navigation command'})
                    return
                
                emit('navigation_result', {'command': command, 'result': result})
                
            except Exception as e:
                logger.error(f"Error handling navigation command: {e}")
                emit('error', {'message': str(e)})
    
    def _read_recent_logs(self, log_file: Path, max_logs: int = 100) -> List[Dict[str, Any]]:
        """Read recent logs from CSV file."""
        try:
            import csv
            logs = []
            
            if not log_file.exists():
                return logs
            
            with open(log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Get the most recent logs
                recent_rows = rows[-max_logs:] if len(rows) > max_logs else rows
                
                for row in recent_rows:
                    # Convert timestamp to readable format
                    try:
                        timestamp = float(row.get('timestamp', 0))
                        row['timestamp_readable'] = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                               time.localtime(timestamp))
                    except:
                        row['timestamp_readable'] = 'Unknown'
                    
                    logs.append(row)
            
            return logs
            
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return []
    
    def initialize_ros2(self):
        """Initialize ROS2 components."""
        try:
            # Initialize ROS2
            if not rclpy.ok():
                rclpy.init()
            
            # Create agent node
            self.agent_node = AgentNode()
            
            # Create web bridge
            self.web_bridge = WebBridge(self.agent_node, self.socketio)
            
            # Create executor
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.agent_node)
            
            logger.info("ROS2 components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ROS2: {e}")
            return False
    
    def start_ros2_thread(self):
        """Start ROS2 executor in a separate thread."""
        def ros2_worker():
            try:
                logger.info("Starting ROS2 executor thread")
                self.executor.spin()
            except Exception as e:
                logger.error(f"ROS2 executor error: {e}")
        
        thread = threading.Thread(target=ros2_worker, daemon=True)
        thread.start()
        return thread
    
    def run(self):
        """Run the web server."""
        try:
            # Initialize ROS2
            if not self.initialize_ros2():
                logger.error("Failed to initialize ROS2. Exiting.")
                return
            
            # Start ROS2 thread
            ros2_thread = self.start_ros2_thread()
            
            # Start web server
            logger.info(f"Starting web server on {self.host}:{self.port}")
            self.socketio.run(self.app, 
                            host=self.host, 
                            port=self.port, 
                            debug=self.debug,
                            use_reloader=False)  # Disable reloader to avoid ROS2 conflicts
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        except Exception as e:
            logger.error(f"Web server error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.agent_node:
                self.agent_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            logger.info("Web server cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point for the web server."""
    server = WebServer()
    server.run()

if __name__ == '__main__':
    main()