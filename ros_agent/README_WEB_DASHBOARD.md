# ROS2 Agent Web Dashboard

A modern web application for monitoring and interacting with the ROS2 agent system in real-time.

## Features

### 🎛️ Real-time Dashboard
- Live robot status and sensor data
- Navigation status with position tracking
- Performance metrics and system health
- Quick action buttons for common tasks

### 📊 Enhanced Logging
- Real-time interaction logs with token usage metrics
- Detailed timing information (prompt_ms, predicted_ms, tokens/second)
- Filtering and search capabilities
- CSV export functionality
- Cost estimation for AI interactions

### 💬 Agent Chat Interface
- Natural language interaction with the AI agent
- Real-time responses with metadata display
- Conversation history
- Function execution results

### 📍 Location Management
- Save labeled positions (e.g., "storage", "home", "kitchen")
- Persistent storage in YAML format
- Quick navigation to saved locations
- Visual location management interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build the ROS2 package:
```bash
colcon build --packages-select ros_agent
source install/setup.bash
```

## Usage

### Launch the Web Dashboard

```bash
ros2 launch ros_agent web_dashboard.launch.py
```

Or with custom parameters:
```bash
ros2 launch ros_agent web_dashboard.launch.py web_port:=8080 web_host:=localhost
```

### Access the Dashboard

Open your web browser and navigate to:
- Default: `http://localhost:5000`
- Custom port: `http://localhost:YOUR_PORT`

## Configuration

Edit `config/config.yaml` to customize:

### Web Server Settings
```yaml
web_server:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 5000       # Web server port
  debug: false     # Enable debug mode
```

### AI Agent Settings
```yaml
gemma:
  api_url: "http://model-runner.docker.internal"
  model_name: "ai/gemma3"
  timeout: 30
  max_retries: 3
```

## API Endpoints

### REST API
- `GET /api/status` - Robot status
- `GET /api/sensors` - Sensor data
- `GET /api/navigation` - Navigation status
- `GET /api/locations` - Location labels
- `POST /api/locations` - Add location
- `DELETE /api/locations/<name>` - Remove location
- `GET /api/logs` - Interaction logs
- `POST /api/query` - Send agent query

### WebSocket Events

**Client → Server:**
- `agent_query` - Send question to AI agent
- `navigation_command` - Navigation commands
- `add_location` - Add labeled position
- `remove_location` - Delete labeled position

**Server → Client:**
- `status_update` - Robot status changes
- `log_entry` - New interaction logged
- `sensor_update` - Sensor data updates
- `navigation_update` - Navigation status changes
- `location_update` - Location labels changed

## Token Usage Tracking

The dashboard now extracts detailed metrics from AI API responses:

### Usage Metrics
- `prompt_tokens` - Input tokens
- `completion_tokens` - Output tokens  
- `total_tokens` - Total tokens used

### Timing Metrics
- `prompt_ms` - Prompt processing time
- `predicted_ms` - Response generation time
- `tokens_per_second` - Generation speed

### Derived Metrics
- `efficiency_score` - Tokens per second
- `cost_estimate` - Estimated cost in USD

## Location Labels Persistence

Location labels are automatically saved to `config/location_labels.yaml`:

```yaml
locations:
  storage:
    x: 1.5
    y: 2.0
    yaw: 0.0
  home:
    x: 0.0
    y: 0.0
    yaw: 1.57
last_updated: 1703123456.789
version: "1.0"
```

## Development

### Project Structure
```
ros_agent/
├── ros_agent/
│   ├── web_server.py      # Flask web server
│   ├── web_bridge.py      # ROS2-Flask communication
│   ├── agent_node.py      # Main ROS2 node (enhanced)
│   ├── gemma_client.py    # AI client (enhanced)
│   ├── statistics_logger.py # Logging (enhanced)
│   └── static/            # Web frontend
│       ├── index.html
│       ├── style.css
│       └── app.js
├── config/
│   ├── config.yaml        # Main configuration
│   └── location_labels.yaml # Auto-generated
├── launch/
│   └── web_dashboard.launch.py
└── requirements.txt       # Dependencies
```

### Key Components

1. **WebServer** (`web_server.py`) - Flask app with SocketIO
2. **WebBridge** (`web_bridge.py`) - Real-time data streaming
3. **AgentNode** (`agent_node.py`) - Enhanced with metadata support
4. **Frontend** (`static/`) - Modern responsive web interface

## Troubleshooting

### ROS2 Parameter Parsing Error
If you get an error like "Cannot have a value before ros__parameters", this means ROS2 is trying to parse the wrong config file. The solution is to use the launch file which handles this automatically:

```bash
ros2 launch ros_agent web_dashboard.launch.py
```

The launch file uses the correct configuration files:
- `config/config.yaml` - General configuration (used by web server)
- `config/ros_agent_params.yaml` - ROS2 parameters (used by agent node)

### WebSocket Connection Issues
- Check firewall settings
- Verify port is not blocked
- Ensure ROS2 agent is running

### AI Agent Not Responding
- Verify Gemma API is accessible
- Check configuration in `config.yaml`
- Review logs for connection errors

### Location Labels Not Saving
- Check write permissions on `config/` directory
- Verify YAML syntax in configuration files

## License

MIT License - see LICENSE file for details.
