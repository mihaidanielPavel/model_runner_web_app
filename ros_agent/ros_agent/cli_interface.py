#!/usr/bin/env python3
"""
CLI Interface for ROS2 Agent

Interactive command-line interface for interacting with the ROS2 agent.
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent_node import AgentNode

logger = logging.getLogger(__name__)

class AgentCLI:
    """Interactive CLI for the ROS2 Agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the CLI interface."""
        # Initialize commands first to ensure they're always available
        self.commands = {
            'help': self._cmd_help,
            'status': self._cmd_status,
            'sensors': self._cmd_sensors,
            'topics': self._cmd_topics,
            'services': self._cmd_services,
            'ask': self._cmd_ask,
            'call': self._cmd_call,
            'clear': self._cmd_clear,
            'config': self._cmd_config,
            'exit': self._cmd_exit,
            'quit': self._cmd_exit
        }
        
        self.console = Console()
        self.config_path = config_path
        
        try:
            # Initialize ROS2
            rclpy.init()
            
            # Initialize agent node
            self.agent_node = AgentNode(config_path)
            
            # Setup executor in separate thread
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.agent_node)
            self.executor_thread = threading.Thread(target=self._run_executor, daemon=True)
            self.executor_thread.start()
            
            # Setup CLI components
            self.session = PromptSession(
                history=FileHistory('.agent_history'),
                completer=self._create_completer()
            )
            
            # Wait for node to initialize
            time.sleep(1.0)
            
            self.console.print(Panel.fit(
                "[bold blue]🤖 ROS2 Agent CLI[/bold blue]\n"
                "Type 'help' for available commands\n"
                "Type 'exit' to quit",
                title="Welcome"
            ))
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI: {e}")
            self.console.print(f"[red]Failed to initialize CLI: {e}[/red]")
            self.console.print("[yellow]CLI will run in limited mode[/yellow]")
            
            # Set up minimal components for error recovery
            self.agent_node = None
            self.executor = None
            self.executor_thread = None
            self.session = None
    
    def _run_executor(self):
        """Run the ROS2 executor in a separate thread."""
        try:
            self.executor.spin()
        except Exception as e:
            logger.error(f"Executor error: {e}")
    
    def _create_completer(self) -> WordCompleter:
        """Create command completer for auto-completion."""
        commands = list(self.commands.keys())
        return WordCompleter(commands, ignore_case=True)
    
    def run(self):
        """Run the interactive CLI."""
        try:
            while True:
                try:
                    # Get user input
                    if self.session:
                        user_input = self.session.prompt(HTML('<ansiblue>🤖 Agent></ansiblue> '))
                    else:
                        # Fallback to simple input if session is not available
                        user_input = input('🤖 Agent> ')
                    
                    if not user_input.strip():
                        continue
                    
                    # Parse command
                    parts = user_input.strip().split()
                    command = parts[0].lower()
                    args = parts[1:] if len(parts) > 1 else []
                    
                    # Execute command
                    if command in self.commands:
                        self.commands[command](args)
                    else:
                        self.console.print(f"[red]Unknown command: {command}[/red]")
                        self.console.print("Type 'help' for available commands")
                
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                except EOFError:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
        
        finally:
            self._cleanup()
    
    def _cmd_help(self, args: List[str]):
        """Show help information."""
        help_text = """
[bold]Available Commands:[/bold]

[cyan]help[/cyan]                    - Show this help message
[cyan]status[/cyan]                  - Show agent status and statistics
[cyan]sensors[/cyan]                 - Show current sensor data summary
[cyan]topics[/cyan]                   - List available ROS2 topics
[cyan]services[/cyan]                 - List available ROS2 services
[cyan]ask <question>[/cyan]          - Ask the AI agent a question
[cyan]call <service> <params>[/cyan] - Call a ROS2 service
[cyan]clear[/cyan]                    - Clear conversation history
[cyan]config[/cyan]                  - Show current configuration
[cyan]exit/quit[/cyan]                - Exit the application

[bold]AI Agent Commands (Token-Efficient):[/bold]
[cyan]ask move to position (2.5, 3.0)[/cyan] - AI executes navigation
[cyan]ask move to storage[/cyan] - AI moves to named location
[cyan]ask go to home[/cyan] - AI moves to named location
[cyan]ask stop[/cyan]                - AI stops navigation
[cyan]ask where am I?[/cyan]          - AI gets current position
[cyan]ask analyze sensor data[/cyan]  - AI analyzes sensors intelligently
[cyan]ask what services are available[/cyan] - AI lists services
[cyan]ask help me navigate safely[/cyan] - AI assists with navigation
[cyan]ask add location storage at (-1, 0.5)[/cyan] - AI adds location label
[cyan]ask list all locations[/cyan] - AI shows saved locations

[bold]Direct Commands (No AI Processing):[/bold]
[cyan]call <service> <params>[/cyan] - Direct service call
[cyan]sensors[/cyan]                 - Direct sensor data display
[cyan]status[/cyan]                  - Direct status display

[bold]Token Optimization Features:[/bold]
- Context is dynamically selected based on question type
- Compact sensor summaries (e.g., "LiDAR: 360pts, avg:2.5m")
- Grouped service summaries (e.g., "costmap:2, map:1")
- Limited conversation history (last 5 messages)
- Response token limit (500 tokens)

[bold]Examples:[/bold]
  ask What is my current position?
  ask move to position (2.5, 3.0)
  ask move to storage
  ask go to home
  ask stop
  ask analyze the LiDAR data
  ask what sensors are working?
  ask add location kitchen at (1, 1)
  ask list all locations
  sensors
  status
        """
        self.console.print(Panel(help_text, title="Help"))
    
    def _cmd_status(self, args: List[str]):
        """Show agent status."""
        try:
            if not self.agent_node:
                self.console.print("[red]Agent node not available[/red]")
                return
                
            status = self.agent_node.get_status()
            
            # Create status table
            table = Table(title="Agent Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Node Name", status['node_name'])
            table.add_row("Active Subscribers", str(status['subscribers']))
            table.add_row("Sensor Data Count", str(status['sensor_data_count']))
            table.add_row("Services Discovered", str(status['services_discovered']))
            table.add_row("Gemma Client", "✅ Available" if status['gemma_client_available'] else "❌ Not Available")
            table.add_row("Auto-Discovery", "✅ Enabled" if status['discovery_enabled'] else "❌ Disabled")
            
            # Add navigation status if available
            if 'navigation_status' in status:
                nav_status = status['navigation_status']
                table.add_row("Navigation Status", nav_status['status'])
                if nav_status['current_pose']:
                    pose = nav_status['current_pose']
                    table.add_row("Current Position", f"({pose['x']:.3f}, {pose['y']:.3f})")
                    table.add_row("Position Frame", pose.get('frame_id', 'unknown'))
            
            # Add statistics info if available
            if hasattr(self.agent_node, 'stats_logger'):
                stats_summary = self.agent_node.stats_logger.get_stats_summary()
                table.add_row("Interactions Logged", str(stats_summary['interaction_count']))
                table.add_row("Log File", stats_summary['log_file'])
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting status: {e}[/red]")
    
    def _cmd_sensors(self, args: List[str]):
        """Show sensor data summary."""
        try:
            if not self.agent_node:
                self.console.print("[red]Agent node not available[/red]")
                return
                
            summary = self.agent_node.get_sensor_summary()
            self.console.print(Panel(summary, title="Sensor Data"))
        except Exception as e:
            self.console.print(f"[red]Error getting sensor data: {e}[/red]")
    
    def _cmd_topics(self, args: List[str]):
        """List available ROS2 topics."""
        try:
            if not self.agent_node:
                self.console.print("[red]Agent node not available[/red]")
                return
                
            topics = self.agent_node.get_topic_names_and_types()
            
            if not topics:
                self.console.print("[yellow]No topics found[/yellow]")
                return
            
            table = Table(title="Available Topics")
            table.add_column("Topic Name", style="cyan")
            table.add_column("Message Types", style="green")
            
            for topic_name, topic_types in sorted(topics):
                types_str = ", ".join(topic_types)
                table.add_row(topic_name, types_str)
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting topics: {e}[/red]")
    
    def _cmd_services(self, args: List[str]):
        """List available ROS2 services."""
        try:
            if not self.agent_node:
                self.console.print("[red]Agent node not available[/red]")
                return
                
            summary = self.agent_node.get_services_summary()
            self.console.print(Panel(summary, title="Available Services"))
        except Exception as e:
            self.console.print(f"[red]Error getting services: {e}[/red]")
    
    def _cmd_ask(self, args: List[str]):
        """Ask the AI agent a question."""
        if not args:
            self.console.print("[red]Please provide a question[/red]")
            self.console.print("Example: ask What is the robot's current position?")
            return
        
        if not self.agent_node:
            self.console.print("[red]Agent node not available[/red]")
            return
        
        question = " ".join(args)
        
        try:
            self.console.print(f"[cyan]Question:[/cyan] {question}")
            self.console.print("[yellow]Thinking...[/yellow]")
            
            response = self.agent_node.ask_agent(question)
            
            if response:
                self.console.print(Panel(response, title="Agent Response"))
            else:
                self.console.print("[red]No response from agent[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error asking agent: {e}[/red]")
    
    def _cmd_call(self, args: List[str]):
        """Call a ROS2 service."""
        if len(args) < 1:
            self.console.print("[red]Please provide service name[/red]")
            self.console.print("Example: call /reset_odometry {}")
            return
        
        if not self.agent_node:
            self.console.print("[red]Agent node not available[/red]")
            return
        
        service_name = args[0]
        request_data = {}
        
        # Parse request parameters if provided
        if len(args) > 1:
            try:
                request_str = " ".join(args[1:])
                request_data = json.loads(request_str)
            except json.JSONDecodeError:
                self.console.print("[red]Invalid JSON in request parameters[/red]")
                return
        
        try:
            self.console.print(f"[cyan]Calling service:[/cyan] {service_name}")
            if request_data:
                self.console.print(f"[cyan]Request data:[/cyan] {json.dumps(request_data, indent=2)}")
            
            response = self.agent_node.call_service(service_name, request_data)
            
            if response:
                self.console.print(Panel(
                    json.dumps(response, indent=2),
                    title=f"Service Response: {service_name}"
                ))
            else:
                self.console.print(f"[red]Service call failed: {service_name}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]Error calling service: {e}[/red]")
    
    def _cmd_clear(self, args: List[str]):
        """Clear conversation history."""
        try:
            if not self.agent_node:
                self.console.print("[red]Agent node not available[/red]")
                return
                
            if self.agent_node.gemma_client:
                self.agent_node.gemma_client.clear_conversation()
                self.console.print("[green]Conversation history cleared[/green]")
            else:
                self.console.print("[yellow]No conversation history to clear[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error clearing history: {e}[/red]")
    
    def _cmd_config(self, args: List[str]):
        """Show current configuration."""
        try:
            if not self.agent_node:
                self.console.print("[red]Agent node not available[/red]")
                return
                
            config_text = json.dumps(self.agent_node.config, indent=2)
            self.console.print(Panel(config_text, title="Current Configuration"))
        except Exception as e:
            self.console.print(f"[red]Error showing config: {e}[/red]")
    
    def _cmd_exit(self, args: List[str]):
        """Exit the application."""
        self.console.print("[yellow]Goodbye![/yellow]")
        raise SystemExit(0)
    
    def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.agent_node:
                self.agent_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main(args=None):
    """Main entry point for the CLI."""
    try:
        cli = AgentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"CLI error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

