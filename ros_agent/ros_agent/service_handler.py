#!/usr/bin/env python3
"""
Service Handler for ROS2 Agent

Handles ROS2 service discovery and calling functionality.
"""

import logging
from typing import Dict, List, Optional, Any
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.client import Client
from rclpy.qos import QoSProfile

logger = logging.getLogger(__name__)

class ServiceHandler:
    """Handles ROS2 service discovery and calling."""
    
    def __init__(self, node: Node):
        """Initialize service handler with ROS2 node."""
        self.node = node
        self.discovered_services: Dict[str, Dict[str, Any]] = {}
        self.service_clients: Dict[str, Client] = {}
        self.last_discovery_time = 0.0
    
    def discover_services(self) -> Dict[str, Dict[str, Any]]:
        """Discover available ROS2 services."""
        try:
            service_names_and_types = self.node.get_service_names_and_types()
            current_time = time.time()
            
            services_info = {}
            for service_name, service_types in service_names_and_types:
                # Skip internal ROS2 services
                if service_name.startswith('/ros_') or service_name.startswith('/_ros_'):
                    continue
                
                services_info[service_name] = {
                    'name': service_name,
                    'types': service_types,
                    'discovered_at': current_time
                }
            
            self.discovered_services = services_info
            self.last_discovery_time = current_time
            
            logger.info(f"Discovered {len(services_info)} services")
            return services_info
            
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return {}
    
    def get_service_list(self) -> List[str]:
        """Get list of available service names."""
        if not self.discovered_services or time.time() - self.last_discovery_time > 5.0:
            self.discover_services()
        
        return list(self.discovered_services.keys())
    
    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific service."""
        if not self.discovered_services or time.time() - self.last_discovery_time > 5.0:
            self.discover_services()
        
        return self.discovered_services.get(service_name)
    
    def call_service(self, service_name: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Call a ROS2 service with the given request data.
        
        Args:
            service_name: Name of the service to call
            request_data: Request parameters as dictionary
            
        Returns:
            Service response or None if failed
        """
        try:
            service_info = self.get_service_info(service_name)
            if not service_info:
                logger.error(f"Service {service_name} not found")
                return None
            
            # Get service type
            service_types = service_info['types']
            if not service_types:
                logger.error(f"No service types found for {service_name}")
                return None
            
            # Use the first available service type
            service_type = service_types[0]
            
            # Create service client if not exists
            if service_name not in self.service_clients:
                try:
                    # Import the service type dynamically
                    service_class = self._import_service_type(service_type)
                    if not service_class:
                        logger.error(f"Could not import service type: {service_type}")
                        return None
                    
                    client = self.node.create_client(service_class, service_name)
                    self.service_clients[service_name] = client
                    
                    # Wait for service to be available
                    if not client.wait_for_service(timeout_sec=5.0):
                        logger.error(f"Service {service_name} not available")
                        return None
                        
                except Exception as e:
                    logger.error(f"Failed to create client for {service_name}: {e}")
                    return None
            
            client = self.service_clients[service_name]
            
            # Create request object
            request = client.srv_type.Request()
            self._populate_request(request, request_data)
            
            # Call service
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
            
            if future.done():
                response = future.result()
                if response:
                    return self._extract_response_data(response)
                else:
                    logger.error(f"Service call failed for {service_name}")
                    return None
            else:
                logger.error(f"Service call timeout for {service_name}")
                return None
                
        except Exception as e:
            logger.error(f"Service call error for {service_name}: {e}")
            return None
    
    def get_service_definition(self, service_name: str) -> Optional[str]:
        """
        Get service definition/interface information for agent to compose requests.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service definition string or None if not found
        """
        try:
            service_info = self.get_service_info(service_name)
            if not service_info:
                return None
            
            service_types = service_info['types']
            if not service_types:
                return None
            
            service_type = service_types[0]
            
            # Get service class to inspect its structure
            service_class = self._import_service_type(service_type)
            if not service_class:
                return f"Service {service_name}: Type {service_type} (import failed)"
            
            # Create a request instance to inspect its fields
            request = service_class.Request()
            
            definition = f"Service: {service_name}\n"
            definition += f"Type: {service_type}\n"
            definition += "Request fields:\n"
            
            # Inspect request fields
            for attr_name in dir(request):
                if not attr_name.startswith('_') and not callable(getattr(request, attr_name)):
                    attr_value = getattr(request, attr_name)
                    definition += f"  - {attr_name}: {type(attr_value).__name__}\n"
            
            return definition
            
        except Exception as e:
            logger.error(f"Failed to get service definition for {service_name}: {e}")
            return None
    
    def compose_service_request(self, service_name: str, natural_language_request: str) -> Optional[Dict[str, Any]]:
        """
        Compose service request from natural language description.
        
        Args:
            service_name: Name of the service to call
            natural_language_request: Natural language description of what to do
            
        Returns:
            Composed request data or None if failed
        """
        try:
            service_info = self.get_service_info(service_name)
            if not service_info:
                return None
            
            service_types = service_info['types']
            if not service_types:
                return None
            
            service_type = service_types[0]
            
            # Handle common navigation2 services
            if service_name.endswith('clear_entirely_local_costmap') or service_name.endswith('clear_entirely_global_costmap'):
                # Clear costmap services typically don't need parameters
                return {}
            
            elif service_name.endswith('get_costmap'):
                # Get costmap service might need layer names
                return {}
            
            elif service_name.endswith('load_map'):
                # Load map service needs map file path
                if 'map' in natural_language_request.lower():
                    # Try to extract map file path from request
                    import re
                    map_pattern = r'([a-zA-Z0-9_/.-]+\.yaml)'
                    match = re.search(map_pattern, natural_language_request)
                    if match:
                        return {'map_url': match.group(1)}
                return {}
            
            elif service_name.endswith('save_map'):
                # Save map service needs file path
                if 'save' in natural_language_request.lower():
                    import re
                    map_pattern = r'([a-zA-Z0-9_/.-]+\.yaml)'
                    match = re.search(map_pattern, natural_language_request)
                    if match:
                        return {'map_url': match.group(1)}
                return {}
            
            # For other services, return empty dict (agent can fill in specific fields)
            return {}
            
        except Exception as e:
            logger.error(f"Failed to compose service request for {service_name}: {e}")
            return None
    
    def _import_service_type(self, service_type: str):
        """Import service type dynamically."""
        try:
            # Parse service type (e.g., "std_srvs/srv/Empty")
            parts = service_type.split('/')
            if len(parts) != 3:
                return None
            
            package_name, module_name, service_name = parts
            
            # Common service imports
            if package_name == 'std_srvs':
                from std_srvs.srv import Empty, SetBool, Trigger
                if service_name == 'Empty':
                    return Empty
                elif service_name == 'SetBool':
                    return SetBool
                elif service_name == 'Trigger':
                    return Trigger
            
            # Navigation2 service imports
            elif package_name == 'nav2_msgs':
                from nav2_msgs.srv import ClearEntireCostmap, GetCostmap, LoadMap, SaveMap
                if service_name == 'ClearEntireCostmap':
                    return ClearEntireCostmap
                elif service_name == 'GetCostmap':
                    return GetCostmap
                elif service_name == 'LoadMap':
                    return LoadMap
                elif service_name == 'SaveMap':
                    return SaveMap
            
            # Add more service types as needed
            logger.warning(f"Unknown service type: {service_type}")
            return None
            
        except ImportError as e:
            logger.error(f"Failed to import service type {service_type}: {e}")
            return None
    
    def _populate_request(self, request, data: Dict[str, Any]):
        """Populate request object with data."""
        for key, value in data.items():
            if hasattr(request, key):
                setattr(request, key, value)
    
    def _extract_response_data(self, response) -> Dict[str, Any]:
        """Extract data from response object."""
        data = {}
        for attr_name in dir(response):
            if not attr_name.startswith('_'):
                attr_value = getattr(response, attr_name)
                if not callable(attr_value):
                    data[attr_name] = str(attr_value)
        return data
    
    def get_services_summary(self) -> str:
        """Get a compact summary of available services."""
        services = self.get_service_list()
        
        if not services:
            return "No services"
        
        # Group services by type for compactness
        service_groups = {}
        for service_name in services:
            if 'costmap' in service_name:
                service_groups['costmap'] = service_groups.get('costmap', 0) + 1
            elif 'map' in service_name:
                service_groups['map'] = service_groups.get('map', 0) + 1
            elif 'navigation' in service_name:
                service_groups['nav'] = service_groups.get('nav', 0) + 1
            else:
                service_groups['other'] = service_groups.get('other', 0) + 1
        
        summary_parts = []
        for group, count in service_groups.items():
            summary_parts.append(f"{group}:{count}")
        
        return f"Services({len(services)}): {', '.join(summary_parts)}"
    
    def cleanup(self):
        """Cleanup service clients."""
        for client in self.service_clients.values():
            try:
                client.destroy()
            except:
                pass
        self.service_clients.clear()

