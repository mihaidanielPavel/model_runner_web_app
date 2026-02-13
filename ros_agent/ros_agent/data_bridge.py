#!/usr/bin/env python3
"""
Data Bridge for ROS2 Agent

This module converts ROS2 sensor messages to human-readable text format
for the AI agent to process and understand.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import math
import time

# ROS2 imports
import rclpy
from sensor_msgs.msg import Image, LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from std_msgs.msg import Header

logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Container for processed sensor data."""
    sensor_type: str
    topic_name: str
    timestamp: float
    data_summary: str
    raw_data: Dict[str, Any]

class DataBridge:
    """Converts ROS2 sensor messages to human-readable text."""
    
    def __init__(self):
        """Initialize the data bridge."""
        self.last_update_times: Dict[str, float] = {}
        self.data_cache: Dict[str, SensorData] = {}
    
    def process_image_message(self, msg: Image, topic_name: str) -> SensorData:
        """
        Process camera image message.
        
        Args:
            msg: ROS2 Image message
            topic_name: Name of the topic
            
        Returns:
            Processed sensor data
        """
        timestamp = time.time()
        
        # Extract basic image information
        width = msg.width
        height = msg.height
        encoding = msg.encoding
        step = msg.step
        
        # Calculate approximate data size
        data_size = len(msg.data)
        
        summary = f"Camera Image: {width}x{height} pixels, {encoding} encoding, {data_size} bytes"
        
        raw_data = {
            'width': width,
            'height': height,
            'encoding': encoding,
            'step': step,
            'data_size': data_size,
            'header': self._process_header(msg.header)
        }
        
        return SensorData(
            sensor_type="camera",
            topic_name=topic_name,
            timestamp=timestamp,
            data_summary=summary,
            raw_data=raw_data
        )
    
    def process_laser_scan_message(self, msg: LaserScan, topic_name: str) -> SensorData:
        """
        Process LiDAR laser scan message.
        
        Args:
            msg: ROS2 LaserScan message
            topic_name: Name of the topic
            
        Returns:
            Processed sensor data
        """
        timestamp = time.time()
        
        # Calculate scan statistics
        ranges = [r for r in msg.ranges if not math.isnan(r) and r > 0]
        intensities = [i for i in msg.intensities if not math.isnan(i)]
        
        if ranges:
            min_range = min(ranges)
            max_range = max(ranges)
            avg_range = sum(ranges) / len(ranges)
            valid_points = len(ranges)
        else:
            min_range = max_range = avg_range = 0.0
            valid_points = 0
        
        # Calculate angular information
        angle_min = math.degrees(msg.angle_min)
        angle_max = math.degrees(msg.angle_max)
        angle_increment = math.degrees(msg.angle_increment)
        
        summary = f"LiDAR Scan: {valid_points} valid points, range {min_range:.2f}-{max_range:.2f}m (avg: {avg_range:.2f}m), angles {angle_min:.1f}° to {angle_max:.1f}°"
        
        raw_data = {
            'ranges': {
                'min': min_range,
                'max': max_range,
                'average': avg_range,
                'valid_count': valid_points,
                'total_count': len(msg.ranges)
            },
            'intensities': {
                'min': min(intensities) if intensities else 0,
                'max': max(intensities) if intensities else 0,
                'average': sum(intensities) / len(intensities) if intensities else 0
            },
            'angles': {
                'min_deg': angle_min,
                'max_deg': angle_max,
                'increment_deg': angle_increment
            },
            'header': self._process_header(msg.header)
        }
        
        return SensorData(
            sensor_type="lidar",
            topic_name=topic_name,
            timestamp=timestamp,
            data_summary=summary,
            raw_data=raw_data
        )
    
    def process_odometry_message(self, msg: Odometry, topic_name: str) -> SensorData:
        """
        Process odometry message.
        
        Args:
            msg: ROS2 Odometry message
            topic_name: Name of the topic
            
        Returns:
            Processed sensor data
        """
        timestamp = time.time()
        
        # Extract position and orientation
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        vel_linear = msg.twist.twist.linear
        vel_angular = msg.twist.twist.angular
        
        # Convert quaternion to euler angles (simplified)
        yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y), 
                        1 - 2 * (ori.y * ori.y + ori.z * ori.z))
        yaw_deg = math.degrees(yaw)
        
        # Calculate speed
        speed = math.sqrt(vel_linear.x**2 + vel_linear.y**2 + vel_linear.z**2)
        
        summary = f"Odometry: Position ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}), Yaw: {yaw_deg:.1f}°, Speed: {speed:.2f} m/s"
        
        raw_data = {
            'position': {
                'x': pos.x,
                'y': pos.y,
                'z': pos.z
            },
            'orientation': {
                'x': ori.x,
                'y': ori.y,
                'z': ori.z,
                'w': ori.w,
                'yaw_deg': yaw_deg
            },
            'velocity': {
                'linear': {
                    'x': vel_linear.x,
                    'y': vel_linear.y,
                    'z': vel_linear.z,
                    'speed': speed
                },
                'angular': {
                    'x': vel_angular.x,
                    'y': vel_angular.y,
                    'z': vel_angular.z
                }
            },
            'header': self._process_header(msg.header)
        }
        
        return SensorData(
            sensor_type="odometry",
            topic_name=topic_name,
            timestamp=timestamp,
            data_summary=summary,
            raw_data=raw_data
        )
    
    def process_imu_message(self, msg: Imu, topic_name: str) -> SensorData:
        """
        Process IMU message.
        
        Args:
            msg: ROS2 Imu message
            topic_name: Name of the topic
            
        Returns:
            Processed sensor data
        """
        timestamp = time.time()
        
        # Extract accelerometer, gyroscope, and magnetometer data
        accel = msg.linear_acceleration
        gyro = msg.angular_velocity
        mag = msg.orientation  # This is actually orientation, not magnetometer
        
        # Calculate magnitudes
        accel_mag = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        gyro_mag = math.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2)
        
        summary = f"IMU: Acceleration {accel_mag:.2f} m/s² ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}), Angular velocity {gyro_mag:.2f} rad/s"
        
        raw_data = {
            'linear_acceleration': {
                'x': accel.x,
                'y': accel.y,
                'z': accel.z,
                'magnitude': accel_mag
            },
            'angular_velocity': {
                'x': gyro.x,
                'y': gyro.y,
                'z': gyro.z,
                'magnitude': gyro_mag
            },
            'orientation': {
                'x': mag.x,
                'y': mag.y,
                'z': mag.z,
                'w': mag.w
            },
            'header': self._process_header(msg.header)
        }
        
        return SensorData(
            sensor_type="imu",
            topic_name=topic_name,
            timestamp=timestamp,
            data_summary=summary,
            raw_data=raw_data
        )
    
    def _process_header(self, header: Header) -> Dict[str, Any]:
        """Process ROS2 message header."""
        return {
            'frame_id': header.frame_id,
            'stamp': {
                'sec': header.stamp.sec,
                'nanosec': header.stamp.nanosec
            }
        }
    
    def update_sensor_data(self, sensor_data: SensorData) -> None:
        """Update cached sensor data."""
        self.data_cache[sensor_data.topic_name] = sensor_data
        self.last_update_times[sensor_data.topic_name] = sensor_data.timestamp
    
    def get_sensor_summary(self) -> str:
        """Get a compact summary of all current sensor data."""
        if not self.data_cache:
            return "No sensor data"
        
        summary_lines = []
        current_time = time.time()
        
        for topic_name, sensor_data in self.data_cache.items():
            age = current_time - sensor_data.timestamp
            age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"
            
            # Create compact summary based on sensor type
            if sensor_data.sensor_type == "camera":
                summary_lines.append(f"Camera({topic_name}): {age_str}")
            elif sensor_data.sensor_type == "lidar":
                ranges = sensor_data.raw_data.get('ranges', {})
                valid_count = ranges.get('valid_count', 0)
                avg_range = ranges.get('average', 0)
                summary_lines.append(f"LiDAR({topic_name}): {valid_count}pts, avg:{avg_range:.1f}m, {age_str}")
            elif sensor_data.sensor_type == "odometry":
                pos = sensor_data.raw_data.get('position', {})
                x, y = pos.get('x', 0), pos.get('y', 0)
                summary_lines.append(f"Odom({topic_name}): ({x:.2f},{y:.2f}), {age_str}")
            elif sensor_data.sensor_type == "imu":
                accel = sensor_data.raw_data.get('linear_acceleration', {})
                mag = accel.get('magnitude', 0)
                summary_lines.append(f"IMU({topic_name}): {mag:.1f}m/s², {age_str}")
            else:
                summary_lines.append(f"{sensor_data.sensor_type}({topic_name}): {age_str}")
        
        return "; ".join(summary_lines)
    
    def get_detailed_sensor_data(self, topic_name: str) -> Optional[str]:
        """Get detailed sensor data for a specific topic."""
        if topic_name not in self.data_cache:
            return None
        
        sensor_data = self.data_cache[topic_name]
        current_time = time.time()
        age = current_time - sensor_data.timestamp
        
        details = [
            f"Detailed Sensor Data for {topic_name}:",
            f"Type: {sensor_data.sensor_type}",
            f"Last Update: {age:.1f} seconds ago",
            f"Summary: {sensor_data.data_summary}",
            "",
            "Raw Data:"
        ]
        
        # Format raw data nicely
        for key, value in sensor_data.raw_data.items():
            if isinstance(value, dict):
                details.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    details.append(f"    {sub_key}: {sub_value}")
            else:
                details.append(f"  {key}: {value}")
        
        return "\n".join(details)
    
    def get_stale_sensors(self, max_age: float = 10.0) -> List[str]:
        """Get list of sensor topics that haven't been updated recently."""
        current_time = time.time()
        stale_sensors = []
        
        for topic_name, last_update in self.last_update_times.items():
            if current_time - last_update > max_age:
                stale_sensors.append(topic_name)
        
        return stale_sensors
    
    def clear_old_data(self, max_age: float = 60.0) -> int:
        """Clear sensor data older than max_age seconds."""
        current_time = time.time()
        removed_count = 0
        
        topics_to_remove = []
        for topic_name, last_update in self.last_update_times.items():
            if current_time - last_update > max_age:
                topics_to_remove.append(topic_name)
        
        for topic_name in topics_to_remove:
            del self.data_cache[topic_name]
            del self.last_update_times[topic_name]
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} stale sensor data entries")
        
        return removed_count

