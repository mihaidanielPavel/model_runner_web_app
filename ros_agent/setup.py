from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros_agent'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'ros_agent', 'static'), glob('ros_agent/static/*')),
    ],
    install_requires=[
        'rclpy',
        'sensor_msgs',
        'nav_msgs',
        'geometry_msgs',
        'std_msgs',
        'requests',
        'pyyaml',
        'prompt_toolkit',
        'rich',
        'flask',
        'flask-socketio',
    ],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='ROS2 Agent for Mobile Robot Interaction',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_agent = ros_agent.cli_interface:main',
            'agent_node = ros_agent.agent_node:main',
            'web_dashboard = ros_agent.web_server:main',
        ],
    },
)
