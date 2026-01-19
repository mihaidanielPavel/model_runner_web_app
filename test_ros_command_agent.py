"""
ROS2 Command Agent Testing Script

This script tests language models' ability to act as natural language interfaces
for ROS2 robot control. The agent should be able to:
1. Understand natural language queries about the robot
2. Generate appropriate ROS2 commands (topic echo, service calls, etc.)
3. Interpret ROS2 responses and provide human-friendly answers

Example workflow:
    User: "Unde este robotul?"
    Agent: Generates → "ros2 topic echo /robot_pose -n 1"
    System: Returns → "x: 10.5, y: 8.2, theta: 1.57"
    Agent: Interprets → "Robotul este la poziția x=10.5, y=8.2, orientat către nord, lângă stația de alimentare."

Usage:
    python test_ros_command_agent.py --models all
    python test_ros_command_agent.py --models ai/qwen3 --output command_agent_results.csv
"""

import requests
import time
import csv
import json
import argparse
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

# Configuration
MODEL_RUNNER_URL = os.getenv('MODEL_RUNNER_URL', 'http://model-runner.docker.internal') #http://localhost:12434/
DEFAULT_OUTPUT = f'ros_command_agent_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# Available models
ALL_MODELS = [
    'ai/smollm2',
    'ai/llama3.2',
    'ai/gemma3',
    'ai/gemma3-qat'
]

# ROS2 Command Agent System Prompt
ROS_COMMAND_AGENT_PROMPT = """You are an intelligent ROS2 command interface agent for a mobile robot platform. Your role is to:

1. **Understand natural language queries** from users about the robot (position, status, battery, sensors, etc.)
2. **Generate exact ROS2 commands** to retrieve the requested information
3. **Interpret ROS2 output** and provide clear, human-friendly responses

Available ROS2 topics and services on the robot:
- /robot_pose (geometry_msgs/Pose2D) - Robot position and orientation
- /battery_status (sensor_msgs/BatteryState) - Battery level and charging status
- /cmd_vel (geometry_msgs/Twist) - Velocity commands
- /scan (sensor_msgs/LaserScan) - LIDAR data
- /odom (nav_msgs/Odometry) - Odometry information
- /robot_status (std_msgs/String) - Robot operational status
- /get_location (std_srvs/Trigger) - Service to get current location
- /goto_position (geometry_msgs/Point) - Service to navigate to a position

**Important**: 
- For queries, generate ONLY the ROS2 command (e.g., "ros2 topic echo /robot_pose -n 1")
- When interpreting data, provide natural language responses in English
- Be precise with command syntax
- Consider the context (e.g., "near the charging station" if x≈10, y≈10)

Respond in this format:
COMMAND: <ros2 command>
EXPECTED: <what data you expect>

Or when interpreting results:
INTERPRETATION: <human-friendly response in English>
"""

# Test scenarios for ROS2 command generation and interpretation
TEST_SCENARIOS = {
    'position_queries': [
        {
            'query': 'Where is the robot?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/robot_pose', '-n'],
            'expected_topics': ['pose', 'position', 'location'],
            'mock_response': 'x: 10.5\ny: 8.2\ntheta: 1.57',
            'expected_interpretation_parts': ['position', 'x', 'y', '10', '8', 'robot']
        },
        {
            'query': 'What is the current location of the robot?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/robot_pose', '-n'],
            'expected_topics': ['pose', 'odom'],
            'mock_response': 'x: 5.0\ny: 3.5\ntheta: 0.0',
            'expected_interpretation_parts': ['location', 'position', 'x', 'y', 'robot']
        },
        {
            'query': 'Where is the robot right now?',
            'expected_command_parts': ['ros2', 'topic', 'echo', 'pose'],
            'expected_topics': ['pose', 'odom', 'location'],
            'mock_response': 'x: 15.2\ny: 20.1\ntheta: 3.14',
            'expected_interpretation_parts': ['robot', 'position', 'x', 'y']
        },
        {
            'query': 'Tell me the robot coordinates',
            'expected_command_parts': ['ros2', 'topic', 'echo', 'pose'],
            'expected_topics': ['pose', 'coordinates'],
            'mock_response': 'x: 0.0\ny: 0.0\ntheta: 0.0',
            'expected_interpretation_parts': ['coordinates', 'x', 'y', '0', 'robot']
        },
        {
            'query': 'Get robot position',
            'expected_command_parts': ['ros2', 'topic', 'echo', 'pose', 'odom'],
            'expected_topics': ['pose', 'position'],
            'mock_response': 'x: 12.3\ny: 7.8\ntheta: 0.78',
            'expected_interpretation_parts': ['position', 'x', 'y', 'robot']
        }
    ],
    'battery_queries': [
        {
            'query': 'What is the battery level?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/battery_status', '-n'],
            'expected_topics': ['battery', 'percentage', 'voltage'],
            'mock_response': 'percentage: 0.75\nvoltage: 12.4',
            'expected_interpretation_parts': ['battery', '75', 'percent', '%']
        },
        {
            'query': 'How much battery does the robot have?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/battery', '-n'],
            'expected_topics': ['battery', 'level'],
            'mock_response': 'percentage: 0.25\nvoltage: 11.2',
            'expected_interpretation_parts': ['battery', '25', 'percent', 'level']
        },
        {
            'query': 'Is the robot charging?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/battery', '-n'],
            'expected_topics': ['battery', 'charging', 'power'],
            'mock_response': 'percentage: 0.85\npower_supply_status: 1',
            'expected_interpretation_parts': ['charging', 'yes', 'no', 'battery']
        },
        {
            'query': 'Check battery status',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/battery'],
            'expected_topics': ['battery', 'status'],
            'mock_response': 'percentage: 0.45\nvoltage: 11.8\npower_supply_status: 0',
            'expected_interpretation_parts': ['battery', 'percent', 'status']
        }
    ],
    'status_queries': [
        {
            'query': 'What is the robot status?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/robot_status', '-n'],
            'expected_topics': ['status', 'state'],
            'mock_response': 'data: "IDLE"',
            'expected_interpretation_parts': ['status', 'idle', 'waiting', 'standby']
        },
        {
            'query': 'What is the robot doing now?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/robot_status', '-n'],
            'expected_topics': ['status', 'activity'],
            'mock_response': 'data: "NAVIGATING"',
            'expected_interpretation_parts': ['navigating', 'moving', 'driving']
        },
        {
            'query': 'Is the robot functioning normally?',
            'expected_command_parts': ['ros2', 'topic', 'echo', 'status'],
            'expected_topics': ['status', 'health', 'diagnostics'],
            'mock_response': 'data: "OK"',
            'expected_interpretation_parts': ['functioning', 'normal', 'ok', 'working']
        },
        {
            'query': 'Get current robot state',
            'expected_command_parts': ['ros2', 'topic', 'echo', 'status'],
            'expected_topics': ['status', 'state'],
            'mock_response': 'data: "CHARGING"',
            'expected_interpretation_parts': ['state', 'charging', 'robot']
        }
    ],
    'sensor_queries': [
        {
            'query': 'What does the robot see in front?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/scan', '-n'],
            'expected_topics': ['scan', 'laser', 'lidar', 'range'],
            'mock_response': 'ranges: [0.5, 0.6, 0.7, 0.8]',
            'expected_interpretation_parts': ['obstacle', 'distance', 'cm', 'm', 'front', 'ahead']
        },
        {
            'query': 'Are there any obstacles around the robot?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/scan', '-n'],
            'expected_topics': ['scan', 'obstacles', 'laser'],
            'mock_response': 'ranges: [0.2, inf, inf, 0.3]',
            'expected_interpretation_parts': ['obstacle', 'yes', 'no', 'distance', 'around']
        },
        {
            'query': 'Check LIDAR sensor data',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/scan'],
            'expected_topics': ['scan', 'lidar', 'sensor'],
            'mock_response': 'ranges: [1.2, 1.3, 1.5, 2.0]',
            'expected_interpretation_parts': ['sensor', 'distance', 'lidar', 'range']
        },
        {
            'query': 'What obstacles are detected?',
            'expected_command_parts': ['ros2', 'topic', 'echo', '/scan'],
            'expected_topics': ['scan', 'obstacles', 'detected'],
            'mock_response': 'ranges: [0.3, 0.4, inf, inf]',
            'expected_interpretation_parts': ['obstacle', 'detected', 'distance', 'meter']
        }
    ],
    'command_generation': [
        {
            'query': 'Send the robot to position x=5, y=10',
            'expected_command_parts': ['ros2', 'service', 'call', '/goto_position', '5', '10'],
            'expected_topics': ['goto', 'navigate', 'position', 'service'],
            'mock_response': 'success: True',
            'expected_interpretation_parts': ['sent', 'command', 'navigate', 'position', 'robot']
        },
        {
            'query': 'Stop the robot',
            'expected_command_parts': ['ros2', 'topic', 'pub', '/cmd_vel', '0'],
            'expected_topics': ['cmd_vel', 'stop', 'velocity'],
            'mock_response': 'publishing',
            'expected_interpretation_parts': ['stopped', 'stop', 'command', 'robot']
        },
        {
            'query': 'Move the robot forward',
            'expected_command_parts': ['ros2', 'topic', 'pub', '/cmd_vel', 'linear'],
            'expected_topics': ['cmd_vel', 'move', 'forward', 'linear'],
            'mock_response': 'publishing',
            'expected_interpretation_parts': ['moving', 'forward', 'command', 'robot']
        },
        {
            'query': 'Navigate to the charging station',
            'expected_command_parts': ['ros2', 'service', 'call', 'goto', 'navigate'],
            'expected_topics': ['navigate', 'goto', 'charging', 'station'],
            'mock_response': 'success: True\nmessage: "Navigating to charging station"',
            'expected_interpretation_parts': ['navigating', 'charging', 'station', 'robot']
        },
        {
            'query': 'Rotate the robot 90 degrees',
            'expected_command_parts': ['ros2', 'topic', 'pub', '/cmd_vel', 'angular'],
            'expected_topics': ['cmd_vel', 'rotate', 'angular'],
            'mock_response': 'publishing',
            'expected_interpretation_parts': ['rotating', 'turn', 'degrees', 'robot']
        }
    ]
}

def run_inference(model: str, prompt: str, system_prompt: str, timeout: int = 120) -> Dict:
    """Run inference on a model and return results."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{MODEL_RUNNER_URL}/engines/llama.cpp/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code != 200:
            return {
                'status': 'error',
                'error': f"HTTP {response.status_code}: {response.text}",
                'response': '',
                'total_time': total_time
            }
        
        result = response.json()
        
        # Extract response text
        response_text = ""
        if 'choices' in result and len(result['choices']) > 0:
            response_text = result['choices'][0].get('message', {}).get('content', '')
        
        # Extract token usage
        usage = result.get('usage', {})
        
        return {
            'status': 'success',
            'response': response_text,
            'total_time': total_time,
            'completion_tokens': usage.get('completion_tokens', 0),
            'prompt_tokens': usage.get('prompt_tokens', 0)
        }
        
    except requests.exceptions.Timeout:
        return {
            'status': 'timeout',
            'error': 'Request timeout',
            'response': '',
            'total_time': timeout
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'response': '',
            'total_time': time.time() - start_time
        }

def evaluate_command_generation(response: str, expected_parts: List[str]) -> Tuple[bool, float]:
    """
    Evaluate if the response contains a valid ROS2 command.
    Returns (is_valid, score)
    """
    response_lower = response.lower()
    
    # Check if response contains expected command parts
    parts_found = sum(1 for part in expected_parts if part.lower() in response_lower)
    score = (parts_found / len(expected_parts)) * 100 if expected_parts else 0
    
    # Check for ROS2 command structure
    has_ros2_command = 'ros2' in response_lower
    has_valid_verb = any(verb in response_lower for verb in ['topic', 'service', 'node', 'param'])
    
    is_valid = has_ros2_command and has_valid_verb and score >= 60
    
    return is_valid, score

def evaluate_interpretation(response: str, expected_parts: List[str]) -> Tuple[bool, float]:
    """
    Evaluate if the interpretation is human-friendly and accurate.
    Returns (is_good, score)
    """
    response_lower = response.lower()
    
    # Check for expected interpretation elements
    parts_found = sum(1 for part in expected_parts if part.lower() in response_lower)
    score = (parts_found / len(expected_parts)) * 100 if expected_parts else 0
    
    # Check for natural language (not just raw data)
    has_numbers = bool(re.search(r'\d+\.?\d*', response))
    has_words = len(response.split()) > 5
    not_just_command = 'ros2' not in response_lower or len(response_lower.split('ros2')) > 2
    
    is_good = has_words and not_just_command and score >= 40
    
    return is_good, score

def test_agent_two_stage(model: str, scenario: Dict, system_prompt: str) -> Dict:
    """
    Test agent in two stages:
    1. Generate ROS2 command from natural language query
    2. Interpret mock ROS2 response into natural language
    """
    query = scenario['query']
    expected_cmd_parts = scenario['expected_command_parts']
    mock_response = scenario['mock_response']
    expected_interp_parts = scenario['expected_interpretation_parts']
    
    # Stage 1: Generate command
    stage1_prompt = f"User asks: '{query}'\n\nGenerate the necessary ROS2 command to get this information."
    
    result1 = run_inference(model, stage1_prompt, system_prompt)
    
    if result1['status'] != 'success':
        return {
            'stage': 'command_generation',
            'status': result1['status'],
            'error': result1.get('error', ''),
            'command_response': '',
            'command_valid': False,
            'command_score': 0,
            'interpretation_response': '',
            'interpretation_valid': False,
            'interpretation_score': 0,
            'total_time': result1['total_time']
        }
    
    command_response = result1['response']
    command_valid, command_score = evaluate_command_generation(command_response, expected_cmd_parts)
    
    # Stage 2: Interpret response
    stage2_prompt = f"""User asked: '{query}'

The ROS2 command was executed and returned the following response:
{mock_response}

Interpret this response and provide a clear, natural language message in English for the user."""
    
    result2 = run_inference(model, stage2_prompt, system_prompt)
    
    if result2['status'] != 'success':
        return {
            'stage': 'interpretation',
            'status': result2['status'],
            'error': result2.get('error', ''),
            'command_response': command_response,
            'command_valid': command_valid,
            'command_score': command_score,
            'interpretation_response': '',
            'interpretation_valid': False,
            'interpretation_score': 0,
            'total_time': result1['total_time'] + result2['total_time']
        }
    
    interpretation_response = result2['response']
    interpretation_valid, interpretation_score = evaluate_interpretation(interpretation_response, expected_interp_parts)
    
    return {
        'stage': 'complete',
        'status': 'success',
        'command_response': command_response,
        'command_valid': command_valid,
        'command_score': command_score,
        'interpretation_response': interpretation_response,
        'interpretation_valid': interpretation_valid,
        'interpretation_score': interpretation_score,
        'total_time': result1['total_time'] + result2['total_time'],
        'total_tokens': result1['completion_tokens'] + result2['completion_tokens']
    }

def save_results(results: List[Dict], output_file: str):
    """Save test results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    fieldnames = [
        'timestamp', 'model', 'category', 'query',
        'command_response', 'command_valid', 'command_score',
        'mock_data', 'interpretation_response', 'interpretation_valid', 'interpretation_score',
        'status', 'stage', 'total_time', 'total_tokens', 'error'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to: {output_file}")

def print_summary(results: List[Dict]):
    """Print a summary of test results."""
    print("\n" + "="*80)
    print("ROS2 COMMAND AGENT TEST SUMMARY")
    print("="*80)
    
    # Group by model
    models = {}
    for result in results:
        model = result['model']
        if model not in models:
            models[model] = {
                'total': 0,
                'success': 0,
                'command_valid': 0,
                'interpretation_valid': 0,
                'both_valid': 0,
                'avg_command_score': [],
                'avg_interpretation_score': [],
                'avg_time': []
            }
        
        stats = models[model]
        stats['total'] += 1
        
        if result['status'] == 'success':
            stats['success'] += 1
            if result.get('command_valid'):
                stats['command_valid'] += 1
            if result.get('interpretation_valid'):
                stats['interpretation_valid'] += 1
            if result.get('command_valid') and result.get('interpretation_valid'):
                stats['both_valid'] += 1
            
            stats['avg_command_score'].append(result.get('command_score', 0))
            stats['avg_interpretation_score'].append(result.get('interpretation_score', 0))
            stats['avg_time'].append(result.get('total_time', 0))
    
    # Print summary for each model
    for model, stats in models.items():
        print(f"\n{model}:")
        print(f"  Total tests: {stats['total']}")
        print(f"  Successful: {stats['success']}")
        print(f"  Valid commands: {stats['command_valid']}/{stats['total']} ({stats['command_valid']/stats['total']*100:.1f}%)")
        print(f"  Valid interpretations: {stats['interpretation_valid']}/{stats['total']} ({stats['interpretation_valid']/stats['total']*100:.1f}%)")
        print(f"  Both valid: {stats['both_valid']}/{stats['total']} ({stats['both_valid']/stats['total']*100:.1f}%)")
        
        if stats['avg_command_score']:
            avg_cmd = sum(stats['avg_command_score']) / len(stats['avg_command_score'])
            avg_interp = sum(stats['avg_interpretation_score']) / len(stats['avg_interpretation_score'])
            avg_time = sum(stats['avg_time']) / len(stats['avg_time'])
            
            print(f"  Avg command score: {avg_cmd:.1f}/100")
            print(f"  Avg interpretation score: {avg_interp:.1f}/100")
            print(f"  Avg total time: {avg_time:.2f}s")
    
    print("\n" + "="*80)

def run_tests(models: List[str], categories: List[str], output_file: str):
    """Run the complete test suite."""
    print("="*80)
    print("ROS2 COMMAND AGENT TESTING - NATURAL LANGUAGE INTERFACE")
    print("="*80)
    print(f"\nModels tested: {', '.join(models)}")
    print(f"Categories: {', '.join(categories)}")
    print(f"\nThe agent will be tested in 2 stages:")
    print("  1. Generate ROS2 command from natural language query")
    print("  2. Interpret ROS2 response into natural language")
    print("\n" + "="*80 + "\n")
    
    results = []
    
    # Collect all scenarios
    all_scenarios = []
    for category in categories:
        if category in TEST_SCENARIOS:
            for scenario in TEST_SCENARIOS[category]:
                all_scenarios.append({**scenario, 'category': category})
    
    total_tests = len(models) * len(all_scenarios)
    current_test = 0
    
    # Run tests
    for model in models:
        print(f"\n{'='*80}")
        print(f"Testing model: {model}")
        print('='*80)
        
        for scenario in all_scenarios:
            current_test += 1
            category = scenario['category']
            query = scenario['query']
            
            print(f"\n[{current_test}/{total_tests}] {category.upper()}")
            print(f"Query: {query}")
            
            # Run two-stage test
            result = test_agent_two_stage(model, scenario, ROS_COMMAND_AGENT_PROMPT)
            
            # Print results
            if result['status'] == 'success':
                print(f"\n  ✓ Stage 1 - Command generation:")
                print(f"    {result['command_response'][:100]}...")
                print(f"    Valid: {'✓' if result['command_valid'] else '✗'} | Score: {result['command_score']:.1f}/100")
                
                print(f"\n  ✓ Stage 2 - Interpretation:")
                print(f"    {result['interpretation_response'][:100]}...")
                print(f"    Valid: {'✓' if result['interpretation_valid'] else '✗'} | Score: {result['interpretation_score']:.1f}/100")
                
                print(f"\n  Total time: {result['total_time']:.2f}s | Tokens: {result.get('total_tokens', 0)}")
            else:
                print(f"\n  ✗ Error in stage '{result['stage']}': {result.get('error', 'Unknown')}")
            
            # Save result
            results.append({
                'timestamp': datetime.now().isoformat(),
                'model': model,
                'category': category,
                'query': query,
                'command_response': result.get('command_response', '')[:500],
                'command_valid': result.get('command_valid', False),
                'command_score': round(result.get('command_score', 0), 2),
                'mock_data': scenario.get('mock_response', ''),
                'interpretation_response': result.get('interpretation_response', '')[:500],
                'interpretation_valid': result.get('interpretation_valid', False),
                'interpretation_score': round(result.get('interpretation_score', 0), 2),
                'status': result['status'],
                'stage': result.get('stage', 'unknown'),
                'total_time': round(result.get('total_time', 0), 2),
                'total_tokens': result.get('total_tokens', 0),
                'error': result.get('error', '')
            })
            
            # Small delay between requests
            time.sleep(0.5)
    
    # Save and summarize
    save_results(results, output_file)
    print_summary(results)
    
    return results

def main():
    global MODEL_RUNNER_URL
    
    parser = argparse.ArgumentParser(
        description='Test ROS2 agents for natural language interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with all categories
  python test_ros_command_agent.py --models all
  
  # Test specific models with position queries
  python test_ros_command_agent.py --models ai/qwen3 --categories position_queries
  
  # Test with custom output file
  python test_ros_command_agent.py --output agent_results.csv
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to test. Use "all" for all models.'
    )
    
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['all'],
        choices=['all', 'position_queries', 'battery_queries', 'status_queries', 'sensor_queries', 'command_generation'],
        help='Test categories.'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Output CSV file.'
    )
    
    parser.add_argument(
        '--model-runner-url',
        default=MODEL_RUNNER_URL,
        help='Model runner API URL.'
    )
    
    args = parser.parse_args()
    
    # Set model runner URL
    MODEL_RUNNER_URL = args.model_runner_url
    
    # Parse models
    if 'all' in args.models:
        models = ALL_MODELS
    else:
        models = args.models
    
    # Parse categories
    if 'all' in args.categories:
        categories = list(TEST_SCENARIOS.keys())
    else:
        categories = args.categories
    
    # Run tests
    try:
        run_tests(models, categories, args.output)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

