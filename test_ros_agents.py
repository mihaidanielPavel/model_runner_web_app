"""
ROS/ROS2 Expert Agent Testing Script

This script evaluates different language models as ROS/ROS2 experts by testing them
with a comprehensive set of prompts covering various aspects of ROS development.

Usage:
    python test_ros_agents.py --models all
    python test_ros_agents.py --models ai/llama3.2 ai/qwen3
    python test_ros_agents.py --prompts beginner
    python test_ros_agents.py --models ai/qwen3 --prompts advanced --output results.csv
"""

import requests
import time
import csv
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import os

# Configuration
MODEL_RUNNER_URL = os.getenv('MODEL_RUNNER_URL', 'http://model-runner.docker.internal') #http://localhost:12434/
DEFAULT_OUTPUT = f'ros_agent_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

# All available models
ALL_MODELS = [
    'ai/smollm2',
    'ai/llama3.2',
    'ai/gemma3',
    'ai/gemma3-qat'
]

# ROS/ROS2 System Prompt for Expert Agent
ROS_EXPERT_SYSTEM_PROMPT = """You are an expert robotics software engineer with deep knowledge of ROS (Robot Operating System) and ROS2. You have extensive experience in:
- ROS and ROS2 architecture, concepts, and best practices
- Creating and managing nodes, topics, services, and actions
- Working with different message types and custom messages
- Launch files, parameter servers, and configuration management
- Robot hardware integration and driver development
- Navigation, SLAM, perception, and manipulation
- Debugging and troubleshooting ROS systems
- Migration from ROS1 to ROS2
- Real-time systems and performance optimization

Provide detailed, accurate, and practical answers. Include code examples when relevant. Focus on best practices and explain the reasoning behind your recommendations."""

# Comprehensive ROS/ROS2 Test Prompts
ROS_TEST_PROMPTS = {
    'beginner': [
        {
            'category': 'Basics',
            'prompt': 'What is the difference between ROS and ROS2? When should I use each?',
            'expected_topics': ['middleware', 'real-time', 'DDS', 'Python 3', 'lifecycle']
        },
        {
            'category': 'Basics',
            'prompt': 'Explain the concept of ROS nodes and topics. How do they communicate?',
            'expected_topics': ['publish-subscribe', 'message passing', 'decoupling']
        },
        {
            'category': 'Installation',
            'prompt': 'How do I install ROS2 on Ubuntu 22.04?',
            'expected_topics': ['apt', 'repositories', 'workspace', 'sourcing']
        },
        {
            'category': 'Basic Programming',
            'prompt': 'Write a simple ROS2 Python publisher node that publishes string messages.',
            'expected_topics': ['rclpy', 'Node', 'create_publisher', 'timer', 'spin']
        },
        {
            'category': 'Basic Programming',
            'prompt': 'How do I create a subscriber in ROS2 to listen to a topic?',
            'expected_topics': ['create_subscription', 'callback', 'QoS']
        }
    ],
    'intermediate': [
        {
            'category': 'Services',
            'prompt': 'Explain the difference between topics and services in ROS2. Provide a Python example of creating a service.',
            'expected_topics': ['synchronous', 'request-response', 'srv file', 'client-server']
        },
        {
            'category': 'Launch Files',
            'prompt': 'How do I create a launch file in ROS2 to start multiple nodes with parameters?',
            'expected_topics': ['launch.py', 'Node', 'LaunchDescription', 'parameters']
        },
        {
            'category': 'Custom Messages',
            'prompt': 'Walk me through creating a custom ROS2 message type. Include the .msg file and CMakeLists setup.',
            'expected_topics': ['interface', 'package', 'rosidl', 'CMakeLists', 'package.xml']
        },
        {
            'category': 'TF',
            'prompt': 'What is TF2 in ROS2? How do I publish and listen to transforms?',
            'expected_topics': ['coordinate frames', 'tf2_ros', 'TransformBroadcaster', 'TransformListener']
        },
        {
            'category': 'Parameters',
            'prompt': 'How do I use parameters in ROS2 nodes? Show how to declare, get, and set parameters dynamically.',
            'expected_topics': ['declare_parameter', 'get_parameter', 'add_on_set_parameters_callback']
        },
        {
            'category': 'Quality of Service',
            'prompt': 'Explain QoS (Quality of Service) profiles in ROS2 and when to use different settings.',
            'expected_topics': ['reliability', 'durability', 'history', 'best effort', 'reliable']
        }
    ],
    'advanced': [
        {
            'category': 'Actions',
            'prompt': 'Implement a ROS2 action server for a long-running task with feedback. Include Python code.',
            'expected_topics': ['action', 'goal', 'feedback', 'result', 'ActionServer']
        },
        {
            'category': 'Lifecycle',
            'prompt': 'Explain ROS2 managed (lifecycle) nodes. Why are they important and how do I implement one?',
            'expected_topics': ['lifecycle', 'state machine', 'configure', 'activate', 'transitions']
        },
        {
            'category': 'Navigation',
            'prompt': 'How do I set up Nav2 for autonomous navigation? What are the key components?',
            'expected_topics': ['costmap', 'planners', 'controllers', 'behavior tree', 'AMCL']
        },
        {
            'category': 'Perception',
            'prompt': 'How do I process point cloud data from a depth camera in ROS2? Include sensor_msgs/PointCloud2.',
            'expected_topics': ['sensor_msgs', 'PointCloud2', 'pcl', 'filtering', 'transformations']
        },
        {
            'category': 'Real-time',
            'prompt': 'What are best practices for achieving real-time performance in ROS2?',
            'expected_topics': ['DDS', 'executors', 'memory allocation', 'priority', 'kernel']
        },
        {
            'category': 'Multi-robot',
            'prompt': 'How do I set up multiple robots in ROS2 with namespace isolation?',
            'expected_topics': ['namespace', 'domain_id', 'tf_prefix', 'remapping']
        }
    ],
    'expert': [
        {
            'category': 'Performance',
            'prompt': 'Analyze and optimize a ROS2 system with high message throughput. What tools and techniques should I use?',
            'expected_topics': ['intra-process', 'zero-copy', 'composition', 'profiling', 'ros2_tracing']
        },
        {
            'category': 'Hardware Integration',
            'prompt': 'Design a hardware driver for a custom sensor in ROS2. What architecture patterns should I follow?',
            'expected_topics': ['lifecycle', 'diagnostics', 'error handling', 'time synchronization', 'urdf']
        },
        {
            'category': 'Security',
            'prompt': 'How do I implement ROS2 security (SROS2) for a production robot system?',
            'expected_topics': ['DDS security', 'certificates', 'permissions', 'encryption', 'authentication']
        },
        {
            'category': 'Testing',
            'prompt': 'What is the best approach for testing ROS2 nodes? Include unit tests and integration tests.',
            'expected_topics': ['pytest', 'launch_testing', 'mock', 'fixtures', 'CI/CD']
        },
        {
            'category': 'Migration',
            'prompt': 'I have a large ROS1 system. What is the best strategy to migrate to ROS2?',
            'expected_topics': ['ros1_bridge', 'porting', 'API differences', 'gradual migration', 'compatibility']
        },
        {
            'category': 'Distributed Systems',
            'prompt': 'Design a distributed ROS2 system across multiple machines. How do I handle discovery, network issues, and synchronization?',
            'expected_topics': ['DDS discovery', 'fastdds', 'QoS', 'time sync', 'network configuration']
        }
    ],
    'troubleshooting': [
        {
            'category': 'Debugging',
            'prompt': 'My ROS2 nodes cannot see each other. How do I troubleshoot this issue?',
            'expected_topics': ['domain_id', 'discovery', 'multicast', 'firewall', 'ros2 daemon']
        },
        {
            'category': 'Performance Issues',
            'prompt': 'My ROS2 system has high latency. What are common causes and solutions?',
            'expected_topics': ['QoS', 'network', 'CPU', 'message size', 'serialization']
        },
        {
            'category': 'Build Problems',
            'prompt': 'My ROS2 package fails to build with CMake errors. What should I check?',
            'expected_topics': ['dependencies', 'package.xml', 'CMakeLists', 'ament', 'sourcing']
        }
    ]
}

def run_inference(model: str, prompt: str, system_prompt: str, timeout: int = 300) -> Dict:
    """Run inference on a model and return results with timing metrics."""
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
                'total_time': total_time,
                'tokens': 0,
                'tokens_per_sec': 0
            }
        
        result = response.json()
        
        # Extract response text
        response_text = ""
        if 'choices' in result and len(result['choices']) > 0:
            response_text = result['choices'][0].get('message', {}).get('content', '')
        
        # Extract token usage
        usage = result.get('usage', {})
        completion_tokens = usage.get('completion_tokens', 0)
        
        # Calculate tokens per second
        tokens_per_sec = completion_tokens / total_time if total_time > 0 else 0
        
        return {
            'status': 'success',
            'response': response_text,
            'total_time': total_time,
            'tokens': completion_tokens,
            'tokens_per_sec': tokens_per_sec,
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }
        
    except requests.exceptions.Timeout:
        return {
            'status': 'timeout',
            'error': 'Request timeout',
            'response': '',
            'total_time': timeout,
            'tokens': 0,
            'tokens_per_sec': 0
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'response': '',
            'total_time': time.time() - start_time,
            'tokens': 0,
            'tokens_per_sec': 0
        }

def evaluate_response_quality(response: str, expected_topics: List[str]) -> Dict:
    """
    Evaluate the quality of the response based on expected topics coverage.
    Returns a score and analysis.
    """
    response_lower = response.lower()
    
    # Check for expected topics
    topics_covered = sum(1 for topic in expected_topics if topic.lower() in response_lower)
    coverage_score = (topics_covered / len(expected_topics)) * 100 if expected_topics else 0
    
    # Response quality metrics
    has_code = 'def ' in response or 'class ' in response or '```' in response
    response_length = len(response)
    
    # Simple quality scoring
    quality_score = 0
    if response_length > 100:
        quality_score += 25
    if response_length > 300:
        quality_score += 25
    if has_code:
        quality_score += 25
    if coverage_score > 50:
        quality_score += 25
    
    return {
        'coverage_score': coverage_score,
        'quality_score': quality_score,
        'topics_covered': topics_covered,
        'total_topics': len(expected_topics),
        'has_code': has_code,
        'response_length': response_length
    }

def save_results(results: List[Dict], output_file: str):
    """Save test results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    fieldnames = [
        'timestamp', 'model', 'category', 'prompt', 'response',
        'status', 'total_time', 'tokens', 'tokens_per_sec',
        'prompt_tokens', 'total_tokens',
        'coverage_score', 'quality_score', 'topics_covered', 'total_topics',
        'has_code', 'response_length', 'error'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to: {output_file}")

def print_summary(results: List[Dict]):
    """Print a summary of test results."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    # Group by model
    models = {}
    for result in results:
        model = result['model']
        if model not in models:
            models[model] = {
                'total': 0,
                'success': 0,
                'error': 0,
                'timeout': 0,
                'avg_time': [],
                'avg_tokens_per_sec': [],
                'avg_quality': [],
                'avg_coverage': []
            }
        
        models[model]['total'] += 1
        models[model][result['status']] = models[model].get(result['status'], 0) + 1
        
        if result['status'] == 'success':
            models[model]['avg_time'].append(result['total_time'])
            models[model]['avg_tokens_per_sec'].append(result['tokens_per_sec'])
            models[model]['avg_quality'].append(result.get('quality_score', 0))
            models[model]['avg_coverage'].append(result.get('coverage_score', 0))
    
    # Print summary for each model
    for model, stats in models.items():
        print(f"\n{model}:")
        print(f"  Total tests: {stats['total']}")
        print(f"  Successful: {stats['success']}")
        print(f"  Errors: {stats.get('error', 0)}")
        print(f"  Timeouts: {stats.get('timeout', 0)}")
        
        if stats['avg_time']:
            avg_time = sum(stats['avg_time']) / len(stats['avg_time'])
            avg_tps = sum(stats['avg_tokens_per_sec']) / len(stats['avg_tokens_per_sec'])
            avg_quality = sum(stats['avg_quality']) / len(stats['avg_quality'])
            avg_coverage = sum(stats['avg_coverage']) / len(stats['avg_coverage'])
            
            print(f"  Avg Response Time: {avg_time:.2f}s")
            print(f"  Avg Tokens/sec: {avg_tps:.2f}")
            print(f"  Avg Quality Score: {avg_quality:.1f}/100")
            print(f"  Avg Coverage Score: {avg_coverage:.1f}%")
    
    print("\n" + "="*80)

def run_tests(models: List[str], prompt_levels: List[str], output_file: str, system_prompt: str):
    """Run the complete test suite."""
    print(f"Starting ROS/ROS2 Agent Testing")
    print(f"Models: {', '.join(models)}")
    print(f"Prompt Levels: {', '.join(prompt_levels)}")
    print(f"System Prompt: {system_prompt[:100]}...")
    print("\n" + "="*80 + "\n")
    
    results = []
    
    # Collect all prompts to test
    all_prompts = []
    for level in prompt_levels:
        if level in ROS_TEST_PROMPTS:
            all_prompts.extend([{**p, 'level': level} for p in ROS_TEST_PROMPTS[level]])
    
    total_tests = len(models) * len(all_prompts)
    current_test = 0
    
    # Run tests
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 80)
        
        for prompt_data in all_prompts:
            current_test += 1
            category = prompt_data['category']
            level = prompt_data['level']
            prompt = prompt_data['prompt']
            expected_topics = prompt_data.get('expected_topics', [])
            
            print(f"\n[{current_test}/{total_tests}] {level.upper()} - {category}")
            print(f"Prompt: {prompt[:80]}...")
            
            # Run inference
            result = run_inference(model, prompt, system_prompt)
            
            # Evaluate response quality
            if result['status'] == 'success':
                quality_metrics = evaluate_response_quality(result['response'], expected_topics)
                result.update(quality_metrics)
                print(f"✓ Success - Time: {result['total_time']:.2f}s, "
                      f"Tokens/s: {result['tokens_per_sec']:.1f}, "
                      f"Quality: {quality_metrics['quality_score']}/100, "
                      f"Coverage: {quality_metrics['coverage_score']:.1f}%")
            else:
                print(f"✗ {result['status'].upper()}: {result.get('error', 'Unknown error')}")
                result.update({
                    'coverage_score': 0,
                    'quality_score': 0,
                    'topics_covered': 0,
                    'total_topics': len(expected_topics),
                    'has_code': False,
                    'response_length': 0
                })
            
            # Save result
            results.append({
                'timestamp': datetime.now().isoformat(),
                'model': model,
                'category': category,
                'prompt': prompt,
                'response': result['response'][:1000],  # Truncate for CSV
                'status': result['status'],
                'total_time': round(result['total_time'], 2),
                'tokens': result['tokens'],
                'tokens_per_sec': round(result['tokens_per_sec'], 2),
                'prompt_tokens': result.get('prompt_tokens', 0),
                'total_tokens': result.get('total_tokens', 0),
                'coverage_score': round(result['coverage_score'], 2),
                'quality_score': result['quality_score'],
                'topics_covered': result['topics_covered'],
                'total_topics': result['total_topics'],
                'has_code': result['has_code'],
                'response_length': result['response_length'],
                'error': result.get('error', '')
            })
            
            # Small delay between requests
            time.sleep(1)
    
    # Save and summarize
    save_results(results, output_file)
    print_summary(results)
    
    return results

def main():
    global MODEL_RUNNER_URL
    parser = argparse.ArgumentParser(
        description='Test language models as ROS/ROS2 expert agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with all prompt levels
  python test_ros_agents.py --models all --prompts all
  
  # Test specific models with beginner prompts
  python test_ros_agents.py --models ai/llama3.2 ai/qwen3 --prompts beginner
  
  # Test advanced and expert prompts only
  python test_ros_agents.py --models all --prompts advanced expert
  
  # Custom output file
  python test_ros_agents.py --models ai/qwen3 --output my_results.csv
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to test. Use "all" for all models or specify model IDs.'
    )
    
    parser.add_argument(
        '--prompts',
        nargs='+',
        default=['all'],
        choices=['all', 'beginner', 'intermediate', 'advanced', 'expert', 'troubleshooting'],
        help='Prompt difficulty levels to test.'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Output CSV file path.'
    )
    
    parser.add_argument(
        '--system-prompt',
        default=ROS_EXPERT_SYSTEM_PROMPT,
        help='Custom system prompt for the agent.'
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
    
    # Parse prompt levels
    if 'all' in args.prompts:
        prompt_levels = list(ROS_TEST_PROMPTS.keys())
    else:
        prompt_levels = args.prompts
    
    # Run tests
    try:
        run_tests(models, prompt_levels, args.output, args.system_prompt)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

