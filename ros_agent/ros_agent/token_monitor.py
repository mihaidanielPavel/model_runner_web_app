#!/usr/bin/env python3
"""
Token Usage Monitor for ROS2 Agent

Simple utility to monitor and estimate token usage for the AI agent.
"""

def estimate_tokens(text: str) -> int:
    """Estimate token count using character-based approximation."""
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def analyze_context_usage(context: str) -> dict:
    """Analyze context usage and provide recommendations."""
    tokens = estimate_tokens(context)
    
    analysis = {
        'estimated_tokens': tokens,
        'usage_percentage': (tokens / 4096) * 100,
        'recommendation': '',
        'is_efficient': tokens < 2000
    }
    
    if tokens < 1000:
        analysis['recommendation'] = 'Very efficient - plenty of room for conversation'
    elif tokens < 2000:
        analysis['recommendation'] = 'Good efficiency - room for detailed responses'
    elif tokens < 3000:
        analysis['recommendation'] = 'Moderate efficiency - consider reducing context'
    else:
        analysis['recommendation'] = 'High usage - context may be too verbose'
    
    return analysis

def optimize_context(context: str, max_tokens: int = 2000) -> str:
    """Optimize context to fit within token limit."""
    if estimate_tokens(context) <= max_tokens:
        return context
    
    # Simple optimization: truncate while preserving structure
    lines = context.split('\n')
    optimized_lines = []
    current_tokens = 0
    
    for line in lines:
        line_tokens = estimate_tokens(line)
        if current_tokens + line_tokens <= max_tokens:
            optimized_lines.append(line)
            current_tokens += line_tokens
        else:
            break
    
    return '\n'.join(optimized_lines)

if __name__ == "__main__":
    # Test the token estimation
    test_context = """
    Robot: Navigation status: idle, Current position (map frame): (1.234, 2.567), Frame: map
    Sensors: LiDAR(/scan): 360pts, avg:2.5m, 1s; Camera(/camera/image_raw): 2s
    Topics: Available topics with latest data: /scan, /camera/image_raw, /amcl_pose
    Services: Services(5): costmap:2, map:1, nav:1, other:1
    Functions: move_to_position(x,y,yaw), stop_navigation(), get_latest_topic_data(topic), call_service(name,data), get_navigation_status()
    """
    
    analysis = analyze_context_usage(test_context)
    print(f"Token Analysis:")
    print(f"  Estimated tokens: {analysis['estimated_tokens']}")
    print(f"  Usage: {analysis['usage_percentage']:.1f}% of 4096 limit")
    print(f"  Recommendation: {analysis['recommendation']}")
    print(f"  Efficient: {analysis['is_efficient']}")
    
    if not analysis['is_efficient']:
        optimized = optimize_context(test_context)
        optimized_analysis = analyze_context_usage(optimized)
        print(f"\nOptimized version:")
        print(f"  Tokens: {optimized_analysis['estimated_tokens']}")
        print(f"  Usage: {optimized_analysis['usage_percentage']:.1f}%")
        print(f"  Text: {optimized}")
