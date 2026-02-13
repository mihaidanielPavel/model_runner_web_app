#!/usr/bin/env python3
"""
Test script to verify static files are accessible.
"""

import requests
import sys

def test_static_files(base_url="http://127.0.0.1:5000"):
    """Test if static files are accessible."""
    print(f"Testing static files at {base_url}")
    
    # Test debug endpoint first
    try:
        response = requests.get(f"{base_url}/debug", timeout=5)
        if response.status_code == 200:
            debug_info = response.json()
            print("✓ Debug endpoint accessible")
            print(f"  Static folder: {debug_info.get('static_folder')}")
            print(f"  Index exists: {debug_info.get('index_exists')}")
            print(f"  App.js exists: {debug_info.get('app_js_exists')}")
            print(f"  Style.css exists: {debug_info.get('style_css_exists')}")
            print(f"  Static files: {debug_info.get('static_files')}")
        else:
            print(f"✗ Debug endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Debug endpoint error: {e}")
    
    # Test main page
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✓ Main page accessible")
            if "ROS2 Agent Dashboard" in response.text:
                print("✓ Dashboard content found")
            else:
                print("⚠ Dashboard content not found (fallback page?)")
        else:
            print(f"✗ Main page failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Main page error: {e}")
    
    # Test static files
    static_files = ['app.js', 'style.css', 'index.html']
    for filename in static_files:
        try:
            response = requests.get(f"{base_url}/{filename}", timeout=5)
            if response.status_code == 200:
                print(f"✓ {filename} accessible")
            else:
                print(f"✗ {filename} failed: {response.status_code}")
        except Exception as e:
            print(f"✗ {filename} error: {e}")

if __name__ == '__main__':
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000"
    test_static_files(base_url)



