#!/usr/bin/env python3
"""Test websearch functionality"""
import requests
import json

# Test 1: Normal request (should work as before)
print("Test 1: Normal request without websearch tool")
response = requests.post(
    "http://localhost:8001/v1/messages",
    headers={
        "Authorization": "Bearer newapi-q2api",
        "Content-Type": "application/json"
    },
    json={
        "model": "claude-sonnet-4.5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}\n")

# Test 2: Websearch request
print("Test 2: Websearch request")
response = requests.post(
    "http://localhost:8001/v1/messages",
    headers={
        "Authorization": "Bearer newapi-q2api",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    },
    json={
        "model": "claude-sonnet-4.5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Perform a web search for the query: Python 3.12 new features"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        "stream": True
    },
    stream=True
)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("SSE Events:")
    for i, line in enumerate(response.iter_lines()):
        if i > 20:  # Limit output
            print("...")
            break
        if line:
            print(line.decode('utf-8'))
else:
    print(f"Error: {response.text}")
