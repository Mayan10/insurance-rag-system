#!/usr/bin/env python3
"""
Test script for the Insurance RAG System API
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed URL
API_KEY = "hackrx-2024-secret-key"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_hackrx_endpoint():
    """Test the main hackrx/run endpoint"""
    print("\nTesting hackrx/run endpoint...")
    
    # Test data based on hackathon requirements
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=test_data,
            timeout=60  # 60 second timeout
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of answers: {len(result.get('answers', []))}")
            
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"\nAnswer {i}:")
                print(f"Question: {test_data['questions'][i-1]}")
                print(f"Answer: {answer[:200]}...")
            
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out (>60 seconds)")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_authentication():
    """Test authentication"""
    print("\nTesting authentication...")
    
    test_data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question"]
    }
    
    # Test without API key
    try:
        response = requests.post(
            f"{API_BASE_URL}/hackrx/run",
            json=test_data,
            timeout=10
        )
        print(f"Without API key - Status: {response.status_code}")
        if response.status_code == 401:
            print("✅ Authentication working correctly")
        else:
            print("❌ Authentication not working")
    except Exception as e:
        print(f"Error testing authentication: {e}")

def main():
    """Run all tests"""
    print("🚀 Insurance RAG System API Tests")
    print("=" * 50)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Test authentication
    test_authentication()
    
    # Test main endpoint
    api_ok = test_hackrx_endpoint()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"API Endpoint: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if health_ok and api_ok:
        print("\n🎉 All tests passed! Your API is ready for the hackathon!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 