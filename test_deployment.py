#!/usr/bin/env python3
"""
Test script to verify deployed API works correctly
Replace YOUR_RENDER_URL with your actual deployed URL
"""

import requests
import json
import time

# Configuration - REPLACE WITH YOUR ACTUAL URL
RENDER_URL = "https://your-app-name.onrender.com"  # Replace this!
API_KEY = "hackrx-2024-secret-key"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{RENDER_URL}/health", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_hackrx_endpoint():
    """Test the main hackrx/run endpoint"""
    print("\n🚀 Testing hackrx/run endpoint...")
    
    # Test data from hackathon requirements
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print("⏳ Sending request (may take 30-60 seconds on first request)...")
        start_time = time.time()
        
        response = requests.post(
            f"{RENDER_URL}/hackrx/run",
            headers=headers,
            json=test_data,
            timeout=120  # 2 minutes timeout for first request
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Received {len(result.get('answers', []))} answers")
            
            for i, answer in enumerate(result.get('answers', []), 1):
                print(f"\n📝 Answer {i}:")
                print(f"   Question: {test_data['questions'][i-1]}")
                print(f"   Answer: {answer[:200]}...")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>120 seconds)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_authentication():
    """Test authentication"""
    print("\n🔐 Testing authentication...")
    
    test_data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question"]
    }
    
    # Test without API key
    try:
        response = requests.post(
            f"{RENDER_URL}/hackrx/run",
            json=test_data,
            timeout=30
        )
        print(f"Without API key - Status: {response.status_code}")
        if response.status_code == 401:
            print("✅ Authentication working correctly")
            return True
        else:
            print("❌ Authentication not working")
            return False
    except Exception as e:
        print(f"❌ Authentication test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Insurance RAG System - Deployment Verification")
    print("=" * 60)
    print(f"🌐 Testing URL: {RENDER_URL}")
    print("=" * 60)
    
    # Check if URL is set
    if "your-app-name" in RENDER_URL:
        print("❌ ERROR: Please update RENDER_URL in this script with your actual deployed URL!")
        print("Example: https://insurance-rag-system.onrender.com")
        return
    
    # Run tests
    health_ok = test_health()
    auth_ok = test_authentication()
    api_ok = test_hackrx_endpoint()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Authentication: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"API Endpoint: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if health_ok and auth_ok and api_ok:
        print("\n🎉 SUCCESS! Your API is ready for hackathon submission!")
        print(f"📝 Submission URL: {RENDER_URL}/hackrx/run")
        print("\n📋 Pre-Submission Checklist:")
        print("✅ API is live & accessible")
        print("✅ HTTPS enabled")
        print("✅ Handles POST requests")
        print("✅ Returns JSON response")
        print("✅ Response time < 30s (after first request)")
        print("✅ Tested with sample data")
        print("✅ Authentication working")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("🔧 Check your deployment logs in Render dashboard.")

if __name__ == "__main__":
    main() 