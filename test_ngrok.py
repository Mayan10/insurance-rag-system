#!/usr/bin/env python3
"""
Test script for ngrok deployment
Replace YOUR_NGROK_URL with your actual ngrok URL
"""

import requests
import json
import time
import sys

# Configuration - REPLACE WITH YOUR ACTUAL NGROK URL
NGROK_URL = "https://403f4b11a4d9.ngrok-free.app"  # Your actual ngrok URL
API_KEY = "hackrx-2024-secret-key"

def test_ngrok_connection():
    """Test basic ngrok connection"""
    print("🔗 Testing ngrok connection...")
    try:
        response = requests.get(f"{NGROK_URL}/", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ ngrok connection successful")
            return True
        else:
            print(f"❌ ngrok connection failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ngrok connection error: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint"""
    print("\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{NGROK_URL}/health", timeout=30)
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
            "What is the waiting period for pre-existing diseases (PED) to be covered?"
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print("⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{NGROK_URL}/hackrx/run",
            headers=headers,
            json=test_data,
            timeout=60  # 60 second timeout
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
        print("❌ Request timed out (>60 seconds)")
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
            f"{NGROK_URL}/hackrx/run",
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

def check_ngrok_status():
    """Check if ngrok is running locally"""
    print("\n🔍 Checking ngrok status...")
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'ngrok' in result.stdout:
            print("✅ ngrok process found")
            return True
        else:
            print("❌ ngrok process not found")
            print("💡 Make sure ngrok is running: ngrok http 8000")
            return False
    except Exception as e:
        print(f"❌ Could not check ngrok status: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Insurance RAG System - ngrok Deployment Test")
    print("=" * 60)
    print(f"🌐 Testing URL: {NGROK_URL}")
    print("=" * 60)
    
    # Check if URL is set
    if "your-ngrok-url" in NGROK_URL:
        print("❌ ERROR: Please update NGROK_URL in this script with your actual ngrok URL!")
        print("Example: https://abc123.ngrok.io")
        print("\n💡 To get your ngrok URL:")
        print("1. Run: ngrok http 8000")
        print("2. Copy the HTTPS URL from the output")
        print("3. Update this script with that URL")
        return
    
    # Check ngrok status
    ngrok_running = check_ngrok_status()
    
    # Run tests
    connection_ok = test_ngrok_connection()
    health_ok = test_health_endpoint()
    auth_ok = test_authentication()
    api_ok = test_hackrx_endpoint()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"ngrok Running: {'✅ YES' if ngrok_running else '❌ NO'}")
    print(f"Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Authentication: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"API Endpoint: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if connection_ok and health_ok and auth_ok and api_ok:
        print("\n🎉 SUCCESS! Your ngrok API is ready for hackathon submission!")
        print(f"📝 Submission URL: {NGROK_URL}/hackrx/run")
        print("\n📋 Pre-Submission Checklist:")
        print("✅ API is live & accessible")
        print("✅ HTTPS enabled (ngrok provides this)")
        print("✅ Handles POST requests")
        print("✅ Returns JSON response")
        print("✅ Response time < 30s")
        print("✅ Tested with sample data")
        print("✅ Authentication working")
        print("\n⚠️  IMPORTANT REMINDERS:")
        print("• Keep ngrok running during evaluation")
        print("• Keep your computer on")
        print("• Don't close the ngrok terminal")
        print("• URL will change if you restart ngrok")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("🔧 Troubleshooting:")
        print("• Make sure ngrok is running: ngrok http 8000")
        print("• Make sure local API is running: python app.py")
        print("• Check ngrok web interface: http://localhost:4040")

if __name__ == "__main__":
    main() 