#!/usr/bin/env python3
"""
Test script for Groq API connection and basic functionality
"""

import os
import asyncio
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test basic Groq API connection"""
    try:
        # Get API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            return False
        
        print("✅ GROQ_API_KEY found")
        
        # Test Groq client initialization
        client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully")
        
        # Test simple query
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Hello! Can you respond with 'Groq API is working'?"}
            ],
            temperature=0.1
        )
        
        print(f"✅ Groq API response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Groq API: {e}")
        return False

async def test_async_groq():
    """Test async Groq API functionality"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)
        
        # Test async-like functionality
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": "Test async functionality"}
            ],
            temperature=0.1
        )
        
        print(f"✅ Async Groq API response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing async Groq API: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Groq API Connection...")
    print("=" * 50)
    
    # Test basic connection
    basic_test = test_groq_connection()
    
    # Test async functionality
    async_test = asyncio.run(test_async_groq())
    
    print("=" * 50)
    if basic_test and async_test:
        print("🎉 All tests passed! Groq API is working correctly.")
    else:
        print("❌ Some tests failed. Please check your API key and network connection.")

if __name__ == "__main__":
    main() 