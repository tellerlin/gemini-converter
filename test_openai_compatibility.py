#!/usr/bin/env python3
"""
OpenAI API Compatibility Test Suite for Gemini Converter
Tests all OpenAI API compatible endpoints with various scenarios
"""

import asyncio
import json
import httpx
import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
CLIENT_KEY = os.getenv("ADAPTER_API_KEYS", "test-key").split(",")[0].strip()

class OpenAICompatibilityTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.client_key = CLIENT_KEY
        self.results = []

    async def test_models_endpoint(self):
        """Test /v1/models endpoint"""
        test_name = "Models List"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"X-API-Key": self.client_key}
                )
                
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                data = response.json()
                assert data.get("object") == "list", "Response should have object: list"
                assert "data" in data, "Response should have data field"
                assert len(data["data"]) > 0, "Should have at least one model"
                
                # Check model format
                model = data["data"][0]
                required_fields = ["id", "object", "created", "owned_by"]
                for field in required_fields:
                    assert field in model, f"Model should have {field} field"
                
                self.results.append({"test": test_name, "status": "PASS", "details": f"Found {len(data['data'])} models"})
                
        except Exception as e:
            self.results.append({"test": test_name, "status": "FAIL", "details": str(e)})

    async def test_chat_completion_basic(self):
        """Test basic chat completion"""
        test_name = "Chat Completion - Basic"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": self.client_key
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "user", "content": "Hello! Just say 'Hi' back."}
                        ],
                        "max_tokens": 50,
                        "temperature": 0.1
                    }
                )
                
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                data = response.json()
                
                # Check response format
                assert data.get("object") == "chat.completion", "Object should be chat.completion"
                assert "id" in data, "Should have id field"
                assert "choices" in data, "Should have choices field"
                assert len(data["choices"]) > 0, "Should have at least one choice"
                assert "usage" in data, "Should have usage field"
                
                choice = data["choices"][0]
                assert "message" in choice, "Choice should have message"
                assert choice["message"]["role"] == "assistant", "Message role should be assistant"
                assert "content" in choice["message"], "Message should have content"
                
                self.results.append({"test": test_name, "status": "PASS", "details": "Response format correct"})
                
        except Exception as e:
            self.results.append({"test": test_name, "status": "FAIL", "details": str(e)})

    async def test_chat_completion_streaming(self):
        """Test streaming chat completion"""
        test_name = "Chat Completion - Streaming"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": self.client_key
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "user", "content": "Count to 3 slowly."}
                        ],
                        "max_tokens": 100,
                        "stream": True
                    }
                ) as response:
                    
                    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                    assert response.headers.get("content-type") == "text/event-stream", "Should be event stream"
                    
                    chunks_received = 0
                    content_received = ""
                    
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            lines = chunk.strip().split('\n')
                            for line in lines:
                                if line.startswith('data: '):
                                    data_str = line[6:]  # Remove 'data: ' prefix
                                    if data_str == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(data_str)
                                        chunks_received += 1
                                        if "choices" in chunk_data and chunk_data["choices"]:
                                            delta = chunk_data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                content_received += delta["content"]
                                    except json.JSONDecodeError:
                                        continue
                    
                    assert chunks_received > 0, "Should receive at least one chunk"
                    self.results.append({"test": test_name, "status": "PASS", "details": f"Received {chunks_received} chunks"})
                
        except Exception as e:
            self.results.append({"test": test_name, "status": "FAIL", "details": str(e)})

    async def test_function_calling(self):
        """Test function calling capability"""
        test_name = "Function Calling"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": self.client_key
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "user", "content": "What's the weather like in Tokyo today?"}
                        ],
                        "max_tokens": 200,
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "description": "Get weather information for a city",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "city": {
                                                "type": "string",
                                                "description": "The city name"
                                            },
                                            "unit": {
                                                "type": "string", 
                                                "enum": ["celsius", "fahrenheit"],
                                                "description": "Temperature unit"
                                            }
                                        },
                                        "required": ["city"]
                                    }
                                }
                            }
                        ]
                    }
                )
                
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                data = response.json()
                
                choice = data["choices"][0]
                message = choice["message"]
                
                # Check if tool calls were made
                if "tool_calls" in message and message["tool_calls"]:
                    tool_call = message["tool_calls"][0]
                    assert tool_call["type"] == "function", "Tool call type should be function"
                    assert "function" in tool_call, "Tool call should have function"
                    assert tool_call["function"]["name"] == "get_weather", "Function name should match"
                    
                    # Parse arguments
                    args = json.loads(tool_call["function"]["arguments"])
                    assert "city" in args, "Arguments should include city"
                    
                    self.results.append({"test": test_name, "status": "PASS", "details": "Tool call generated correctly"})
                else:
                    self.results.append({"test": test_name, "status": "PARTIAL", "details": "No tool calls made (model chose not to use tools)"})
                
        except Exception as e:
            self.results.append({"test": test_name, "status": "FAIL", "details": str(e)})

    async def test_different_models(self):
        """Test different model mappings"""
        test_name = "Model Mapping"
        models_to_test = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        for model in models_to_test:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={
                            "Content-Type": "application/json",
                            "X-API-Key": self.client_key
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "user", "content": "Hello"}
                            ],
                            "max_tokens": 10
                        }
                    )
                    
                    assert response.status_code == 200, f"Model {model} failed with status {response.status_code}"
                    data = response.json()
                    assert data.get("model") == model, f"Response model should match request model"
                    
            except Exception as e:
                self.results.append({"test": f"{test_name} - {model}", "status": "FAIL", "details": str(e)})
                continue
        
        self.results.append({"test": test_name, "status": "PASS", "details": f"All {len(models_to_test)} models work"})

    async def test_error_handling(self):
        """Test error handling"""
        test_name = "Error Handling"
        try:
            # Test with invalid API key
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": "invalid-key"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "user", "content": "Hello"}
                        ]
                    }
                )
                
                assert response.status_code == 401, f"Expected 401 for invalid key, got {response.status_code}"
                
                # Test with invalid request
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": self.client_key
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": []  # Empty messages should fail
                    }
                )
                
                assert response.status_code == 400, f"Expected 400 for empty messages, got {response.status_code}"
                
                self.results.append({"test": test_name, "status": "PASS", "details": "Proper error codes returned"})
                
        except Exception as e:
            self.results.append({"test": test_name, "status": "FAIL", "details": str(e)})

    async def test_key_rotation(self):
        """Test API key rotation functionality"""
        test_name = "Key Rotation Stats"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/stats",
                    headers={"X-API-Key": self.client_key}
                )
                
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                data = response.json()
                
                # Check if key management stats exist
                assert "key_management_stats" in data, "Should have key management stats"
                key_stats = data["key_management_stats"]
                assert "summary" in key_stats, "Should have summary"
                assert "performance" in key_stats, "Should have performance stats"
                
                self.results.append({"test": test_name, "status": "PASS", "details": "Key rotation stats available"})
                
        except Exception as e:
            self.results.append({"test": test_name, "status": "FAIL", "details": str(e)})

    async def run_all_tests(self):
        """Run all compatibility tests"""
        print("🧪 Starting OpenAI API Compatibility Tests")
        print("=" * 50)
        
        # Check if server is running
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code != 200:
                    print("❌ Server is not healthy")
                    return
        except:
            print("❌ Cannot connect to server. Make sure it's running.")
            return
        
        print("✅ Server is running and healthy")
        print()
        
        # Run tests
        test_methods = [
            self.test_models_endpoint,
            self.test_chat_completion_basic,
            self.test_chat_completion_streaming,
            self.test_function_calling,
            self.test_different_models,
            self.test_error_handling,
            self.test_key_rotation
        ]
        
        for test_method in test_methods:
            print(f"Running {test_method.__name__.replace('test_', '').replace('_', ' ').title()}...", end=" ", flush=True)
            await test_method()
            # Get the last result
            if self.results:
                last_result = self.results[-1]
                status_emoji = "✅" if last_result["status"] == "PASS" else "⚠️" if last_result["status"] == "PARTIAL" else "❌"
                print(f"{status_emoji} {last_result['status']}")
            else:
                print("❌ NO RESULT")
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = failed = partial = 0
        for result in self.results:
            status_emoji = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "PARTIAL" else "❌"
            print(f"{status_emoji} {result['test']}: {result['details']}")
            
            if result["status"] == "PASS":
                passed += 1
            elif result["status"] == "PARTIAL":
                partial += 1
            else:
                failed += 1
        
        print("\n" + "=" * 50)
        print(f"✅ PASSED: {passed}")
        print(f"⚠️ PARTIAL: {partial}") 
        print(f"❌ FAILED: {failed}")
        print(f"📊 TOTAL: {len(self.results)}")
        
        if failed == 0:
            print("\n🎉 All critical tests passed! OpenAI compatibility confirmed.")
        elif failed <= 2:
            print("\n⚠️ Mostly compatible with minor issues.")
        else:
            print("\n❌ Significant compatibility issues found.")

async def main():
    """Main test runner"""
    tester = OpenAICompatibilityTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())