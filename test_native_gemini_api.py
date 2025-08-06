#!/usr/bin/env python3
# test_native_gemini_api.py - 测试原生Gemini API功能

import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, Any

# 配置
BASE_URL = "http://localhost:8000"
API_KEY = "test-key"  # 请替换为您的实际API密钥

class GeminiAPITester:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key},
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """测试健康检查"""
        print("🔍 Testing health check...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                print(f"✅ Health check: {data['status']} (Active keys: {data['key_summary']['active']})")
                return data["key_summary"]["active"] > 0
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def test_gemini_health_check(self) -> bool:
        """测试Gemini API健康检查"""
        print("🔍 Testing native Gemini health check...")
        try:
            async with self.session.get(f"{self.base_url}/gemini/health") as response:
                data = await response.json()
                print(f"✅ Native Gemini health: {data['status']} (Available keys: {data['availableKeys']})")
                return data["availableKeys"] > 0
        except Exception as e:
            print(f"❌ Native Gemini health check failed: {e}")
            return False
    
    async def test_list_gemini_models(self) -> bool:
        """测试列出Gemini模型"""
        print("🔍 Testing list Gemini models...")
        try:
            async with self.session.get(f"{self.base_url}/gemini/v1beta/models") as response:
                data = await response.json()
                models = data.get("models", [])
                print(f"✅ Found {len(models)} Gemini models:")
                for model in models[:2]:  # 显示前两个
                    print(f"   - {model['name']} ({model['displayName']})")
                return len(models) > 0
        except Exception as e:
            print(f"❌ List Gemini models failed: {e}")
            return False
    
    async def test_gemini_generate_content(self) -> bool:
        """测试Gemini生成内容（非流式）"""
        print("🔍 Testing Gemini generateContent (non-streaming)...")
        try:
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Hello! Please respond with a simple greeting."}]
                    }
                ],
                "generation_config": {
                    "temperature": 0.7,
                    "max_output_tokens": 100
                }
            }
            
            model = "gemini-1.5-flash-latest"
            async with self.session.post(
                f"{self.base_url}/gemini/v1beta/models/{model}:generateContent",
                json=request_data
            ) as response:
                data = await response.json()
                
                if "candidates" in data and data["candidates"]:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and "text" in parts[0]:
                            text = parts[0]["text"]
                            print(f"✅ Generated content: {text[:100]}...")
                            
                            # 显示使用统计
                            if "usageMetadata" in data:
                                usage = data["usageMetadata"]
                                print(f"   Tokens: prompt={usage.get('promptTokenCount', 0)}, "
                                      f"completion={usage.get('candidatesTokenCount', 0)}, "
                                      f"total={usage.get('totalTokenCount', 0)}")
                            
                            return True
                
                print(f"❌ Unexpected response format: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return False
                
        except Exception as e:
            print(f"❌ Gemini generateContent failed: {e}")
            return False
    
    async def test_gemini_stream_generate_content(self) -> bool:
        """测试Gemini流式生成内容"""
        print("🔍 Testing Gemini streamGenerateContent (streaming)...")
        try:
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Please count from 1 to 5, one number per response chunk."}]
                    }
                ],
                "generation_config": {
                    "temperature": 0.1,
                    "max_output_tokens": 50
                }
            }
            
            model = "gemini-1.5-flash-latest"
            chunks_received = 0
            content_parts = []
            
            async with self.session.post(
                f"{self.base_url}/gemini/v1beta/models/{model}:streamGenerateContent",
                json=request_data
            ) as response:
                
                if response.content_type != "application/json":
                    print(f"❌ Wrong content type: {response.content_type}")
                    return False
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        try:
                            chunk_data = json.loads(line)
                            chunks_received += 1
                            
                            if "candidates" in chunk_data and chunk_data["candidates"]:
                                candidate = chunk_data["candidates"][0]
                                if "content" in candidate and "parts" in candidate["content"]:
                                    parts = candidate["content"]["parts"]
                                    for part in parts:
                                        if "text" in part:
                                            content_parts.append(part["text"])
                            
                            if chunks_received <= 3:  # 显示前几个块
                                print(f"   Chunk {chunks_received}: {str(chunk_data)[:100]}...")
                                
                        except json.JSONDecodeError:
                            print(f"   Invalid JSON chunk: {line[:50]}...")
                
            full_content = "".join(content_parts)
            print(f"✅ Stream completed: {chunks_received} chunks, content: {full_content[:100]}...")
            return chunks_received > 0
            
        except Exception as e:
            print(f"❌ Gemini streamGenerateContent failed: {e}")
            return False
    
    async def test_gemini_with_tools(self) -> bool:
        """测试Gemini工具调用功能"""
        print("🔍 Testing Gemini with function calling...")
        try:
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "What's the weather like in Beijing? Please use the get_weather function."}]
                    }
                ],
                "tools": [
                    {
                        "function_declarations": [
                            {
                                "name": "get_weather",
                                "description": "Get current weather for a location",
                                "parameters": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "location": {
                                            "type": "STRING",
                                            "description": "The location to get weather for"
                                        },
                                        "unit": {
                                            "type": "STRING",
                                            "enum": ["celsius", "fahrenheit"],
                                            "description": "Temperature unit"
                                        }
                                    },
                                    "required": ["location"]
                                }
                            }
                        ]
                    }
                ],
                "tool_config": {
                    "function_calling_config": {"mode": "AUTO"}
                },
                "generation_config": {
                    "temperature": 0.1
                }
            }
            
            model = "gemini-1.5-pro-latest"
            async with self.session.post(
                f"{self.base_url}/gemini/v1beta/models/{model}:generateContent",
                json=request_data
            ) as response:
                data = await response.json()
                
                if "candidates" in data and data["candidates"]:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        
                        # 查找函数调用
                        function_calls = [part for part in parts if "functionCall" in part]
                        if function_calls:
                            fc = function_calls[0]["functionCall"]
                            print(f"✅ Function call detected: {fc['name']} with args: {fc.get('args', {})}")
                            return True
                        else:
                            # 可能是文本响应
                            text_parts = [part for part in parts if "text" in part]
                            if text_parts:
                                print(f"✅ Text response (no function call): {text_parts[0]['text'][:100]}...")
                                return True
                
                print(f"❌ Unexpected response format: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return False
                
        except Exception as e:
            print(f"❌ Gemini function calling failed: {e}")
            return False

    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 Starting Native Gemini API Tests...")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Native Gemini Health Check", self.test_gemini_health_check),
            ("List Gemini Models", self.test_list_gemini_models),
            ("Generate Content (Non-streaming)", self.test_gemini_generate_content),
            ("Generate Content (Streaming)", self.test_gemini_stream_generate_content),
            ("Function Calling", self.test_gemini_with_tools)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 40)
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: ERROR - {e}")
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Native Gemini API is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the logs above.")
        
        return passed == total


async def main():
    """主函数"""
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        print("⚠️  Warning: Using default API key. Pass your API key as argument:")
        print("   python test_native_gemini_api.py YOUR_API_KEY")
        api_key = API_KEY
    
    async with GeminiAPITester(BASE_URL, api_key) as tester:
        success = await tester.run_all_tests()
        return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        sys.exit(1)