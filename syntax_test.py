#!/usr/bin/env python3
# syntax_test.py - 语法和导入测试（不需要API密钥）

import sys
import os

def test_imports():
    """测试模块导入"""
    print("🔍 Testing module imports...")
    
    try:
        # 测试gemini_adapter导入
        sys.path.append('.')
        from src.gemini_adapter import (
            NativeGeminiAdapter, 
            GeminiGenerateContentRequest, 
            GeminiStreamGenerateContentRequest,
            GeminiContent,
            GeminiGenerationConfig
        )
        print("✅ gemini_adapter imports successful")
        
        # 测试pydantic模型创建
        content = GeminiContent(
            role="user",
            parts=[{"text": "Hello"}]
        )
        print(f"✅ GeminiContent model created: role={content.role}, parts_count={len(content.parts)}")
        
        # 测试请求模型创建
        request = GeminiGenerateContentRequest(
            contents=[content],
            generation_config=GeminiGenerationConfig(temperature=0.7)
        )
        print(f"✅ GeminiGenerateContentRequest created: contents_count={len(request.contents)}")
        
        # 测试适配器类
        print("✅ NativeGeminiAdapter class available")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_model_validation():
    """测试Pydantic模型验证"""
    print("\n🔍 Testing model validation...")
    
    try:
        from src.gemini_adapter import GeminiGenerateContentRequest, GeminiContent
        
        # 测试正确的请求
        valid_request = GeminiGenerateContentRequest(
            contents=[
                GeminiContent(role="user", parts=[{"text": "Hello"}])
            ]
        )
        print("✅ Valid request created successfully")
        
        # 测试错误的请求（应该失败）
        try:
            invalid_request = GeminiGenerateContentRequest(contents=[])
            print("❌ Invalid request should have failed but didn't")
            return False
        except Exception:
            print("✅ Invalid request correctly rejected")
        
        return True
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False

def test_main_syntax():
    """测试main.py语法（不初始化配置）"""
    print("\n🔍 Testing main.py syntax...")
    
    try:
        # 暂时设置虚拟环境变量以通过配置验证
        os.environ['GEMINI_API_KEYS'] = 'test-key-for-syntax-check'
        os.environ['ADAPTER_API_KEYS'] = 'test-adapter-key'
        
        # 只导入必要的函数，不初始化整个应用
        from src.main import (
            verify_api_key, 
            verify_admin_key,
            GeminiKeyManager,
            OAIStyleGeminiAdapter
        )
        print("✅ main.py imports successful")
        
        # 清理环境变量
        if 'GEMINI_API_KEYS' in os.environ:
            del os.environ['GEMINI_API_KEYS']
        if 'ADAPTER_API_KEYS' in os.environ:
            del os.environ['ADAPTER_API_KEYS']
        
        return True
    except Exception as e:
        print(f"❌ main.py syntax test failed: {e}")
        return False

def test_openai_adapter_syntax():
    """测试openai_adapter语法"""
    print("\n🔍 Testing openai_adapter syntax...")
    
    try:
        from src.openai_adapter import (
            ChatCompletionRequest,
            ChatMessage,
            APIConfig
        )
        print("✅ openai_adapter imports successful")
        
        # 测试创建简单的请求
        message = ChatMessage(role="user", content="Hello")
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[message]
        )
        print("✅ ChatCompletionRequest created successfully")
        
        return True
    except Exception as e:
        print(f"❌ openai_adapter syntax test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Starting Syntax and Import Tests...")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Model Validation", test_model_validation),
        ("OpenAI Adapter Syntax", test_openai_adapter_syntax),
        ("Main Module Syntax", test_main_syntax),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results[test_name] = result
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
        print("🎉 All syntax tests passed! Code structure is correct.")
        return True
    else:
        print("⚠️  Some syntax tests failed. Please check the code.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        sys.exit(1)