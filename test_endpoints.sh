#!/bin/bash

# ==============================================================================
# OpenAI-Compatible Gemini Adapter v4.0.0 - 功能测试脚本
#
# 说明:
# 1. 替换下面的 CLIENT_KEY 和 ADMIN_KEY 的占位符值。
# 2. 赋予脚本执行权限: chmod +x test_endpoints.sh
# 3. 运行脚本: ./test_endpoints.sh
# ==============================================================================

# --- 配置 ---
# ‼️ 重要: 请将它们替换为你 .env 文件中的真实密钥。
export CLIENT_KEY="your-client-key"
export ADMIN_KEY="your-admin-key"

# 正在运行的服务的基础 URL
BASE_URL="http://localhost:8000"

# --- 脚本逻辑 (无需修改以下内容) ---

# 打印格式化头部的函数
print_header() {
    echo ""
    echo "============================================================"
    echo "▶️  测试: $1"
    echo "============================================================"
}

# 检查密钥是否为占位符
if [ "$CLIENT_KEY" = "your-client-key" ] || [ "$ADMIN_KEY" = "your-admin-key" ]; then
    echo "❌ 错误: 请编辑此脚本并替换占位符 API 密钥。"
    exit 1
fi

# --- 1. 服务状态和公共端点 ---
print_header "服务状态 (无需认证)"
echo "--> 正在检查 Docker 容器状态..."
docker-compose ps
echo -e "\n--> 正在 Ping 根端点..."
curl -s ${BASE_URL}/ | jq
echo -e "\n--> 正在 Ping 健康检查端点..."
curl -s ${BASE_URL}/health | jq

# --- 2. 核心 API 功能 (需要客户端密钥) ---
print_header "核心 API 功能 (需要客户端密钥)"
echo "--> 正在列出可用模型 (OpenAI 格式)..."
curl -s -H "Authorization: Bearer $CLIENT_KEY" "${BASE_URL}/v1/models" | jq

echo -e "\n--> 简单的问答测试 (非流式)..."
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "你好！用一句话简单解释一下什么是人工智能。"}]
  }' | jq

echo -e "\n--> 流式响应测试..."
curl -s -N -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "写一首关于宇宙的五行短诗。"}],
    "stream": true
  }'

echo -e "\n\n--> 工具调用 (Function Calling) 测试 (非流式)..."
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CLIENT_KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "上海今天的天气怎么样？"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' | jq

# --- 3. 监控和统计端点 (需要客户端密钥) ---
print_header "监控和统计端点 (需要客户端密钥)"
echo "--> 正在获取密钥使用统计..."
curl -s -H "Authorization: Bearer $CLIENT_KEY" "${BASE_URL}/stats" | jq
echo -e "\n--> 正在获取性能指标..."
curl -s -H "Authorization: Bearer $CLIENT_KEY" "${BASE_URL}/metrics" | jq

# --- 4. 管理端点 (需要管理密钥) ---
print_header "管理端点 (需要管理密钥)"
echo "--> 正在尝试重置一个不存在的密钥 (应该返回 404)..."
curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/admin/reset-key/fakekeyprefix" -H "X-API-Key: $ADMIN_KEY"
echo ""

echo ""
print_header "所有测试已完成!"
echo "如果你看到每个测试都有 JSON 输出且没有 curl 错误，那么你的部署就成功了。"
echo "============================================================"
echo ""
