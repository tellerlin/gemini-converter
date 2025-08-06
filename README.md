# OpenAI 兼容 Gemini 适配器 + 原生 Gemini API

一个高性能反向代理服务，提供双重 API 接口：将 Google Gemini API 转换为完全兼容 OpenAI API 的接口，同时支持原生 Gemini API 格式。内置智能密钥轮询、自动故障转移和冷却机制，提供稳定可靠的大模型接入服务。

## ✨ 主要功能

### 🔄 双重 API 支持
- 🤖 **完整 OpenAI API 兼容** - 支持 `/v1/chat/completions`、流式响应、工具调用
- 🆕 **原生 Gemini API 接口** - 支持 `/gemini/v1beta/models/{model}:generateContent` 和流式响应
- 🔧 **统一密钥管理** - 两套 API 使用相同的密钥池和轮询机制

### 🔑 智能管理
- 🔄 **智能密钥管理** - 自动轮询多个 API 密钥，失败自动切换
- ⚡ **高性能响应** - 真正的流式输出，企业级缓存机制
- 🛡️ **安全认证** - 客户端密钥和管理密钥双重保护
- 📊 **实时监控** - 密钥状态、性能指标、健康检查
- 🐳 **Docker 部署** - 一键启动，无需复杂配置

## 📋 安装要求

- **Docker** 和 **Docker Compose** 
- **Google Gemini API 密钥** ([申请地址](https://aistudio.google.com/app/apikey))
- **Git** 用于获取源码

## 🚀 快速安装

### 1. 获取源码

```bash
git clone https://github.com/tellerlin/gemini-converter.git
cd gemini-converter
```

### 2. 配置密钥

复制配置模板并编辑：

```bash
cp .env.example .env
nano .env  # 或使用你喜欢的编辑器
```

**必须配置的参数：**

```bash
# Gemini API 密钥（多个用逗号分隔）
GEMINI_API_KEYS=your-gemini-key1,your-gemini-key2

# 客户端访问密钥（建议使用 openssl rand -hex 32 生成）
SECURITY_ADAPTER_API_KEYS=your-secure-client-key

# 管理员密钥（可选，用于管理功能）
SECURITY_ADMIN_API_KEYS=your-secure-admin-key
```

### 3. 启动服务

```bash
docker-compose up -d --build
```

### 4. 验证运行

```bash
# 检查容器状态
docker-compose ps

# 查看启动日志
docker-compose logs --tail 20

# 测试服务
curl http://localhost:8000/health
```

如果看到类似 `"status": "healthy"` 的响应，说明服务已成功启动。

## 🔄 更新与维护

### 更新到最新版本

```bash
# 停止当前服务
docker-compose down

# 获取最新代码
git pull

# 重新构建并启动
docker-compose up -d --build
```

### 日常维护

```bash
# 查看服务状态
docker-compose ps

# 查看实时日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 查看资源使用情况
docker stats
```

### 备份配置

建议定期备份你的 `.env` 配置文件：

```bash
cp .env .env.backup.$(date +%Y%m%d)
```

## 🧪 功能测试

使用内置测试脚本验证所有功能：

### OpenAI 兼容接口测试

```bash
# 1. 配置测试脚本中的密钥
nano test_endpoints.sh
# 将 CLIENT_KEY 和 ADMIN_KEY 替换为你 .env 文件中的实际密钥

# 2. 赋予执行权限
chmod +x test_endpoints.sh

# 3. 运行测试
./test_endpoints.sh
```

### 原生 Gemini API 测试

```bash
# 1. 运行原生 Gemini API 测试
python3 test_native_gemini_api.py your-client-key

# 2. 或者使用语法测试（不需要真实密钥）
python3 syntax_test.py
```

### 测试内容

测试脚本会验证以下功能：

**OpenAI 兼容接口：**
- ✅ **服务状态** - 健康检查和基础连接
- ✅ **API 兼容性** - OpenAI 格式的模型列表和聊天接口  
- ✅ **流式响应** - 实时数据流输出
- ✅ **工具调用** - Function Calling 功能
- ✅ **监控端点** - 统计数据和性能指标
- ✅ **管理功能** - 密钥管理和重置

**原生 Gemini API：**
- ✅ **模型列表** - 获取 Gemini 原生格式的模型信息
- ✅ **内容生成** - 非流式 generateContent 接口
- ✅ **流式生成** - 流式 streamGenerateContent 接口  
- ✅ **工具调用** - Gemini 原生函数调用格式
- ✅ **错误处理** - 统一的错误处理和重试机制

### 手动验证

也可以手动测试关键功能：

**OpenAI 兼容接口：**
```bash
# 健康检查
curl http://localhost:8000/health

# 获取模型列表（需要客户端密钥）
curl -H "Authorization: Bearer your-client-key" \
     http://localhost:8000/v1/models

# 测试聊天功能（需要客户端密钥）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-client-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**原生 Gemini API 接口：**
```bash
# 获取 Gemini 模型列表
curl -H "X-API-Key: your-client-key" \
     http://localhost:8000/gemini/v1beta/models

# Gemini 健康检查
curl http://localhost:8000/gemini/health

# 测试 Gemini generateContent（非流式）
curl -X POST "http://localhost:8000/gemini/v1beta/models/gemini-1.5-flash-latest:generateContent" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-client-key" \
  -d '{
    "contents": [
      {
        "role": "user", 
        "parts": [{"text": "Hello! Please introduce yourself."}]
      }
    ],
    "generation_config": {
      "temperature": 0.7,
      "max_output_tokens": 100
    }
  }'

# 测试 Gemini streamGenerateContent（流式）
curl -X POST "http://localhost:8000/gemini/v1beta/models/gemini-1.5-flash-latest:streamGenerateContent" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-client-key" \
  -d '{
    "contents": [
      {
        "role": "user", 
        "parts": [{"text": "Count from 1 to 5 slowly."}]
      }
    ],
    "generation_config": {
      "temperature": 0.1
    }
  }'
```

## 📱 使用方法

### OpenAI 兼容接口

将任何支持 OpenAI API 的客户端指向你的适配器：

**配置参数：**
- **API 地址**: `http://your-server:8000/v1`  
- **API 密钥**: 你的 `SECURITY_ADAPTER_API_KEYS` 值
- **模型**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4o` 等

### 原生 Gemini API 接口

对于需要使用 Gemini 原生格式的应用：

**配置参数：**
- **API 地址**: `http://your-server:8000/gemini/v1beta`  
- **API 密钥**: 你的 `SECURITY_ADAPTER_API_KEYS` 值
- **模型**: `gemini-1.5-pro-latest`, `gemini-1.5-flash-latest` 等

### 常见客户端

| 客户端 | API Base URL 设置位置 |
|--------|---------------------|
| **Cursor** | Settings → Models → Override OpenAI Base URL |
| **JetBrains IDE** | AI Assistant → OpenAI → Custom server URL |
| **Open-WebUI** | Settings → Connections → OpenAI API |
| **ChatGPT Next Web** | Settings → API Host |

### 编程接口

使用任何 OpenAI 客户端库，只需修改 base_url：

**Python (openai 库) - OpenAI 兼容**
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-client-key"  # 你的 SECURITY_ADAPTER_API_KEYS
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Python - 原生 Gemini API 格式**
```python
import requests
import json

def call_gemini_api(content: str, model: str = "gemini-1.5-flash-latest"):
    url = f"http://localhost:8000/gemini/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-client-key"  # 你的 SECURITY_ADAPTER_API_KEYS
    }
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": content}]
            }
        ],
        "generation_config": {
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 使用示例
result = call_gemini_api("Hello! How are you?")
if result.get("candidates"):
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    print(text)
```

**Python - 原生 Gemini 流式响应**
```python
import requests
import json

def stream_gemini_api(content: str, model: str = "gemini-1.5-flash-latest"):
    url = f"http://localhost:8000/gemini/v1beta/models/{model}:streamGenerateContent"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-client-key"
    }
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": content}]
            }
        ],
        "generation_config": {
            "temperature": 0.7
        }
    }
    
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = json.loads(line.decode('utf-8'))
                    if chunk_data.get("candidates"):
                        candidate = chunk_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    print(part["text"], end="", flush=True)
                except json.JSONDecodeError:
                    continue
        print()  # 换行

# 使用示例
stream_gemini_api("Please count from 1 to 10.")
```

**JavaScript/Node.js - OpenAI 兼容**
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:8000/v1',
    apiKey: 'your-client-key'  // 你的 SECURITY_ADAPTER_API_KEYS
});

const response = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: 'Hello!' }]
});
```

**JavaScript - 原生 Gemini API 格式**
```javascript
async function callGeminiAPI(content, model = 'gemini-1.5-flash-latest') {
    const url = `http://localhost:8000/gemini/v1beta/models/${model}:generateContent`;
    const headers = {
        'Content-Type': 'application/json',
        'X-API-Key': 'your-client-key'  // 你的 SECURITY_ADAPTER_API_KEYS
    };
    const data = {
        contents: [
            {
                role: 'user',
                parts: [{ text: content }]
            }
        ],
        generation_config: {
            temperature: 0.7,
            max_output_tokens: 1000
        }
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(data)
    });
    
    return await response.json();
}

// 使用示例
const result = await callGeminiAPI("Hello! How are you?");
if (result.candidates && result.candidates.length > 0) {
    const text = result.candidates[0].content.parts[0].text;
    console.log(text);
}
```

### 支持功能

**OpenAI 兼容接口：**
- ✅ **聊天对话** - `/v1/chat/completions`
- ✅ **流式响应** - `stream: true`
- ✅ **工具调用** - Function Calling
- ✅ **模型列表** - `/v1/models`
- ✅ **JSON 模式** - `response_format: {"type": "json_object"}`

**原生 Gemini API 接口：**
- ✅ **内容生成** - `/gemini/v1beta/models/{model}:generateContent`
- ✅ **流式生成** - `/gemini/v1beta/models/{model}:streamGenerateContent`
- ✅ **模型列表** - `/gemini/v1beta/models`
- ✅ **函数调用** - Native Gemini function calling format
- ✅ **安全设置** - Safety settings and content filtering
- ✅ **生成配置** - Temperature, top_p, top_k, max_output_tokens

## 🔧 故障排查

### 常见问题

**服务无法启动**
```bash
# 检查端口是否被占用
lsof -i :8000

# 查看详细错误日志
docker-compose logs gemini-converter-adapter

# 检查配置文件
cat .env | grep -v "^#"
```

**API 密钥问题**
```bash
# 验证 Gemini API 密钥
docker-compose run --rm gemini-converter-adapter python api_key_checker.py

# 检查密钥格式（不应包含引号或空格）
grep "GEMINI_API_KEYS" .env
```

**连接超时或请求失败**
```bash
# 检查网络连接
curl -I https://generativelanguage.googleapis.com

# 查看实时请求日志
docker-compose logs -f --tail 100

# 检查服务状态
curl http://localhost:8000/health
```

### 诊断工具

**健康检查脚本**
```bash
# 运行完整诊断
docker-compose run --rm gemini-converter-adapter python diagnose_script.py
```

**查看统计信息**
```bash
# 密钥使用情况（需要客户端密钥）
curl -H "Authorization: Bearer your-client-key" \
     http://localhost:8000/stats

# 系统指标
curl -H "Authorization: Bearer your-client-key" \
     http://localhost:8000/metrics
```

### 性能优化

**调整并发数**
```bash
# 在 .env 文件中设置
SERVICE_WORKERS=4  # 根据 CPU 核心数调整
```

**启用缓存**
```bash
# 在 .env 文件中设置
CACHE_ENABLED=True
CACHE_MAX_SIZE=1000
CACHE_TTL=300
```

## 📡 API 端点

### OpenAI 兼容接口

| 端点 | 用途 | 认证要求 |
|------|------|----------|
| `GET /health` | 服务健康检查 | 无 |
| `GET /v1/models` | 获取可用模型列表（OpenAI格式） | 客户端密钥 |
| `POST /v1/chat/completions` | 聊天对话接口（OpenAI格式） | 客户端密钥 |
| `GET /stats` | 查看使用统计 | 客户端密钥 |
| `GET /metrics` | 性能指标 | 客户端密钥 |

### 原生 Gemini API 接口

| 端点 | 用途 | 认证要求 |
|------|------|----------|
| `GET /gemini/health` | Gemini API 健康检查 | 无 |
| `GET /gemini/v1beta/models` | 获取 Gemini 模型列表（原生格式） | 客户端密钥 |
| `POST /gemini/v1beta/models/{model}:generateContent` | 内容生成（非流式，原生格式） | 客户端密钥 |
| `POST /gemini/v1beta/models/{model}:streamGenerateContent` | 内容生成（流式，原生格式） | 客户端密钥 |

### 管理接口

| 端点 | 用途 | 认证要求 |
|------|------|----------|
| `POST /admin/keys` | 动态添加 API 密钥 | 管理员密钥 |
| `DELETE /admin/keys` | 动态移除 API 密钥 | 管理员密钥 |
| `PUT /admin/keys/{key_id}` | 更新密钥状态 | 管理员密钥 |

### 认证方式

**客户端密钥** - 支持两种格式：
```bash
# Bearer Token 格式（推荐用于 OpenAI 兼容接口）
curl -H "Authorization: Bearer your-client-key" http://localhost:8000/v1/models

# X-API-Key Header 格式（推荐用于 Gemini 原生接口）
curl -H "X-API-Key: your-client-key" http://localhost:8000/gemini/v1beta/models
```

**管理员密钥** (X-API-Key Header)
```bash
curl -H "X-API-Key: your-admin-key" http://localhost:8000/admin/keys
```

### Swagger/OpenAPI 文档

服务启动后，可访问以下地址查看完整的 API 文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

这些文档包含了所有端点的详细说明、请求/响应格式和交互式测试功能。

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📞 支持

如有问题，请在 GitHub 上创建 [Issue](https://github.com/tellerlin/gemini-converter/issues)

