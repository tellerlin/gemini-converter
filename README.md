# 高性能 OpenAI 兼容 Gemini 适配器 v4.0.0

一个强大的反向代理服务，它将 Google 的 Gemini API 转换为与 OpenAI API 完全兼容的接口。它内置了智能的 Gemini API 密钥轮询、自动故障转移和冷却机制，为你的应用提供高可用的、统一的大模型接入层。

---

## ✨ 核心特性

- 🤖 **完整 OpenAI API 兼容性**: 完全支持 OpenAI 的 `/v1/chat/completions` 接口，包括完整的流式响应和 JSON 输出。
- 🔑 **智能密钥管理**: 自动轮询多个 Gemini API 密钥。失败的密钥会立即进入冷却期，并自动切换到下一个可用密钥，确保服务不中断。
- 🛠️ **支持工具调用 (Function Calling)**: 无缝转换 OpenAI 和 Gemini 之间的工具调用格式，在流式和非流式模式下均可正常工作。
- ⚡ **真·流式响应**: 以字节级精度模拟 OpenAI 的流式响应（Server-Sent Events），提供最佳的实时体验。
- 🛡️ **企业级安全**: 所有敏感端点均强制要求 API 密钥认证，支持客户端密钥和独立的管理密钥。
- 📊 **实时监控**: 提供服务健康、密钥状态、请求性能和缓存命中率的实时监控端点。
- 🐳 **一键 Docker 部署**: 使用 Docker 和 Docker Compose，提供最简化的安装部署流程。

## 🚀 快速开始

本指南将引导你以最简单的方式启动并运行服务。

### 先决条件

- **Docker** 和 **Docker Compose** 已安装。
- **Google Gemini API 密钥** ([在此处获取](https://aistudio.google.com/app/apikey))。
- **Git** 用于克隆代码仓库。

### 步骤 1: 克隆仓库

```bash
git clone [https://github.com/tellerlin/gemini-claude.git](https://github.com/tellerlin/gemini-claude.git)
cd gemini-claude
````

### 步骤 2: 配置 API 密钥

创建你的 `.env` 配置文件并填入必要的 API 密钥。

```bash
# 从示例文件创建你的配置文件
cp .env.example .env

# 使用编辑器打开 .env 文件 (例如 nano)
nano .env
```

在 `.env` 文件中, 你 **必须** 设置以下两个值:

  - `GEMINI_API_KEYS`: 你的一个或多个 Google Gemini API 密钥，用逗号分隔。
  - `SECURITY_ADAPTER_API_KEYS`: 用于保护你的适配器服务的客户端密钥。你的应用程序将使用此密钥进行认证。建议使用 `openssl rand -hex 32` 生成一个。

### 步骤 3: 构建和部署

这个命令会构建 Docker 镜像并在后台启动服务。

```bash
docker-compose up -d --build
```

### 步骤 4: 验证部署

检查容器是否正在运行，并查看日志确认启动成功。

```bash
# 检查容器状态 (应该显示 'running')
docker-compose ps

# 查看最新日志，确保没有错误
docker-compose logs --tail 100
```

如果日志显示 "OpenAI-Style Gemini Adapter started successfully."，则服务已成功启动并准备就绪！ 服务地址为 `http://localhost:8000`。

## 🧪 测试你的部署

项目包含一个全面的测试脚本 `test_endpoints.sh` 来验证所有核心功能。

### 1\. 在脚本中设置密钥

打开测试脚本:

```bash
nano test_endpoints.sh
```

在脚本顶部，将 `CLIENT_KEY` 和 `ADMIN_KEY` 的占位符替换为你在 `.env` 文件中设置的真实密钥。

### 2\. 赋予脚本执行权限

此命令只需运行一次：

```bash
chmod +x test_endpoints.sh
```

### 3\. 运行测试

```bash
# 确保你的适配器服务正在运行！
./test_endpoints.sh
```

脚本将对你的服务运行一系列测试，包括文本生成、流式响应和工具调用。如果所有测试都通过并输出了漂亮的 JSON，那么你的部署就成功了！

## ⚙️ 客户端配置

### 通用客户端配置 (例如 Cursor, JetBrains IDE Copilot, Open-WebUI)

要将任何支持 OpenAI API 的客户端连接到你的适配器：

1.  **API 地址 (API Base URL)**: 填入你的适配器地址：`http://<你的服务器IP>:8000/v1`
2.  **API 密钥 (API Key)**: 填入你在 `.env` 文件中设置的 `SECURITY_ADAPTER_API_KEYS` 的值。
3.  **模型名称**: 选择一个兼容的模型，例如 `gpt-4o` 或 `gpt-3.5-turbo`。

### 代码中使用 (Python aiohttp 示例)

```python
import aiohttp
import asyncio
import json

ADAPTER_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "your-secure-client-key" # 替换为你的 SECURITY_ADAPTER_API_KEYS

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! What is the capital of France?"}
    ],
    "stream": False # 修改为 True 来测试流式响应
}

async def main():
    async with aiohttp.ClientSession() as session:
        async with session.post(ADAPTER_URL, headers=headers, data=json.dumps(payload)) as response:
            if response.status == 200:
                result = await response.json()
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.status}")
                print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())
```

## 🩺 故障排查与诊断

如果你遇到问题，这些诊断脚本可以帮助你。在你的项目根目录中运行它们。

### 1\. 检查 Gemini API 密钥有效性

`api_key_checker.py` 脚本会测试你 `.env` 文件中的每一个 `GEMINI_API_KEYS`，验证其有效性、配额和模型访问权限。

```bash
docker-compose run --rm gemini-claude-adapter python api_key_checker.py
```

### 2\. 通用健康与依赖检查

`diagnose_script.py` 会执行通用健康检查，验证项目文件和 Python 依赖是否完整。

```bash
docker-compose run --rm gemini-claude-adapter python diagnose_script.py
```

## 📡 API 端点

### 主要端点 (需要客户端密钥)

| 端点 | 方法 | 描述 |
| :--- | :--- | :--- |
| `/v1/chat/completions` | `POST` | **兼容 OpenAI 的主聊天接口** |
| `/v1/models` | `GET` | 列出可用的模型 (OpenAI 格式) |
| `/stats` | `GET` | 查看密钥使用和性能统计 |
| `/metrics`| `GET` | 详细的性能指标 |

### 管理端点 (需要管理密钥)

| 端点 | 方法 | 描述 |
| :--- | :--- | :--- |
| `/admin/reset-key/{prefix}` | `POST` | 手动重置一个失败或冷却中的 Gemini 密钥 |

### 公开端点

| 端点 | 方法 | 描述 |
| :--- | :--- | :--- |
| `/` | `GET` | 服务信息 |
| `/health` | `GET` | 基础健康检查，显示可用密钥状态 |

