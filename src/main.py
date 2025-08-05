# src/main.py
import asyncio
import time
import os
from typing import Dict, Optional, Any, Set, AsyncGenerator, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# 导入新的 openai_adapter 模块
from src.openai_adapter import (
    ChatCompletionRequest, APIConfig
)
from src.config import load_configuration, get_config
from src.performance import initialize_performance_modules, get_performance_stats, monitor_performance

# 加载环境变量
load_dotenv()

# --- 密钥状态与信息 ---
class KeyStatus(Enum):
    ACTIVE = "active"
    COOLING = "cooling"
    FAILED = "failed"

@dataclass
class APIKeyInfo:
    key: str
    status: KeyStatus = KeyStatus.ACTIVE
    failure_count: int = 0
    cooling_until: Optional[float] = None

# --- 依赖注入与安全 ---
config = get_config()
api_config: Optional[APIConfig] = None
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
valid_api_keys: Set[str] = set(config.SECURITY_ADAPTER_API_KEYS)
admin_api_keys: Set[str] = set(config.SECURITY_ADMIN_API_KEYS)

async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header), 
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
):
    """验证API密钥，支持X-API-Key头和Bearer Token两种方式"""
    if not valid_api_keys:
        # 在不安全的模式下允许所有请求
        logger.warning("Running in insecure mode - no API keys configured")
        return "insecure_mode"
    
    key = api_key or (bearer.credentials if bearer else None)
    if key and key in valid_api_keys:
        return key
    
    raise HTTPException(
        status_code=401, 
        detail="Invalid API Key or Bearer Token. Use X-API-Key header or Authorization: Bearer <token>"
    )

async def verify_admin_key(
    api_key: Optional[str] = Depends(api_key_header), 
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
):
    """验证管理员API密钥"""
    if not admin_api_keys:
        raise HTTPException(status_code=403, detail="Admin API keys not configured")
    
    key = api_key or (bearer.credentials if bearer else None)
    if key and key in admin_api_keys:
        return key
    
    raise HTTPException(status_code=403, detail="Invalid Admin API Key")

# --- Gemini 密钥管理器 ---
class GeminiKeyManager:
    def __init__(self):
        self.keys: Dict[str, APIKeyInfo] = {
            key: APIKeyInfo(key=key) for key in config.GEMINI_API_KEYS if key
        }
        self.lock = asyncio.Lock()
        self.last_used_key_index = -1
        
        if not self.keys:
            raise ValueError("No valid GEMINI_API_KEYS provided.")
        
        logger.info(f"Initialized {len(self.keys)} Gemini API keys.")

    async def get_available_key(self) -> Optional[APIKeyInfo]:
        """获取可用的API密钥，使用轮询策略"""
        async with self.lock:
            self._recover_keys()
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            
            if not active_keys:
                logger.warning("No active Gemini API keys available")
                return None
            
            # 轮询策略
            self.last_used_key_index = (self.last_used_key_index + 1) % len(active_keys)
            selected_key = active_keys[self.last_used_key_index]
            logger.debug(f"Selected key: {selected_key.key[:8]}...")
            return selected_key

    def _recover_keys(self):
        """恢复冷却中的密钥"""
        now = time.time()
        recovered_count = 0
        
        for key_info in self.keys.values():
            if (key_info.status == KeyStatus.COOLING and 
                key_info.cooling_until and 
                now > key_info.cooling_until):
                
                key_info.status = KeyStatus.ACTIVE
                key_info.failure_count = 0
                key_info.cooling_until = None
                recovered_count += 1
                logger.info(f"Key {key_info.key[:8]}... recovered to ACTIVE.")
        
        if recovered_count > 0:
            logger.info(f"Recovered {recovered_count} keys from cooling state")

    async def mark_key_failed(self, key: str, error: Exception):
        """标记密钥失败并进入冷却或永久失败状态"""
        async with self.lock:
            if key not in self.keys:
                logger.warning(f"Attempt to mark unknown key as failed: {key[:8]}...")
                return
            
            key_info = self.keys[key]
            key_info.failure_count += 1
            
            # 判断是否为永久性错误
            is_permanent = isinstance(error, (
                google_exceptions.PermissionDenied, 
                google_exceptions.Unauthenticated,
                google_exceptions.InvalidArgument
            ))
            
            if is_permanent or key_info.failure_count >= config.GEMINI_MAX_RETRIES:
                key_info.status = KeyStatus.FAILED
                status_msg = "permanently FAILED"
            else:
                key_info.status = KeyStatus.COOLING
                key_info.cooling_until = time.time() + config.GEMINI_COOLING_PERIOD
                status_msg = f"COOLING for {config.GEMINI_COOLING_PERIOD}s"
            
            logger.warning(
                f"Key {key_info.key[:8]}... marked as {status_msg}. "
                f"Failure count: {key_info.failure_count}. "
                f"Reason: {type(error).__name__}: {str(error)}"
            )
    
    async def get_stats(self) -> Dict[str, int]:
        """获取密钥使用统计"""
        async with self.lock:
            self._recover_keys()
            return {
                "total": len(self.keys),
                "active": sum(1 for k in self.keys.values() if k.status == KeyStatus.ACTIVE),
                "cooling": sum(1 for k in self.keys.values() if k.status == KeyStatus.COOLING),
                "failed": sum(1 for k in self.keys.values() if k.status == KeyStatus.FAILED),
            }

key_manager: Optional[GeminiKeyManager] = None

# --- OpenAI 风格的 Gemini 适配器 ---
class OAIStyleGeminiAdapter:
    def __init__(self, key_mgr: GeminiKeyManager, api_cfg: APIConfig):
        self.key_manager = key_mgr
        self.api_config = api_cfg

    async def process_chat_completion(self, request: ChatCompletionRequest) -> Union[Dict, AsyncGenerator[str, None]]:
        """处理聊天完成请求，支持重试机制"""
        last_error = None
        max_attempts = min(config.GEMINI_MAX_RETRIES + 1, len(self.key_manager.keys))
        
        logger.info(f"Processing chat completion request for model: {request.model}, stream: {request.stream}")
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.error("No available Gemini API keys. All keys are cooling or have failed.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5)
                    continue
                else:
                    break

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}...")
            
            try:
                # 配置Gemini API
                genai.configure(api_key=key_info.key)
                
                # 转换请求格式
                messages, system_prompt = self.api_config.openai_to_gemini.convert_messages(request.messages)
                tools = self.api_config.openai_to_gemini.convert_tools(request.tools)
                model_name = self.api_config.openai_to_gemini.convert_model(request.model)

                logger.debug(f"Converted to Gemini: model={model_name}, messages={len(messages)}, tools={len(tools) if tools else 0}")

                # 创建生成模型
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt,
                    tools=tools
                )
                
                # 生成配置
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                # 请求配置
                request_options = {
                    'timeout': config.GEMINI_REQUEST_TIMEOUT
                }
                
                if request.stream:
                    # 流式响应
                    logger.debug("Creating streaming response")
                    stream = model.generate_content_async(
                        messages, 
                        stream=True, 
                        generation_config=generation_config,
                        request_options=request_options
                    )
                    return self.api_config.gemini_to_openai.convert_stream_response(stream, request)
                else:
                    # 非流式响应
                    logger.debug("Creating non-streaming response")
                    response = await model.generate_content_async(
                        messages, 
                        generation_config=generation_config,
                        request_options=request_options
                    )
                    return self.api_config.gemini_to_openai.convert_response(response, request)
            
            except Exception as e:
                last_error = e
                error_msg = f"Attempt {attempt+1} failed with key {key_info.key[:8]}. Error: {type(e).__name__}: {str(e)}"
                logger.warning(error_msg)
                
                await self.key_manager.mark_key_failed(key_info.key, e)
                
                # 如果不是最后一次尝试，等待一下再重试（指数退避）
                if attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 30)  # 最多等待30秒
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # 所有尝试都失败了
        detail = f"All {max_attempts} attempts failed. Last error: {type(last_error).__name__}: {str(last_error)}"
        logger.error(detail)
        raise HTTPException(status_code=502, detail=detail)

adapter: Optional[OAIStyleGeminiAdapter] = None

# --- FastAPI 应用生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global key_manager, adapter, api_config
    
    # 启动时的初始化
    logger.info("🚀 Starting OpenAI-Style Gemini Adapter...")
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/adapter_{time}.log", 
        rotation="1 day", 
        retention="7 days", 
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    try:
        # 初始化性能监控模块
        initialize_performance_modules(
            cache_enabled=config.CACHE_ENABLED,
            cache_max_size=config.CACHE_MAX_SIZE,
            cache_ttl=config.CACHE_TTL,
            cache_key_prefix=config.CACHE_KEY_PREFIX
        )
        logger.info("✅ Performance monitoring initialized")
        
        # 初始化API配置和适配器
        api_config = APIConfig()
        logger.info("✅ API configuration initialized")
        
        key_manager = GeminiKeyManager()
        logger.info("✅ Gemini key manager initialized")
        
        adapter = OAIStyleGeminiAdapter(key_manager, api_config)
        logger.info("✅ OpenAI-Style Gemini Adapter initialized")
        
        # 输出启动信息
        stats = await key_manager.get_stats()
        logger.info(f"🔑 API Keys: {stats['active']} active, {stats['cooling']} cooling, {stats['failed']} failed")
        logger.info("🎯 OpenAI-Style Gemini Adapter started successfully!")
        
        yield
        
    except Exception as e:
        logger.critical(f"❌ Application failed to start: {e}", exc_info=True)
        raise
    finally:
        # 关闭时的清理
        logger.info("🛑 Adapter shutting down...")

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="OpenAI-Style Gemini Adapter",
    description="""
    Bridges OpenAI API requests to Google's Gemini Pro with smart key rotation.
    
    ## Features
    - 🔄 Smart API key rotation and recovery
    - 🚀 Streaming and non-streaming responses
    - 🛠️ Function/tool calling support
    - 📊 Performance monitoring
    - 🔒 Secure API key management
    - 🌐 OpenAI-compatible endpoints
    
    ## Authentication
    Use either:
    - `X-API-Key` header
    - `Authorization: Bearer <token>` header
    """,
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该更严格
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- 主要端点 ---

@app.post("/v1/chat/completions", 
          summary="Create chat completion", 
          description="Create a chat completion, compatible with OpenAI API")
async def create_chat_completion(
    request: ChatCompletionRequest, 
    client_key: str = Depends(verify_api_key)
):
    """OpenAI兼容的聊天完成端点"""
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        async with monitor_performance("create_chat_completion"):
            response = await adapter.process_chat_completion(request)
            
            if request.stream:
                return StreamingResponse(
                    response, 
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream",
                        "Access-Control-Allow-Origin": "*",
                    }
                )
            else:
                return JSONResponse(content=response)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/models", 
         summary="List models", 
         description="List available models in OpenAI format")
async def list_models(client_key: str = Depends(verify_api_key)):
    """OpenAI兼容的模型列表端点"""
    model_data = [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-4o",
            "parent": None,
        },
        {
            "id": "gpt-4-turbo",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-4-turbo",
            "parent": None,
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-3.5-turbo",
            "parent": None,
        },
    ]
    return {"object": "list", "data": model_data}

# --- 健康检查和监控端点 ---

@app.get("/health", 
         summary="Health check", 
         description="Check service health and key availability")
async def health_check():
    """健康检查端点"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    stats = await key_manager.get_stats()
    is_healthy = stats["active"] > 0
    status_code = 200 if is_healthy else 503
    
    health_data = {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": int(time.time()),
        "service": "OpenAI-Style Gemini Adapter",
        "version": "4.0.0",
        **stats
    }
    
    return JSONResponse(content=health_data, status_code=status_code)

@app.get("/stats", 
         summary="Get statistics", 
         description="Get service performance and key usage statistics")
async def get_stats(client_key: str = Depends(verify_api_key)):
    """统计信息端点"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    key_stats = await key_manager.get_stats()
    perf_stats = get_performance_stats()
    
    return {
        "timestamp": int(time.time()),
        "key_stats": key_stats,
        "performance_stats": perf_stats,
        "service_info": {
            "name": "OpenAI-Style Gemini Adapter",
            "version": "4.0.0",
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }
    }

@app.get("/admin/keys", 
         summary="Admin: Key status", 
         description="Get detailed key status information (admin only)")
async def admin_key_status(admin_key: str = Depends(verify_admin_key)):
    """管理员密钥状态端点"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    detailed_stats = []
    async with key_manager.lock:
        key_manager._recover_keys()
        for key, info in key_manager.keys.items():
            detailed_stats.append({
                "key_id": key[:8] + "..." + key[-4:],
                "status": info.status.value,
                "failure_count": info.failure_count,
                "cooling_until": info.cooling_until,
                "cooling_remaining": max(0, (info.cooling_until or 0) - time.time()) if info.cooling_until else 0
            })
    
    return {
        "timestamp": int(time.time()),
        "keys": detailed_stats,
        "summary": await key_manager.get_stats()
    }

# --- 基础端点 ---

@app.get("/", 
         summary="Service info", 
         description="Get basic service information")
async def root():
    """根端点，返回服务基本信息"""
    return {
        "service": "OpenAI-Style Gemini Adapter",
        "version": "4.0.0",
        "status": "running",
        "timestamp": int(time.time()),
        "documentation": {
            "openapi": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/favicon.ico")
async def favicon():
    """防止favicon请求产生404错误"""
    return JSONResponse(content={"detail": "No favicon available"}, status_code=404)

# --- 启动时记录开始时间 ---
@app.on_event("startup")
async def startup_event():
    """记录应用启动时间"""
    app.state.start_time = time.time()

# --- 异常处理 ---
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """404错误处理"""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "type": "not_found",
                "message": f"The requested endpoint {request.url.path} was not found",
                "available_endpoints": [
                    "/v1/chat/completions",
                    "/v1/models", 
                    "/health",
                    "/stats",
                    "/docs"
                ]
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """500错误处理"""
    logger.error(f"Internal server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_server_error",
                "message": "An internal server error occurred",
                "request_id": str(time.time())
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
