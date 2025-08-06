# src/main.py - 完整版本
import asyncio
import time
import os
from typing import Dict, Optional, Any, Set, AsyncGenerator, Union, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Depends, Body
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
from src.config import get_config
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
    last_used: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0

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

# --- 增强的 Gemini 密钥管理器 ---
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
        """获取可用的API密钥，使用智能轮询策略"""
        async with self.lock:
            self._recover_keys()
            active_keys = [k for k in self.keys.values() if k.status == KeyStatus.ACTIVE]
            
            if not active_keys:
                logger.warning("No active Gemini API keys available")
                return None
            
            # 智能轮询策略：优先选择最近未使用的密钥
            now = time.time()
            active_keys.sort(key=lambda k: k.last_used or 0)
            
            # 如果有从未使用的密钥，优先使用
            unused_keys = [k for k in active_keys if k.last_used is None]
            if unused_keys:
                selected_key = unused_keys[0]
            else:
                # 使用轮询策略
                self.last_used_key_index = (self.last_used_key_index + 1) % len(active_keys)
                selected_key = active_keys[self.last_used_key_index]
            
            # 更新使用记录
            selected_key.last_used = now
            selected_key.total_requests += 1
            
            logger.debug(f"Selected key: {selected_key.key[:8]}... (total requests: {selected_key.total_requests})")
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
                key_info.cooling_until = None
                recovered_count += 1
                logger.info(f"Key {key_info.key[:8]}... recovered to ACTIVE.")
        
        if recovered_count > 0:
            logger.info(f"Recovered {recovered_count} keys from cooling state")

    async def mark_key_success(self, key: str):
        """标记密钥使用成功"""
        async with self.lock:
            if key in self.keys:
                key_info = self.keys[key]
                key_info.successful_requests += 1
                # 成功请求后重置失败计数（部分重置）
                if key_info.failure_count > 0:
                    key_info.failure_count = max(0, key_info.failure_count - 1)

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
            
            # 判断是否为临时性错误（如配额限制）
            is_quota_error = isinstance(error, google_exceptions.ResourceExhausted)
            
            if is_permanent:
                key_info.status = KeyStatus.FAILED
                status_msg = "permanently FAILED"
            elif is_quota_error:
                # 配额错误使用更长的冷却时间
                key_info.status = KeyStatus.COOLING
                key_info.cooling_until = time.time() + (config.GEMINI_COOLING_PERIOD * 3)
                status_msg = f"COOLING (quota) for {config.GEMINI_COOLING_PERIOD * 3}s"
            elif key_info.failure_count >= config.GEMINI_MAX_RETRIES:
                key_info.status = KeyStatus.FAILED
                status_msg = "permanently FAILED (max retries exceeded)"
            else:
                key_info.status = KeyStatus.COOLING
                # 指数退避冷却时间
                cooling_time = config.GEMINI_COOLING_PERIOD * (2 ** (key_info.failure_count - 1))
                key_info.cooling_until = time.time() + min(cooling_time, 3600)  # 最多1小时
                status_msg = f"COOLING for {min(cooling_time, 3600)}s"
            
            logger.warning(
                f"Key {key_info.key[:8]}... marked as {status_msg}. "
                f"Failure count: {key_info.failure_count}. "
                f"Success rate: {key_info.successful_requests}/{key_info.total_requests}. "
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

    async def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细的密钥统计信息"""
        async with self.lock:
            self._recover_keys()
            
            total_requests = sum(k.total_requests for k in self.keys.values())
            total_successful = sum(k.successful_requests for k in self.keys.values())
            
            return {
                "summary": await self.get_stats(),
                "performance": {
                    "total_requests": total_requests,
                    "successful_requests": total_successful,
                    "success_rate": total_successful / total_requests if total_requests > 0 else 0,
                    "average_requests_per_key": total_requests / len(self.keys) if self.keys else 0
                },
                "keys": [
                    {
                        "key_id": key[:8] + "..." + key[-4:],
                        "status": info.status.value,
                        "failure_count": info.failure_count,
                        "total_requests": info.total_requests,
                        "successful_requests": info.successful_requests,
                        "success_rate": info.successful_requests / info.total_requests if info.total_requests > 0 else 0,
                        "cooling_until": info.cooling_until,
                        "cooling_remaining": max(0, (info.cooling_until or 0) - time.time()) if info.cooling_until else 0,
                        "last_used": info.last_used
                    }
                    for key, info in self.keys.items()
                ]
            }

    async def add_key(self, key: str) -> bool:
        """动态添加新的API密钥"""
        async with self.lock:
            if key in self.keys:
                return False  # 密钥已存在
            
            self.keys[key] = APIKeyInfo(key=key)
            logger.info(f"Added new API key: {key[:8]}...")
            return True

    async def remove_key(self, key: str) -> bool:
        """动态移除API密钥"""
        async with self.lock:
            if key not in self.keys:
                return False  # 密钥不存在
            
            del self.keys[key]
            logger.info(f"Removed API key: {key[:8]}...")
            return True

    async def update_key_status(self, key: str, status: KeyStatus) -> bool:
        """更新密钥状态"""
        async with self.lock:
            if key not in self.keys:
                return False
            
            self.keys[key].status = status
            if status == KeyStatus.ACTIVE:
                self.keys[key].cooling_until = None
                self.keys[key].failure_count = 0
            
            logger.info(f"Updated key {key[:8]}... status to {status.value}")
            return True

key_manager: Optional[GeminiKeyManager] = None

# --- 增强的 OpenAI 风格的 Gemini 适配器 ---
class OAIStyleGeminiAdapter:
    def __init__(self, key_mgr: GeminiKeyManager, api_cfg: APIConfig):
        self.key_manager = key_mgr
        self.api_config = api_cfg

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """验证请求参数，返回错误信息或None"""
        try:
            # 验证消息
            if not request.messages:
                return "Messages array cannot be empty"
            
            # 验证工具定义
            if request.tools:
                for i, tool in enumerate(request.tools):
                    if not isinstance(tool, dict):
                        return f"Tool {i} must be a dictionary"
                    
                    if tool.get("type") != "function":
                        return f"Tool {i} type must be 'function'"
                    
                    function = tool.get("function", {})
                    if not function.get("name"):
                        return f"Tool {i} function must have a name"
                    
                    # 验证参数schema
                    parameters = function.get("parameters", {})
                    if not isinstance(parameters, dict):
                        return f"Tool {i} parameters must be a dictionary"
            
            # 验证温度参数
            if not (0.0 <= request.temperature <= 2.0):
                return "Temperature must be between 0.0 and 2.0"
            
            # 验证top_p参数
            if request.top_p is not None and not (0.0 <= request.top_p <= 1.0):
                return "top_p must be between 0.0 and 1.0"
            
            # 验证max_tokens
            if request.max_tokens is not None and request.max_tokens <= 0:
                return "max_tokens must be positive"
            
            # 验证n参数
            if hasattr(request, 'n') and request.n is not None and (request.n < 1 or request.n > 10):
                return "n must be between 1 and 10"
            
            return None
            
        except Exception as e:
            return f"Request validation error: {str(e)}"

    async def process_chat_completion(self, request: ChatCompletionRequest) -> Union[Dict, AsyncGenerator[str, None]]:
        """处理聊天完成请求，支持增强的重试机制和工具调用"""
        
        # 验证请求
        validation_error = self._validate_request(request)
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        last_error = None
        max_attempts = min(config.GEMINI_MAX_RETRIES + 1, len(self.key_manager.keys))
        
        candidate_count = getattr(request, 'n', 1) or 1
        
        logger.info(f"Processing chat completion request for model: {request.model}, stream: {request.stream}, tools: {len(request.tools) if request.tools else 0}, n: {candidate_count}")
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.error("No available Gemini API keys. All keys are cooling or have failed.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(min(5 * (attempt + 1), 30))  # 指数退避等待
                    continue
                else:
                    break

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}... (req: {key_info.total_requests})")
            
            try:
                # 配置Gemini API
                genai.configure(api_key=key_info.key)
                
                # 转换请求格式
                messages, system_prompt = self.api_config.openai_to_gemini.convert_messages(request.messages)
                if not messages:
                    raise HTTPException(status_code=400, detail="No valid messages after conversion")
                
                # 转换工具定义和工具选择
                tools = None
                tool_config = None
                if request.tools:
                    try:
                        tools, tool_config = self.api_config.openai_to_gemini.convert_tools(request.tools, request.tool_choice)
                        logger.debug(f"Converted {len(request.tools)} tools to Gemini format with tool_config: {tool_config}")
                    except Exception as tool_error:
                        logger.error(f"Error converting tools: {tool_error}")
                        raise HTTPException(status_code=400, detail=f"Tool conversion error: {str(tool_error)}")
                
                model_name = self.api_config.openai_to_gemini.convert_model(request.model)
                logger.debug(f"Converted to Gemini: model={model_name}, messages={len(messages)}, tools={len(tools) if tools else 0}")

                # 创建生成模型
                model_kwargs = {
                    "model_name": model_name,
                    "system_instruction": system_prompt,
                }
                
                if tools:
                    model_kwargs["tools"] = tools

                model = genai.GenerativeModel(**model_kwargs)
                
                # 增强的生成配置，包含top_p和response_format支持
                generation_config_kwargs = {
                    "max_output_tokens": min(request.max_tokens or 8192, 8192),
                    "temperature": request.temperature,
                    "candidate_count": candidate_count,  # 支持多候选回复
                }
                
                # 添加top_p支持
                if request.top_p is not None:
                    generation_config_kwargs["top_p"] = request.top_p
                
                # 添加JSON输出格式支持
                if request.response_format and request.response_format.get("type") == "json_object":
                    generation_config_kwargs["response_mime_type"] = "application/json"
                    logger.debug("Enabled JSON response format")
                
                generation_config = genai.types.GenerationConfig(**generation_config_kwargs)
                
                # 请求配置，包含tool_config
                request_options = {
                    'timeout': config.GEMINI_REQUEST_TIMEOUT
                }
                
                # API调用参数
                api_call_kwargs = {
                    "generation_config": generation_config,
                    "request_options": request_options
                }
                
                # 添加tool_config支持
                if tool_config:
                    api_call_kwargs["tool_config"] = tool_config
                    logger.debug(f"Using tool_config: {tool_config}")
                
                if request.stream:
                    if candidate_count > 1:
                        raise HTTPException(status_code=400, detail="Streaming is not supported when n > 1.")
                        
                    # 流式响应
                    logger.debug("Creating streaming response")
                    try:
                        stream = model.generate_content_async(
                            messages, 
                            stream=True, 
                            **api_call_kwargs
                        )
                        
                        # 包装流式响应以处理成功/失败
                        async def wrapped_stream():
                            try:
                                async for chunk in self.api_config.gemini_to_openai.convert_stream_response(stream, request):
                                    yield chunk
                                # 如果流完成没有异常，标记成功
                                await self.key_manager.mark_key_success(key_info.key)
                            except Exception as stream_error:
                                await self.key_manager.mark_key_failed(key_info.key, stream_error)
                                raise
                        
                        return wrapped_stream()
                        
                    except Exception as stream_setup_error:
                        logger.error(f"Error setting up stream: {stream_setup_error}")
                        await self.key_manager.mark_key_failed(key_info.key, stream_setup_error)
                        raise
                else:
                    # 非流式响应
                    logger.debug("Creating non-streaming response")
                    response = await model.generate_content_async(
                        messages, 
                        **api_call_kwargs
                    )
                    
                    # 检查响应有效性
                    if not response.candidates:
                        raise ValueError("No candidates in Gemini response")
                    
                    # 标记成功并转换响应
                    await self.key_manager.mark_key_success(key_info.key)
                    converted_response = self.api_config.gemini_to_openai.convert_response(response, request)
                    
                    return converted_response
            
            except google_exceptions.ResourceExhausted as quota_error:
                last_error = quota_error
                logger.warning(f"Quota exhausted for key {key_info.key[:8]}...: {quota_error}")
                await self.key_manager.mark_key_failed(key_info.key, quota_error)
                continue
                
            except google_exceptions.InvalidArgument as arg_error:
                last_error = arg_error
                logger.error(f"Invalid argument for key {key_info.key[:8]}...: {arg_error}")
                await self.key_manager.mark_key_failed(key_info.key, arg_error)
                # 参数错误通常是请求本身的问题，不需要重试其他密钥
                raise HTTPException(status_code=400, detail=f"Invalid request: {str(arg_error)}")
                
            except google_exceptions.PermissionDenied as perm_error:
                last_error = perm_error
                logger.error(f"Permission denied for key {key_info.key[:8]}...: {perm_error}")
                await self.key_manager.mark_key_failed(key_info.key, perm_error)
                continue
                
            except google_exceptions.Unauthenticated as auth_error:
                last_error = auth_error
                logger.error(f"Authentication failed for key {key_info.key[:8]}...: {auth_error}")
                await self.key_manager.mark_key_failed(key_info.key, auth_error)
                continue
                
            except asyncio.TimeoutError as timeout_error:
                last_error = timeout_error
                logger.warning(f"Timeout for key {key_info.key[:8]}...: {timeout_error}")
                await self.key_manager.mark_key_failed(key_info.key, timeout_error)
                continue
                
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
        if isinstance(last_error, google_exceptions.ResourceExhausted):
            detail = "All API keys have reached their quota limits. Please try again later."
            status_code = 429  # Too Many Requests
        elif isinstance(last_error, google_exceptions.PermissionDenied):
            detail = "All API keys lack necessary permissions."
            status_code = 403  # Forbidden
        elif isinstance(last_error, google_exceptions.Unauthenticated):
            detail = "All API keys are invalid or expired."
            status_code = 401  # Unauthorized
        else:
            detail = f"All {max_attempts} attempts failed. Last error: {type(last_error).__name__}: {str(last_error)}"
            status_code = 502  # Bad Gateway
            
        logger.error(detail)
        raise HTTPException(status_code=status_code, detail=detail)

adapter: Optional[OAIStyleGeminiAdapter] = None

# --- FastAPI 应用生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global key_manager, adapter, api_config
    
    # 启动时的初始化
    logger.info("🚀 Starting OpenAI-Style Gemini Adapter...")
    app.state.start_time = time.time()
    
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
    Advanced bridge between OpenAI API and Google's Gemini Pro with enhanced features.
    
    ## Features
    - 🔄 Intelligent API key rotation with success tracking
    - 🚀 Streaming and non-streaming responses
    - 🛠️ Enhanced function/tool calling support with validation
    - 📊 Comprehensive performance monitoring
    - 🔒 Secure API key management with detailed statistics
    - 🌐 Full OpenAI API compatibility
    - ⚡ Optimized error handling and recovery
    - 🎯 Smart quota management and exponential backoff
    - 🔧 Dynamic key management for runtime flexibility
    - 🎛️ Advanced generation parameters (top_p, JSON format, multiple candidates)
    
    ## Authentication
    Use either:
    - `X-API-Key` header
    - `Authorization: Bearer <token>` header
    
    ## Error Handling
    - Automatic retry with exponential backoff
    - Smart key rotation on failures
    - Detailed error reporting and logging
    """,
    version="4.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.SERVICE_CORS_ORIGINS, # 使用配置文件中的设置
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- 主要端点 ---

@app.post("/v1/chat/completions", 
          summary="Create chat completion", 
          description="Create a chat completion with enhanced tool support, compatible with OpenAI API")
async def create_chat_completion(
    request: ChatCompletionRequest, 
    client_key: str = Depends(verify_api_key)
):
    """OpenAI兼容的聊天完成端点，支持增强的工具调用"""
    if not adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        async with monitor_performance("create_chat_completion"):
            response = await adapter.process_chat_completion(request)
            
            if request.stream:
                return StreamingResponse(
                    response, 
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream",
                        "Access-Control-Allow-Origin": "*",
                        "X-Accel-Buffering": "no",  # 禁用nginx缓冲
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
         description="List available models in OpenAI format with accurate metadata")
async def list_models(client_key: str = Depends(verify_api_key)):
    """OpenAI兼容的模型列表端点，提供准确的模型信息"""
    current_time = int(time.time())
    # 修复：提供准确的max_tokens并完成列表
    model_data = [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": current_time,
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-4o",
            "parent": None,
            "context_window": 1048576,
            "max_tokens": 8192, # Gemini 1.5 Pro max output is 8192
            "capabilities": ["chat", "tools", "streaming", "json_mode", "vision"]
        },
        {
            "id": "gpt-4-turbo",
            "object": "model",
            "created": current_time,
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-4-turbo",
            "parent": None,
            "context_window": 1048576,
            "max_tokens": 8192,
            "capabilities": ["chat", "tools", "streaming", "json_mode", "vision"]
        },
        {
            "id": "gpt-4o-mini",
            "object": "model",
            "created": current_time,
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-4o-mini",
            "parent": None,
            "context_window": 1048576,
            "max_tokens": 8192, # Gemini 1.5 Flash max output is 8192
            "capabilities": ["chat", "tools", "streaming", "json_mode", "vision"]
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": current_time,
            "owned_by": "openai-emulated",
            "permission": [],
            "root": "gpt-3.5-turbo",
            "parent": None,
            "context_window": 1048576,
            "max_tokens": 8192,
            "capabilities": ["chat", "tools", "streaming", "json_mode", "vision"]
        },
    ]
    return {"object": "list", "data": model_data}

# --- 健康检查和监控端点 ---

@app.get("/health", 
         summary="Health check", 
         description="Check service health and key availability with detailed status")
async def health_check():
    """增强的健康检查端点"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    stats = await key_manager.get_stats()
    detailed_stats = await key_manager.get_detailed_stats()
    
    is_healthy = stats["active"] > 0
    status_code = 200 if is_healthy else 503
    
    health_data = {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": int(time.time()),
        "service": "OpenAI-Style Gemini Adapter",
        "version": "4.2.0",
        "key_summary": stats,
        "performance": detailed_stats["performance"],
        "uptime": time.time() - getattr(app.state, 'start_time', time.time()),
        "message": "All systems operational" if is_healthy else "Some API keys unavailable"
    }
    
    return JSONResponse(content=health_data, status_code=status_code)

@app.get("/stats", 
         summary="Get statistics", 
         description="Get comprehensive service performance and key usage statistics")
async def get_stats(client_key: str = Depends(verify_api_key)):
    """增强的统计信息端点"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")

    key_stats = await key_manager.get_detailed_stats()
    perf_stats = get_performance_stats()

    # Combinar todas las estadísticas en una sola respuesta
    all_stats = {
        "key_management_stats": key_stats,
        "adapter_performance_stats": perf_stats.get("performance_stats", {}),
        "cache_stats": perf_stats.get("cache_stats", {})
    }

    return JSONResponse(content=all_stats)


# --- 新增：动态密钥管理端点 ---

@app.post("/admin/keys", 
          summary="Add a new Gemini API key", 
          tags=["Admin Key Management"],
          status_code=201)
async def add_gemini_key(
    admin_key: str = Depends(verify_admin_key),
    key_to_add: str = Body(..., embed=True, description="The Gemini API key to add.")
):
    """动态添加一个新的Gemini API密钥到池中。"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    success = await key_manager.add_key(key_to_add)
    if not success:
        raise HTTPException(status_code=409, detail="API key already exists.")
    
    return {"status": "success", "message": f"API key starting with {key_to_add[:8]} added."}


@app.delete("/admin/keys", 
            summary="Remove a Gemini API key", 
            tags=["Admin Key Management"],
            status_code=200)
async def remove_gemini_key(
    admin_key: str = Depends(verify_admin_key),
    key_to_remove: str = Body(..., embed=True, description="The Gemini API key to remove.")
):
    """从池中动态移除一个指定的Gemini API密钥。"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")

    success = await key_manager.remove_key(key_to_remove)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found.")
    
    return {"status": "success", "message": f"API key starting with {key_to_remove[:8]} removed."}


@app.put("/admin/keys/{key_id}", 
         summary="Update status of a Gemini API key", 
         tags=["Admin Key Management"],
         status_code=200)
async def update_gemini_key_status(
    key_id: str,
    status: KeyStatus,
    admin_key: str = Depends(verify_admin_key),
):
    """手动更新一个密钥的状态（例如，将一个冷却中的密钥重置为激活状态）。"""
    if not key_manager:
        raise HTTPException(status_code=503, detail="Key Manager not initialized")
    
    # 因为我们只存储了部分key信息，需要找到完整的key
    full_key = None
    for k in key_manager.keys.keys():
        if k.startswith(key_id):
            full_key = k
            break
            
    if not full_key:
         raise HTTPException(status_code=404, detail=f"No key found starting with '{key_id}'")

    success = await key_manager.update_key_status(full_key, status)
    if not success:
        # This case is unlikely if the above check passes, but for completeness
        raise HTTPException(status_code=404, detail="API key not found for status update.")

    return {"status": "success", "message": f"Status of key {full_key[:8]}... updated to {status.value}."}
