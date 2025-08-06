# src/gemini_adapter.py - 原生Gemini API适配器
import asyncio
import time
import json
import uuid
from typing import Dict, Optional, Any, List, AsyncGenerator, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException
from loguru import logger
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import GenerationConfig, ContentDict, PartDict


# ========== Gemini API 数据模型 ==========

class GeminiContent(BaseModel):
    """Gemini API Content格式"""
    role: str = Field(..., description="Role: 'user' or 'model'")
    parts: List[Dict[str, Any]] = Field(..., description="Content parts")

    class Config:
        extra = 'allow'


class GeminiGenerationConfig(BaseModel):
    """Gemini API GenerationConfig格式"""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    candidate_count: Optional[int] = Field(None, ge=1, le=8)
    max_output_tokens: Optional[int] = Field(None, ge=1)
    stop_sequences: Optional[List[str]] = None
    response_mime_type: Optional[str] = None

    class Config:
        extra = 'allow'


class GeminiSafetySettings(BaseModel):
    """Gemini API SafetySettings格式"""
    category: str
    threshold: str

    class Config:
        extra = 'allow'


class GeminiGenerateContentRequest(BaseModel):
    """Gemini API generateContent请求格式"""
    contents: List[GeminiContent] = Field(..., description="List of content items")
    generation_config: Optional[GeminiGenerationConfig] = None
    safety_settings: Optional[List[GeminiSafetySettings]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_config: Optional[Dict[str, Any]] = None
    system_instruction: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'

    @field_validator('contents')
    @classmethod
    def validate_contents(cls, v):
        """验证contents不能为空"""
        if not v:
            raise ValueError("Contents cannot be empty")
        return v


class GeminiStreamGenerateContentRequest(BaseModel):
    """Gemini API streamGenerateContent请求格式"""
    contents: List[GeminiContent] = Field(..., description="List of content items")
    generation_config: Optional[GeminiGenerationConfig] = None
    safety_settings: Optional[List[GeminiSafetySettings]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_config: Optional[Dict[str, Any]] = None
    system_instruction: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'

    @field_validator('contents')
    @classmethod
    def validate_contents(cls, v):
        """验证contents不能为空"""
        if not v:
            raise ValueError("Contents cannot be empty")
        return v


# ========== 原生Gemini适配器 ==========

class NativeGeminiAdapter:
    """原生Gemini API适配器，直接处理Gemini格式的请求和响应"""
    
    def __init__(self, key_manager):
        self.key_manager = key_manager
        logger.info("🔧 Native Gemini API Adapter initialized")

    def _validate_request(self, request: Union[GeminiGenerateContentRequest, GeminiStreamGenerateContentRequest]) -> Optional[str]:
        """验证Gemini API请求参数"""
        try:
            # 验证contents
            if not request.contents:
                return "Contents array cannot be empty"
            
            for i, content in enumerate(request.contents):
                if not content.parts:
                    return f"Content {i} parts cannot be empty"
                
                # 验证角色
                if content.role not in ["user", "model"]:
                    return f"Content {i} role must be 'user' or 'model', got: {content.role}"
            
            # 验证generation_config
            if request.generation_config:
                config = request.generation_config
                if config.temperature is not None and not (0.0 <= config.temperature <= 2.0):
                    return "Temperature must be between 0.0 and 2.0"
                
                if config.top_p is not None and not (0.0 <= config.top_p <= 1.0):
                    return "top_p must be between 0.0 and 1.0"
                
                if config.max_output_tokens is not None and config.max_output_tokens <= 0:
                    return "max_output_tokens must be positive"
                
                if config.candidate_count is not None and not (1 <= config.candidate_count <= 8):
                    return "candidate_count must be between 1 and 8"
            
            # 验证工具
            if request.tools:
                for i, tool in enumerate(request.tools):
                    if not isinstance(tool, dict):
                        return f"Tool {i} must be a dictionary"
                    
                    if "function_declarations" not in tool:
                        return f"Tool {i} must have function_declarations"
            
            return None
            
        except Exception as e:
            return f"Request validation error: {str(e)}"

    def _convert_to_genai_format(self, request: Union[GeminiGenerateContentRequest, GeminiStreamGenerateContentRequest]) -> tuple:
        """将请求转换为python genai库的格式"""
        try:
            # 转换contents
            contents = []
            for content in request.contents:
                genai_content = ContentDict(
                    role=content.role,
                    parts=[PartDict(**part) for part in content.parts]
                )
                contents.append(genai_content)
            
            # 转换generation_config
            generation_config = None
            if request.generation_config:
                config_dict = request.generation_config.model_dump(exclude_none=True)
                generation_config = GenerationConfig(**config_dict)
            
            # 转换tools
            tools = None
            if request.tools:
                # 直接使用原始工具格式，让genai库处理
                tools = request.tools
            
            # 转换tool_config
            tool_config = request.tool_config
            
            # 转换system_instruction
            system_instruction = None
            if request.system_instruction:
                if isinstance(request.system_instruction, dict):
                    if "parts" in request.system_instruction:
                        # 如果是完整的instruction格式
                        system_instruction = request.system_instruction
                    else:
                        # 如果只是文本内容
                        text = request.system_instruction.get("text", "")
                        if text:
                            system_instruction = {"parts": [{"text": text}]}
            
            # 转换safety_settings
            safety_settings = None
            if request.safety_settings:
                safety_settings = [setting.model_dump() for setting in request.safety_settings]
            
            return contents, generation_config, tools, tool_config, system_instruction, safety_settings
            
        except Exception as e:
            logger.error(f"Error converting request to genai format: {e}")
            raise ValueError(f"Request format conversion error: {str(e)}")

    def _format_gemini_response(self, response: genai.types.GenerateContentResponse) -> Dict[str, Any]:
        """将genai响应格式化为Gemini API格式"""
        try:
            formatted_response = {
                "candidates": [],
                "promptFeedback": None,
                "usageMetadata": None
            }
            
            # 处理候选回复
            if response.candidates:
                for candidate in response.candidates:
                    formatted_candidate = {
                        "content": {
                            "parts": [],
                            "role": "model"
                        },
                        "finishReason": str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else None,
                        "index": getattr(candidate, 'index', 0),
                        "safetyRatings": []
                    }
                    
                    # 处理内容parts
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            part_dict = {}
                            
                            # 处理文本内容
                            if hasattr(part, 'text') and part.text:
                                part_dict["text"] = part.text
                            
                            # 处理函数调用
                            if hasattr(part, 'function_call') and part.function_call:
                                part_dict["functionCall"] = {
                                    "name": part.function_call.name,
                                    "args": dict(part.function_call.args) if part.function_call.args else {}
                                }
                            
                            # 处理函数响应
                            if hasattr(part, 'function_response') and part.function_response:
                                part_dict["functionResponse"] = {
                                    "name": part.function_response.name,
                                    "response": part.function_response.response if part.function_response.response else {}
                                }
                            
                            if part_dict:
                                formatted_candidate["content"]["parts"].append(part_dict)
                    
                    # 处理安全评级
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            formatted_candidate["safetyRatings"].append({
                                "category": str(rating.category) if hasattr(rating, 'category') else "UNKNOWN",
                                "probability": str(rating.probability) if hasattr(rating, 'probability') else "UNKNOWN"
                            })
                    
                    formatted_response["candidates"].append(formatted_candidate)
            
            # 处理使用统计
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                formatted_response["usageMetadata"] = {
                    "promptTokenCount": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "candidatesTokenCount": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "totalTokenCount": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            # 处理提示反馈
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                formatted_response["promptFeedback"] = {
                    "blockReason": str(response.prompt_feedback.block_reason) if hasattr(response.prompt_feedback, 'block_reason') else None,
                    "safetyRatings": []
                }
                
                if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                        formatted_response["promptFeedback"]["safetyRatings"].append({
                            "category": str(rating.category) if hasattr(rating, 'category') else "UNKNOWN",
                            "probability": str(rating.probability) if hasattr(rating, 'probability') else "UNKNOWN"
                        })
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting Gemini response: {e}")
            return {
                "candidates": [{
                    "content": {
                        "parts": [{"text": f"Error formatting response: {str(e)}"}],
                        "role": "model"
                    },
                    "finishReason": "ERROR",
                    "index": 0
                }],
                "promptFeedback": None,
                "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
            }

    async def _format_gemini_stream_chunk(self, chunk: genai.types.GenerateContentResponse) -> str:
        """将流式响应块格式化为Gemini API格式"""
        try:
            formatted_chunk = self._format_gemini_response(chunk)
            return json.dumps(formatted_chunk, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"Error formatting stream chunk: {e}")
            error_chunk = {
                "candidates": [{
                    "content": {
                        "parts": [{"text": f"[Stream Error: {str(e)}]"}],
                        "role": "model"
                    },
                    "finishReason": "ERROR",
                    "index": 0
                }]
            }
            return json.dumps(error_chunk, ensure_ascii=False) + "\n"

    async def process_generate_content(self, request: GeminiGenerateContentRequest, model_name: str) -> Dict[str, Any]:
        """处理generateContent请求（非流式）"""
        
        # 验证请求
        validation_error = self._validate_request(request)
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        last_error = None
        max_attempts = min(3, len(self.key_manager.keys))  # 最多尝试3次或所有可用密钥
        
        logger.info(f"Processing Gemini generateContent request for model: {model_name}")
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.error("No available Gemini API keys")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(min(5 * (attempt + 1), 30))
                    continue
                else:
                    break

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}...")
            
            try:
                # 配置Gemini API
                genai.configure(api_key=key_info.key)
                
                # 转换请求格式
                contents, generation_config, tools, tool_config, system_instruction, safety_settings = self._convert_to_genai_format(request)
                
                # 创建生成模型
                model_kwargs = {"model_name": model_name}
                if system_instruction:
                    model_kwargs["system_instruction"] = system_instruction
                if tools:
                    model_kwargs["tools"] = tools
                
                model = genai.GenerativeModel(**model_kwargs)
                
                # API调用参数
                api_call_kwargs = {}
                if generation_config:
                    api_call_kwargs["generation_config"] = generation_config
                if tool_config:
                    api_call_kwargs["tool_config"] = tool_config
                if safety_settings:
                    api_call_kwargs["safety_settings"] = safety_settings
                
                # 调用API
                response = await model.generate_content_async(contents, **api_call_kwargs)
                
                # 检查响应有效性
                if not response.candidates:
                    raise ValueError("No candidates in Gemini response")
                
                # 标记成功并返回格式化的响应
                await self.key_manager.mark_key_success(key_info.key)
                return self._format_gemini_response(response)
            
            except google_exceptions.ResourceExhausted as quota_error:
                last_error = quota_error
                logger.warning(f"Quota exhausted for key {key_info.key[:8]}...: {quota_error}")
                await self.key_manager.mark_key_failed(key_info.key, quota_error)
                continue
                
            except google_exceptions.InvalidArgument as arg_error:
                last_error = arg_error
                logger.error(f"Invalid argument for key {key_info.key[:8]}...: {arg_error}")
                await self.key_manager.mark_key_failed(key_info.key, arg_error)
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
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed with key {key_info.key[:8]}. Error: {type(e).__name__}: {str(e)}")
                await self.key_manager.mark_key_failed(key_info.key, e)
                
                if attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 30)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        # 所有尝试都失败了
        if isinstance(last_error, google_exceptions.ResourceExhausted):
            detail = "All API keys have reached their quota limits. Please try again later."
            status_code = 429
        elif isinstance(last_error, google_exceptions.PermissionDenied):
            detail = "All API keys lack necessary permissions."
            status_code = 403
        elif isinstance(last_error, google_exceptions.Unauthenticated):
            detail = "All API keys are invalid or expired."
            status_code = 401
        else:
            detail = f"All {max_attempts} attempts failed. Last error: {type(last_error).__name__}: {str(last_error)}"
            status_code = 502
            
        logger.error(detail)
        raise HTTPException(status_code=status_code, detail=detail)

    async def process_stream_generate_content(self, request: GeminiStreamGenerateContentRequest, model_name: str) -> AsyncGenerator[str, None]:
        """处理streamGenerateContent请求（流式）"""
        
        # 验证请求
        validation_error = self._validate_request(request)
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        last_error = None
        max_attempts = min(3, len(self.key_manager.keys))
        
        logger.info(f"Processing Gemini streamGenerateContent request for model: {model_name}")
        
        for attempt in range(max_attempts):
            key_info = await self.key_manager.get_available_key()
            if not key_info:
                logger.error("No available Gemini API keys")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(min(5 * (attempt + 1), 30))
                    continue
                else:
                    break

            logger.info(f"Attempt {attempt + 1}/{max_attempts} using key {key_info.key[:8]}...")
            
            try:
                # 配置Gemini API
                genai.configure(api_key=key_info.key)
                
                # 转换请求格式
                contents, generation_config, tools, tool_config, system_instruction, safety_settings = self._convert_to_genai_format(request)
                
                # 创建生成模型
                model_kwargs = {"model_name": model_name}
                if system_instruction:
                    model_kwargs["system_instruction"] = system_instruction
                if tools:
                    model_kwargs["tools"] = tools
                
                model = genai.GenerativeModel(**model_kwargs)
                
                # API调用参数
                api_call_kwargs = {"stream": True}
                if generation_config:
                    api_call_kwargs["generation_config"] = generation_config
                if tool_config:
                    api_call_kwargs["tool_config"] = tool_config
                if safety_settings:
                    api_call_kwargs["safety_settings"] = safety_settings
                
                # 调用流式API
                stream = model.generate_content_async(contents, **api_call_kwargs)
                
                # 包装流式响应
                async def wrapped_stream():
                    try:
                        async for chunk in stream:
                            yield await self._format_gemini_stream_chunk(chunk)
                        # 流完成，标记成功
                        await self.key_manager.mark_key_success(key_info.key)
                    except Exception as stream_error:
                        await self.key_manager.mark_key_failed(key_info.key, stream_error)
                        raise
                
                return wrapped_stream()
                
            except Exception as e:
                last_error = e
                logger.warning(f"Stream setup failed: {e}")
                await self.key_manager.mark_key_failed(key_info.key, e)
                
                if attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt, 30)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    continue
        
        # 所有尝试都失败了
        detail = f"All {max_attempts} attempts failed. Last error: {type(last_error).__name__}: {str(last_error)}"
        logger.error(detail)
        raise HTTPException(status_code=502, detail=detail)