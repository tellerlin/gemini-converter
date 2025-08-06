# src/openai_adapter.py
import json
import uuid
import time
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal, Tuple

from pydantic import BaseModel, Field, validator
import logging

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentDict, PartDict, Tool as GeminiTool, FunctionDeclaration

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== OpenAI API 数据模型 ==========
class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @validator('content', pre=True)
    def validate_content(cls, v, values):
        role = values.get('role')
        # tool角色的消息必须有content
        if role == 'tool' and not v:
            raise ValueError("Tool messages must have content")
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True

    class Config:
        extra = 'allow'

    @validator('tools')
    def validate_tools(cls, v):
        if v is not None:
            for tool in v:
                if not isinstance(tool, dict):
                    raise ValueError("Each tool must be a dictionary")
                if tool.get("type") != "function":
                    raise ValueError("Only function tools are supported")
                function = tool.get("function", {})
                if not function.get("name"):
                    raise ValueError("Function must have a name")
        return v


# ========== 转换器类 ==========

class OpenAIToGeminiConverter:
    """
    将 OpenAI API 请求格式转换为 Google Gemini API 格式。
    """
    
    def convert_model(self, openai_model: str) -> str:
        """将OpenAI模型名称映射到Gemini模型名称"""
        model_map = {
            "gpt-4o": "gemini-1.5-pro-latest",
            "gpt-4o-mini": "gemini-1.5-flash-latest",
            "gpt-4-turbo": "gemini-1.5-pro-latest",
            "gpt-4": "gemini-1.5-pro-latest",
            "gpt-3.5-turbo": "gemini-1.5-flash-latest",
        }
        mapped_model = model_map.get(openai_model, "gemini-1.5-pro-latest")
        logger.debug(f"Mapped model {openai_model} -> {mapped_model}")
        return mapped_model

    def _convert_schema_to_gemini(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归地将OpenAI的JSON Schema转换为Gemini的格式。"""
        if not isinstance(json_schema, dict):
            logger.warning("Schema is not a dictionary, returning empty schema")
            return {"type": "OBJECT", "properties": {}}
            
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER", 
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
            "null": "STRING"  # Gemini doesn't have null type, treat as string
        }
        
        schema_type = json_schema.get("type", "object")
        gemini_type = type_mapping.get(schema_type, "STRING")
        
        if gemini_type not in type_mapping.values():
            logger.warning(f"Unknown schema type: {schema_type}, defaulting to STRING")
            gemini_type = "STRING"

        gemini_schema = {"type": gemini_type}
        
        # 复制基本属性
        for key in ["description"]:
            if key in json_schema and json_schema[key]:
                gemini_schema[key] = str(json_schema[key])
        
        # 处理枚举
        if "enum" in json_schema and isinstance(json_schema["enum"], list):
            gemini_schema["enum"] = [str(item) for item in json_schema["enum"]]
        
        # 处理对象类型
        if gemini_type == "OBJECT":
            properties = json_schema.get("properties", {})
            if isinstance(properties, dict):
                gemini_schema["properties"] = {
                    k: self._convert_schema_to_gemini(v)
                    for k, v in properties.items()
                    if isinstance(v, dict)
                }
            
            required = json_schema.get("required", [])
            if isinstance(required, list):
                gemini_schema["required"] = [str(item) for item in required]
        
        # 处理数组类型
        elif gemini_type == "ARRAY":
            items = json_schema.get("items")
            if isinstance(items, dict):
                gemini_schema["items"] = self._convert_schema_to_gemini(items)
            else:
                # 默认数组项类型
                gemini_schema["items"] = {"type": "STRING"}
            
        return gemini_schema

    def convert_tools(self, tools: Optional[List[Dict]], tool_choice: Optional[Union[str, Dict]] = None) -> Optional[List[GeminiTool]]:
        """将OpenAI工具定义转换为Gemini工具定义"""
        if not tools:
            return None
        
        gemini_functions = []
        for i, tool in enumerate(tools):
            try:
                if not isinstance(tool, dict):
                    logger.warning(f"Tool {i} is not a dictionary, skipping")
                    continue
                    
                if tool.get("type") != "function":
                    logger.warning(f"Tool {i} is not a function type, skipping")
                    continue
                
                func_info = tool.get("function", {})
                if not isinstance(func_info, dict):
                    logger.warning(f"Tool {i} function info is not a dictionary, skipping")
                    continue
                    
                name = func_info.get("name")
                if not name or not isinstance(name, str):
                    logger.warning(f"Tool {i} missing valid name, skipping")
                    continue
                
                description = func_info.get("description", "")
                if not isinstance(description, str):
                    description = str(description) if description else ""
                
                parameters = func_info.get("parameters", {"type": "object", "properties": {}})
                if not isinstance(parameters, dict):
                    logger.warning(f"Function {name} has invalid parameters, using empty schema")
                    parameters = {"type": "object", "properties": {}}

                # 创建函数声明
                try:
                    function_declaration = FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters=self._convert_schema_to_gemini(parameters)
                    )
                    gemini_functions.append(function_declaration)
                    logger.debug(f"Successfully converted function: {name}")
                except Exception as e:
                    logger.error(f"Error creating FunctionDeclaration for {name}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error converting tool {i}: {e}")
                continue
        
        if gemini_functions:
            logger.info(f"Converted {len(gemini_functions)} functions to Gemini format")
            return [GeminiTool(function_declarations=gemini_functions)]
        
        logger.warning("No valid functions found in tools")
        return None

    def convert_messages(self, messages: List[ChatMessage]) -> Tuple[List[ContentDict], Optional[str]]:
        """将OpenAI消息格式转换为Gemini消息格式"""
        if not messages:
            logger.warning("No messages provided")
            return [], None
            
        gemini_messages = []
        system_prompt = None
        
        # 首先提取系统消息
        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages:
            system_contents = []
            for msg in system_messages:
                if isinstance(msg.content, str) and msg.content.strip():
                    system_contents.append(msg.content.strip())
            if system_contents:
                system_prompt = "\n\n".join(system_contents)
                logger.debug(f"Extracted system prompt: {system_prompt[:100]}...")
        
        # 然后处理非系统消息
        non_system_messages = [msg for msg in messages if msg.role != "system"]
        
        for i, msg in enumerate(non_system_messages):
            try:
                # 确定角色映射
                if msg.role == "user":
                    role = "user"
                elif msg.role in ["assistant", "tool"]:
                    role = "model"
                else:
                    logger.warning(f"Unknown role {msg.role} in message {i}, treating as user")
                    role = "user"
                
                parts = []
                
                # 处理普通文本内容
                if isinstance(msg.content, str) and msg.content:
                    parts.append(PartDict(text=msg.content))
                elif isinstance(msg.content, list):
                    # 处理多模态内容
                    for content_part in msg.content:
                        if isinstance(content_part, dict):
                            if content_part.get("type") == "text":
                                text = content_part.get("text", "")
                                if text:
                                    parts.append(PartDict(text=str(text)))
                            elif content_part.get("type") == "image_url":
                                # Gemini支持图片，但需要不同的格式
                                logger.warning("Image content not yet supported in this converter")

                # 处理assistant的工具调用
                if msg.role == "assistant" and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        try:
                            if not isinstance(tool_call, (dict, ToolCall)):
                                logger.warning("Invalid tool_call format")
                                continue
                                
                            if isinstance(tool_call, dict):
                                tool_call_dict = tool_call
                            else:
                                tool_call_dict = tool_call.dict()
                            
                            if tool_call_dict.get("type") == "function":
                                func_info = tool_call_dict.get("function", {})
                                function_name = func_info.get("name")
                                arguments = func_info.get("arguments", "{}")
                                
                                if not function_name:
                                    logger.warning("Tool call missing function name")
                                    continue
                                
                                # 解析参数
                                try:
                                    if isinstance(arguments, str):
                                        parsed_args = json.loads(arguments)
                                    elif isinstance(arguments, dict):
                                        parsed_args = arguments
                                    else:
                                        parsed_args = {}
                                except json.JSONDecodeError as e:
                                    logger.error(f"Invalid JSON in tool call arguments: {e}")
                                    parsed_args = {}
                                
                                parts.append(PartDict(
                                    function_call={
                                        'name': function_name, 
                                        'args': parsed_args
                                    }
                                ))
                                logger.debug(f"Added function call: {function_name}")
                                
                        except Exception as e:
                            logger.error(f"Error processing tool call: {e}")
                            continue

                # 处理tool角色的响应
                if msg.role == "tool":
                    function_name = msg.name or "unknown_function"
                    content = msg.content or ""
                    
                    # 尝试解析content为JSON，如果失败则作为字符串处理
                    try:
                        if isinstance(content, str):
                            # 尝试解析为JSON
                            try:
                                parsed_content = json.loads(content)
                            except json.JSONDecodeError:
                                parsed_content = content
                        else:
                            parsed_content = content
                    except Exception:
                        parsed_content = str(content)
                    
                    parts.append(PartDict(
                        function_response={
                            'name': function_name, 
                            'response': parsed_content
                        }
                    ))
                    logger.debug(f"Added function response: {function_name}")

                # 如果没有内容但有角色，添加空文本
                if not parts and role == "user":
                    parts.append(PartDict(text=""))

                # 添加到消息列表
                if parts:
                    gemini_messages.append(ContentDict(role=role, parts=parts))
                    logger.debug(f"Converted message {i}: role={role}, parts_count={len(parts)}")
                    
            except Exception as e:
                logger.error(f"Error converting message {i}: {e}")
                continue

        logger.info(f"Converted {len(messages)} OpenAI messages to {len(gemini_messages)} Gemini messages")
        return gemini_messages, system_prompt


class GeminiToOpenAIConverter:
    """
    将 Gemini API 响应格式转换为 OpenAI API 响应格式。
    """
    
    def _map_finish_reason(self, reason) -> str:
        """映射Gemini的结束原因到OpenAI格式"""
        if not reason:
            return "stop"
            
        reason_str = str(reason).upper()
        if "MAX_TOKENS" in reason_str or "LENGTH" in reason_str:
            return "length"
        elif "TOOL" in reason_str or "FUNCTION" in reason_str:
            return "tool_calls"
        elif "SAFETY" in reason_str:
            return "content_filter"
        elif "STOP" in reason_str:
            return "stop"
        else:
            return "stop"

    def convert_response(self, gemini_response: genai.types.GenerateContentResponse, original_request: ChatCompletionRequest) -> Dict[str, Any]:
        """将Gemini响应转换为OpenAI格式"""
        model = original_request.model
        finish_reason = "stop"
        tool_calls = []
        text_content = ""

        try:
            if gemini_response.candidates and len(gemini_response.candidates) > 0:
                candidate = gemini_response.candidates[0]
                
                # 处理完成原因
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = self._map_finish_reason(candidate.finish_reason)
                
                # 处理内容
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # 处理文本内容
                        if hasattr(part, 'text') and part.text:
                            text_content += part.text
                        
                        # 处理工具调用
                        if hasattr(part, 'function_call') and part.function_call:
                            try:
                                tool_call_id = f"call_{uuid.uuid4().hex}"
                                function_name = part.function_call.name
                                function_args = dict(part.function_call.args) if part.function_call.args else {}
                                
                                tool_calls.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": json.dumps(function_args, ensure_ascii=False)
                                    }
                                })
                                logger.debug(f"Converted function call: {function_name}")
                            except Exception as e:
                                logger.error(f"Error processing function call: {e}")
                                continue

            # 构建消息对象
            message = {
                "role": "assistant",
                "content": text_content if text_content else None,
            }
            
            # 只有在有工具调用时才添加tool_calls字段
            if tool_calls:
                message["tool_calls"] = tool_calls
                if not text_content:
                    message["content"] = None  # 明确设置为None当只有工具调用时

            choice = {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }

            # 处理使用统计
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(gemini_response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(gemini_response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(gemini_response.usage_metadata, 'total_token_count', 0)
                }

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [choice],
                "usage": usage,
                "system_fingerprint": None
            }
            
            logger.debug(f"Converted response: finish_reason={finish_reason}, tool_calls={len(tool_calls)}, content_length={len(text_content)}")
            return response
            
        except Exception as e:
            logger.error(f"Error converting Gemini response: {e}")
            # 返回错误响应
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error processing response: {str(e)}"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "system_fingerprint": None
            }

    async def convert_stream_response(self, gemini_stream, original_request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """将Gemini流式响应转换为OpenAI格式"""
        model = original_request.model
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        # 用于跟踪工具调用的状态
        active_tool_calls = {}
        tool_call_index = 0
        first_chunk_sent = False
        content_buffer = ""

        try:
            async for chunk in gemini_stream:
                try:
                    if not chunk.candidates or len(chunk.candidates) == 0:
                        continue
                    
                    candidate = chunk.candidates[0]
                    delta = {}
                    
                    # 发送初始角色块
                    if not first_chunk_sent:
                        delta["role"] = "assistant"
                        first_chunk_sent = True
                        
                        choice = {"index": 0, "delta": delta}
                        openai_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [choice]
                        }
                        yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                        continue
                    
                    # 处理内容
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # 处理文本内容
                            if hasattr(part, 'text') and part.text:
                                new_content = part.text
                                # 只发送新增的内容
                                if new_content != content_buffer:
                                    delta_content = new_content[len(content_buffer):]
                                    if delta_content:
                                        delta["content"] = delta_content
                                        content_buffer = new_content

                            # 处理工具调用
                            if hasattr(part, 'function_call') and part.function_call:
                                function_name = part.function_call.name
                                function_args = dict(part.function_call.args) if part.function_call.args else {}
                                
                                # 为新的工具调用创建条目
                                if function_name not in active_tool_calls:
                                    tool_call_id = f"call_{uuid.uuid4().hex}"
                                    active_tool_calls[function_name] = {
                                        "index": tool_call_index,
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": function_name,
                                            "arguments": ""
                                        }
                                    }
                                    tool_call_index += 1
                                    
                                    # 发送工具调用开始
                                    delta["tool_calls"] = [{
                                        "index": active_tool_calls[function_name]["index"],
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {"name": function_name, "arguments": ""}
                                    }]
                                
                                # 更新参数（增量方式）
                                new_args = json.dumps(function_args, ensure_ascii=False)
                                current_args = active_tool_calls[function_name]["function"]["arguments"]
                                
                                if new_args != current_args:
                                    delta_args = new_args[len(current_args):]
                                    if delta_args:
                                        active_tool_calls[function_name]["function"]["arguments"] = new_args
                                        delta["tool_calls"] = [{
                                            "index": active_tool_calls[function_name]["index"],
                                            "function": {"arguments": delta_args}
                                        }]
                    
                    # 发送内容块
                    if delta:
                        choice = {"index": 0, "delta": delta}
                        openai_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [choice]
                        }
                        yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"

                    # 处理结束原因
                    if (hasattr(candidate, 'finish_reason') and 
                        candidate.finish_reason and 
                        candidate.finish_reason.name != "FINISH_REASON_UNSPECIFIED"):
                        
                        finish_reason = self._map_finish_reason(candidate.finish_reason)
                        final_choice = {
                            "index": 0, 
                            "delta": {}, 
                            "finish_reason": finish_reason
                        }
                        final_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [final_choice]
                        }
                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                        break
                        
                except Exception as chunk_error:
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    # 发送错误内容但继续处理
                    error_delta = {"content": f"[Error processing chunk: {str(chunk_error)}]"}
                    error_choice = {"index": 0, "delta": error_delta}
                    error_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk", 
                        "created": created_time,
                        "model": model,
                        "choices": [error_choice]
                    }
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                    continue

        except Exception as stream_error:
            logger.error(f"Error in stream conversion: {stream_error}")
            # 发送最终错误块
            error_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\n[Stream Error: {str(stream_error)}]"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

        # 发送流结束标志
        yield "data: [DONE]\n\n"


class APIConfig:
    """持有转换器实例的配置类。"""
    def __init__(self):
        self.openai_to_gemini = OpenAIToGeminiConverter()
        self.gemini_to_openai = GeminiToOpenAIConverter()
        logger.info("🚀 OpenAI-Gemini API Compatibility Layer Initialized (Enhanced Version).")
