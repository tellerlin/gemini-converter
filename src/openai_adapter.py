# src/openai_adapter.py
import json
import uuid
import time
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal, Tuple

from pydantic import BaseModel, Field
import logging

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentDict, PartDict, Tool as GeminiTool, FunctionDeclaration

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== OpenAI API 数据模型 ==========
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None] = None # content 可以为 None，例如在 tool_calls 中
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    stream: bool = False
    temperature: float = 1.0
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        extra = 'allow'


# ========== 转换器类 ==========

class OpenAIToGeminiConverter:
    """
    将 OpenAI API 请求格式转换为 Google Gemini API 格式。
    """
    def convert_model(self, openai_model: str) -> str:
        """将OpenAI模型名称映射到Gemini模型名称"""
        model_map = {
            "gpt-4o": "gemini-1.5-pro-latest",
            "gpt-4-turbo": "gemini-1.5-pro-latest",
            "gpt-4": "gemini-pro",
            "gpt-3.5-turbo": "gemini-1.5-flash-latest",
        }
        mapped_model = model_map.get(openai_model, "gemini-1.5-pro-latest")
        logger.debug(f"Mapped model {openai_model} -> {mapped_model}")
        return mapped_model

    def _convert_schema_to_gemini(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归地将OpenAI的JSON Schema转换为Gemini的格式。"""
        if not isinstance(json_schema, dict):
            return {}
            
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
        }
        
        schema_type = json_schema.get("type")
        gemini_type = type_mapping.get(schema_type)
        if not gemini_type:
            logger.warning(f"Unknown schema type: {schema_type}")
            return {}

        gemini_schema = {"type": gemini_type}
        
        # 复制基本属性
        for key in ["description", "enum"]:
            if key in json_schema:
                gemini_schema[key] = json_schema[key]
        
        # 处理对象类型
        if gemini_type == "OBJECT" and "properties" in json_schema:
            gemini_schema["properties"] = {
                k: self._convert_schema_to_gemini(v)
                for k, v in json_schema["properties"].items()
            }
            if "required" in json_schema:
                gemini_schema["required"] = json_schema["required"]
        
        # 处理数组类型
        if gemini_type == "ARRAY" and "items" in json_schema:
            gemini_schema["items"] = self._convert_schema_to_gemini(json_schema["items"])
            
        return gemini_schema

    def convert_tools(self, tools: Optional[List[Dict]]) -> Optional[List[GeminiTool]]:
        """将OpenAI工具定义转换为Gemini工具定义"""
        if not tools:
            return None
        
        gemini_functions = []
        for tool in tools:
            try:
                if tool.get("type") == "function" and "function" in tool:
                    func_info = tool["function"]
                    name = func_info.get("name")
                    description = func_info.get("description", "")
                    parameters = func_info.get("parameters")

                    if not name:
                        logger.warning("Function missing name, skipping")
                        continue
                    
                    if not parameters:
                        logger.warning(f"Function {name} missing parameters, using empty schema")
                        parameters = {"type": "object", "properties": {}}

                    function_declaration = FunctionDeclaration(
                        name=name,
                        description=description,
                        parameters=self._convert_schema_to_gemini(parameters)
                    )
                    gemini_functions.append(function_declaration)
                    logger.debug(f"Converted function: {name}")
            except Exception as e:
                logger.error(f"Error converting tool {tool}: {e}")
                continue
        
        if gemini_functions:
            return [GeminiTool(function_declarations=gemini_functions)]
        return None

    def convert_messages(self, messages: List[ChatMessage]) -> Tuple[List[ContentDict], Optional[str]]:
        """将OpenAI消息格式转换为Gemini消息格式"""
        gemini_messages = []
        system_prompt = None
        
        for i, msg in enumerate(messages):
            try:
                # 提取 system prompt
                if msg.role == "system":
                    if isinstance(msg.content, str):
                        system_prompt = msg.content
                    else:
                        logger.warning(f"System message {i} has non-string content, skipping")
                    continue
                
                # 确定角色映射
                role = "user" if msg.role in ["user", "tool"] else "model"
                parts = []
                
                # 处理文本内容
                if isinstance(msg.content, str) and msg.content:
                    parts.append(PartDict(text=msg.content))
                elif isinstance(msg.content, list):
                    # 处理多模态内容（如果需要的话）
                    for content_part in msg.content:
                        if isinstance(content_part, dict) and content_part.get("type") == "text":
                            text = content_part.get("text", "")
                            if text:
                                parts.append(PartDict(text=text))

                # 处理来自 assistant 的 tool_calls
                if msg.role == "assistant" and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        try:
                            if tool_call.get("type") == "function":
                                func = tool_call["function"]
                                function_name = func.get("name")
                                arguments_str = func.get("arguments", "{}")
                                
                                if not function_name:
                                    logger.warning("Tool call missing function name")
                                    continue
                                
                                # 解析参数
                                try:
                                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                                except json.JSONDecodeError as e:
                                    logger.error(f"Invalid JSON in tool call arguments: {e}")
                                    arguments = {}
                                
                                parts.append(PartDict(
                                    function_call={
                                        'name': function_name, 
                                        'args': arguments
                                    }
                                ))
                        except Exception as e:
                            logger.error(f"Error processing tool call: {e}")
                            continue

                # 处理来自 tool 角色的响应
                if msg.role == "tool":
                    function_name = msg.name or "unknown_function"
                    content = msg.content or ""
                    parts.append(PartDict(
                        function_response={
                            'name': function_name, 
                            'response': {'content': content}
                        }
                    ))

                # 添加到消息列表
                if parts:
                    gemini_messages.append(ContentDict(role=role, parts=parts))
                    logger.debug(f"Converted message {i}: role={role}, parts_count={len(parts)}")
                    
            except Exception as e:
                logger.error(f"Error converting message {i}: {e}")
                continue

        logger.info(f"Converted {len(messages)} OpenAI messages to {len(gemini_messages)} Gemini messages")
        if system_prompt:
            logger.debug(f"Extracted system prompt: {system_prompt[:100]}...")
            
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
        if "TOOL" in reason_str or "FUNCTION" in reason_str:
            return "tool_calls"
        if "STOP" in reason_str:
            return "stop"
        return "stop"

    def convert_response(self, gemini_response: genai.types.GenerateContentResponse, original_request: ChatCompletionRequest) -> Dict[str, Any]:
        """将Gemini响应转换为OpenAI格式"""
        model = original_request.model
        finish_reason = "stop"
        tool_calls = []
        text_content = ""

        try:
            if gemini_response.candidates:
                candidate = gemini_response.candidates[0]
                finish_reason = self._map_finish_reason(candidate.finish_reason)
                
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # 处理文本内容
                        if hasattr(part, 'text') and part.text:
                            text_content += part.text
                        
                        # 处理工具调用
                        if hasattr(part, 'function_call') and part.function_call:
                            try:
                                tool_calls.append({
                                    "id": f"call_{uuid.uuid4().hex}",
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(dict(part.function_call.args))
                                    }
                                })
                            except Exception as e:
                                logger.error(f"Error processing function call: {e}")

            # 构建消息对象
            message = {
                "role": "assistant",
                "content": text_content if text_content else None,
            }
            
            # 只有在有工具调用时才添加tool_calls字段
            if tool_calls:
                message["tool_calls"] = tool_calls

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
                "usage": usage
            }
            
            logger.debug(f"Converted response: finish_reason={finish_reason}, tool_calls={len(tool_calls)}")
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
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    async def convert_stream_response(self, gemini_stream, original_request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """将Gemini流式响应转换为OpenAI格式"""
        model = original_request.model
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        # 用于处理工具调用的状态变量
        tool_call_chunks = {}
        tool_call_index = 0
        
        # 跟踪是否已发送第一个包含角色的块
        first_chunk_sent = False

        try:
            async for chunk in gemini_stream:
                try:
                    if not chunk.candidates:
                        continue
                    
                    candidate = chunk.candidates[0]
                    delta = {}
                    
                    # 1. 发送角色块 (仅一次)
                    if not first_chunk_sent:
                        delta["role"] = "assistant"
                        first_chunk_sent = True
                    
                    # 2. 处理内容和工具调用
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # 处理文本增量
                            if hasattr(part, 'text') and part.text:
                                delta["content"] = part.text

                            # 处理工具调用增量
                            if hasattr(part, 'function_call') and part.function_call:
                                fc_name = part.function_call.name
                                fc_args = dict(part.function_call.args)
                                
                                # 为新的工具调用创建条目
                                if fc_name not in tool_call_chunks:
                                    tool_call_chunks[fc_name] = {
                                        "index": tool_call_index,
                                        "id": f"call_{uuid.uuid4().hex}",
                                        "type": "function",
                                        "function": {"name": fc_name, "arguments": ""}
                                    }
                                    tool_call_index += 1
                                
                                # 更新参数
                                tool_call_chunks[fc_name]["function"]["arguments"] = json.dumps(fc_args)
                                
                                # 构造增量格式
                                delta["tool_calls"] = [tool_call_chunks[fc_name]]
                    
                    # 3. 构造并发送块
                    if delta:
                        choice = {"index": 0, "delta": delta}
                        openai_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [choice]
                        }
                        yield f"data: {json.dumps(openai_chunk)}\n\n"

                    # 4. 处理结束原因
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason and candidate.finish_reason.name != "FINISH_REASON_UNSPECIFIED":
                        finish_reason = self._map_finish_reason(candidate.finish_reason)
                        final_choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
                        final_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [final_choice]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        break
                        
                except Exception as chunk_error:
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

        except Exception as stream_error:
            logger.error(f"Error in stream conversion: {stream_error}")
            # 发送错误块
            error_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Error: {str(stream_error)}"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # 5. 发送流结束标志
        yield "data: [DONE]\n\n"


class APIConfig:
    """持有转换器实例的配置类。"""
    def __init__(self):
        self.openai_to_gemini = OpenAIToGeminiConverter()
        self.gemini_to_openai = GeminiToOpenAIConverter()
        logger.info("🚀 OpenAI-Gemini API Compatibility Layer Initialized (Core Logic).")
