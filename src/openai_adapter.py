# src/openai_adapter.py - 修复版本
import json
import uuid
import time
import re
from typing import List, Dict, Optional, Any, Union, AsyncGenerator, Literal, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
import logging

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, ContentDict, PartDict, Tool as GeminiTool, FunctionDeclaration
from google.ai.generativelanguage_v1beta.types import content as glm_content

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型映射配置（外部化）
MODEL_MAPPING = {
    "gpt-4o": "gemini-1.5-pro-latest",
    "gpt-4o-mini": "gemini-1.5-flash-latest",
    "gpt-4-turbo": "gemini-1.5-pro-latest",
    "gpt-4": "gemini-1.5-pro-latest",
    "gpt-3.5-turbo": "gemini-1.5-flash-latest",
    "gpt-4-1106-preview": "gemini-1.5-pro-latest",
    "gpt-4-0125-preview": "gemini-1.5-pro-latest",
    "gpt-3.5-turbo-1106": "gemini-1.5-flash-latest",
}

# ========== OpenAI API 数据模型 - 增强版 ==========
class ToolFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @field_validator('content', mode='before')
    @classmethod
    def validate_content(cls, v, values):
        """验证消息内容的有效性"""
        if hasattr(values, 'data'):
            role = values.data.get('role') if hasattr(values, 'data') else values.get('role')
        else:
            role = values.get('role')
        # tool角色的消息必须有content
        if role == 'tool' and not v:
            raise ValueError("Tool messages must have content")
        return v

    @field_validator('tool_calls', mode='before')
    @classmethod
    def validate_tool_calls(cls, v, values):
        """验证工具调用只能在assistant消息中使用"""
        if v is not None:
            if hasattr(values, 'data'):
                role = values.data.get('role') if hasattr(values, 'data') else values.get('role')
            else:
                role = values.get('role')
            if role != 'assistant':
                raise ValueError("Only assistant messages can have tool_calls")
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    response_format: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'

    @field_validator('tools', mode='before')
    @classmethod
    def validate_tools(cls, v):
        if v is not None:
            for i, tool in enumerate(v):
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool {i} must be a dictionary")
                if tool.get("type") != "function":
                    raise ValueError(f"Tool {i}: Only function tools are supported")
                function = tool.get("function", {})
                if not function.get("name"):
                    raise ValueError(f"Tool {i}: Function must have a name")
                # 验证参数schema
                parameters = function.get("parameters", {})
                if not isinstance(parameters, dict):
                    raise ValueError(f"Tool {i}: Function parameters must be a dictionary")
        return v

    @model_validator(mode='after')
    def validate_tools_and_choice(self):
        """验证工具和工具选择的一致性"""
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice can only be set when tools are provided")
        return self


# ========== 转换器类 - 修复版 ==========

class OpenAIToGeminiConverter:
    """
    将 OpenAI API 请求格式转换为 Google Gemini API 格式。
    修复了工具调用和消息转换的各种问题。
    """

    def convert_model(self, openai_model: str) -> str:
        """将OpenAI模型名称映射到Gemini模型名称（使用外部配置）"""
        mapped_model = MODEL_MAPPING.get(openai_model, "gemini-1.5-pro-latest")
        logger.debug(f"Mapped model {openai_model} -> {mapped_model}")
        return mapped_model

    def _convert_schema_to_gemini(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归地将OpenAI的JSON Schema转换为Gemini的格式（修复了null类型处理）。"""
        if not isinstance(json_schema, dict):
            logger.warning("Schema is not a dictionary, returning empty schema")
            return {"type": "OBJECT", "properties": {}}

        # 更精确的类型映射
        type_mapping = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "object": "OBJECT",
            "array": "ARRAY",
            # 修复：将null类型映射为STRING，但添加特殊标记
            "null": "STRING"
        }

        schema_type = json_schema.get("type", "object").lower()
        gemini_type = type_mapping.get(schema_type, "STRING")

        gemini_schema = {"type": gemini_type}

        # 复制基本属性
        if "description" in json_schema and json_schema["description"]:
            description = str(json_schema["description"])
            # 修复：为null类型添加特殊说明
            if schema_type == "null":
                description += " (Note: null values should be represented as empty strings)"
            gemini_schema["description"] = description
        elif schema_type == "null":
            gemini_schema["description"] = "Null value (use empty string)"

        # 处理格式限制
        if "format" in json_schema:
            gemini_schema["format"] = str(json_schema["format"])

        # 处理枚举（修复：为null类型特殊处理）
        if "enum" in json_schema and isinstance(json_schema["enum"], list):
            enum_values = []
            for item in json_schema["enum"]:
                if item is None:
                    enum_values.append("")  # 将null转换为空字符串
                else:
                    enum_values.append(str(item))
            gemini_schema["enum"] = enum_values

        # 处理数值范围
        for key in ["minimum", "maximum", "minLength", "maxLength"]:
            if key in json_schema and isinstance(json_schema[key], (int, float)):
                gemini_schema[key.lower()] = json_schema[key]

        # 处理对象类型
        if gemini_type == "OBJECT":
            properties = json_schema.get("properties", {})
            if isinstance(properties, dict):
                gemini_schema["properties"] = {
                    k: self._convert_schema_to_gemini(v)
                    for k, v in properties.items()
                    if isinstance(v, dict)
                }
            else:
                gemini_schema["properties"] = {}

            required = json_schema.get("required", [])
            if isinstance(required, list) and required:
                gemini_schema["required"] = [str(item) for item in required]

        # 处理数组类型
        elif gemini_type == "ARRAY":
            items = json_schema.get("items")
            if isinstance(items, dict):
                gemini_schema["items"] = self._convert_schema_to_gemini(items)
            elif isinstance(items, list) and items:
                # 如果items是数组，取第一个作为模板
                gemini_schema["items"] = self._convert_schema_to_gemini(items[0])
            else:
                # 默认数组项类型
                gemini_schema["items"] = {"type": "STRING"}

        return gemini_schema

    def _convert_tool_choice_to_tool_config(self, tool_choice: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """将OpenAI的tool_choice转换为Gemini的tool_config"""
        if tool_choice is None or tool_choice == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}

        if tool_choice == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        
        if tool_choice == "required":
             # Gemini doesn't have a direct equivalent for "required" like OpenAI.
             # "ANY" is the closest, as it allows the model to decide which function to call.
            logger.warning("OpenAI 'tool_choice: required' is mapped to Gemini 'mode: ANY'.")
            return {"function_calling_config": {"mode": "ANY"}}

        # 处理指定特定函数的情况
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_name = tool_choice.get("function", {}).get("name")
                if function_name:
                    return {
                        "function_calling_config": {
                            "mode": "ANY",
                            "allowed_function_names": [function_name]
                        }
                    }

        # 如果是字符串且不是 "auto" 或 "none"，假设是函数名
        if isinstance(tool_choice, str) and tool_choice not in ["auto", "none"]:
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [tool_choice]
                }
            }

        logger.warning(f"Unsupported tool_choice format: {tool_choice}, using AUTO")
        return {"function_calling_config": {"mode": "AUTO"}}

    def convert_tools(self, tools: Optional[List[Dict]], tool_choice: Optional[Union[str, Dict[str, Any]]] = None) -> Tuple[Optional[List[GeminiTool]], Optional[Dict[str, Any]]]:
        """
        将OpenAI工具定义转换为Gemini工具定义和工具配置
        修复：现在正确处理tool_choice参数
        返回：(gemini_tools, tool_config)
        """
        if not tools:
            return None, None

        gemini_functions = []
        for i, tool in enumerate(tools):
            try:
                if not isinstance(tool, dict):
                    logger.warning(f"Tool {i} is not a dictionary, skipping")
                    continue

                if tool.get("type") != "function":
                    logger.warning(f"Tool {i} is not a function type (got: {tool.get('type')}), skipping")
                    continue

                func_info = tool.get("function", {})
                if not isinstance(func_info, dict):
                    logger.warning(f"Tool {i} function info is not a dictionary, skipping")
                    continue

                name = func_info.get("name")
                if not name or not isinstance(name, str) or not name.strip():
                    logger.warning(f"Tool {i} missing valid name, skipping")
                    continue

                # 修复：更严格的函数名称格式验证（只允许字母、数字、下划线）
                if not re.match(r'^[a-zA-Z0-9_]+$', name):
                    logger.warning(f"Tool {i} has invalid function name format: {name}. Only alphanumeric characters and underscores are allowed.")
                    continue

                description = func_info.get("description", "")
                if not isinstance(description, str):
                    description = str(description) if description else f"Function: {name}"

                parameters = func_info.get("parameters", {"type": "object", "properties": {}})
                if not isinstance(parameters, dict):
                    logger.warning(f"Function {name} has invalid parameters, using empty schema")
                    parameters = {"type": "object", "properties": {}}

                # 创建函数声明，添加更好的错误处理
                try:
                    converted_params = self._convert_schema_to_gemini(parameters)

                    function_declaration = FunctionDeclaration(
                        name=name,
                        description=description or f"Function: {name}",
                        parameters=converted_params
                    )
                    gemini_functions.append(function_declaration)
                    logger.debug(f"Successfully converted function: {name}")

                except Exception as e:
                    logger.error(f"Error creating FunctionDeclaration for {name}: {e}")
                    logger.debug(f"Function parameters: {parameters}")
                    continue

            except Exception as e:
                logger.error(f"Error converting tool {i}: {e}")
                continue

        gemini_tools = None
        tool_config = self._convert_tool_choice_to_tool_config(tool_choice)

        if gemini_functions:
            logger.info(f"Converted {len(gemini_functions)} out of {len(tools)} functions to Gemini format")
            gemini_tools = [GeminiTool(function_declarations=gemini_functions)]
            logger.debug(f"Converted tool_choice {tool_choice} to tool_config: {tool_config}")
        else:
            logger.warning("No valid functions found in tools")

        return gemini_tools, tool_config

    def convert_messages(self, messages: List[ChatMessage]) -> Tuple[List[ContentDict], Optional[str]]:
        """将OpenAI消息格式转换为Gemini消息格式，优化工具调用处理"""
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
                elif isinstance(msg.content, list):
                    # 处理多模态系统消息
                    for content_part in msg.content:
                        if isinstance(content_part, dict) and content_part.get("type") == "text":
                            text = content_part.get("text", "")
                            if text and text.strip():
                                system_contents.append(text.strip())

            if system_contents:
                system_prompt = "\n\n".join(system_contents)
                logger.debug(f"Extracted system prompt: {system_prompt[:100]}...")

        # 处理非系统消息
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
                    # 处理多模态内容（优化）
                    for content_part in msg.content:
                        if isinstance(content_part, dict):
                            content_type = content_part.get("type")
                            if content_type == "text":
                                text = content_part.get("text", "")
                                if text:
                                    parts.append(PartDict(text=str(text)))
                            elif content_type == "image_url":
                                # 支持图片内容（Gemini 1.5支持）
                                image_url = content_part.get("image_url", {})
                                url = image_url.get("url", "")
                                if url:
                                    if url.startswith("data:image"):
                                        # Base64图片
                                        try:
                                            import base64
                                            # 解析data URL
                                            header, data = url.split(",", 1)
                                            base64.b64decode(data) # Just to validate
                                            parts.append(PartDict(inline_data={
                                                "mime_type": header.split(";")[0].split(":")[1],
                                                "data": data
                                            }))
                                        except Exception as e:
                                            logger.warning(f"Failed to process base64 image: {e}")
                                    else:
                                        # URL图片（注意：Gemini可能不支持外部URL）
                                        logger.warning("External image URLs may not be supported by Gemini")

                # 处理assistant的工具调用（优化）
                if msg.role == "assistant" and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        try:
                            if isinstance(tool_call, dict):
                                # 处理字典格式的tool_call
                                func_info = tool_call.get("function", {})
                            elif hasattr(tool_call, 'function'):
                                # 处理ToolCall对象
                                func_info = {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            else:
                                logger.warning("Invalid tool_call format")
                                continue

                            function_name = func_info.get("name")
                            arguments = func_info.get("arguments", "{}")

                            if not function_name:
                                logger.warning("Tool call missing function name")
                                continue

                            # 解析参数（更健壮的处理）
                            try:
                                if isinstance(arguments, str):
                                    if arguments.strip():
                                        parsed_args = json.loads(arguments)
                                    else:
                                        parsed_args = {}
                                elif isinstance(arguments, dict):
                                    parsed_args = arguments
                                else:
                                    logger.warning(f"Unexpected arguments type: {type(arguments)}")
                                    parsed_args = {}
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON in tool call arguments for {function_name}: {e}")
                                logger.debug(f"Arguments string: {arguments}")
                                parsed_args = {}

                            parts.append(PartDict(
                                function_call=glm_content.FunctionCall(
                                    name=function_name,
                                    args=parsed_args
                                )
                            ))
                            logger.debug(f"Added function call: {function_name} with args: {list(parsed_args.keys())}")

                        except Exception as e:
                            logger.error(f"Error processing tool call: {e}")
                            continue

                # 处理tool角色的响应（优化）
                if msg.role == "tool":
                    function_name = msg.name or "unknown_function"
                    content = msg.content or ""

                    # 处理工具响应内容
                    try:
                        if isinstance(content, str):
                            # 尝试解析为JSON，但保留原始字符串作为备选
                            try:
                                parsed_content = json.loads(content)
                                response_content = parsed_content
                            except json.JSONDecodeError:
                                response_content = {"result": content}
                        elif isinstance(content, (dict, list)):
                            response_content = content
                        else:
                            response_content = {"result": str(content)}
                    except Exception:
                        response_content = {"result": str(content)}

                    parts.append(PartDict(
                        function_response=glm_content.FunctionResponse(
                            name=function_name,
                            response=response_content
                        )
                    ))
                    logger.debug(f"Added function response: {function_name}")

                # 如果没有内容但是用户消息，添加空文本
                if not parts and role == "user":
                    parts.append(PartDict(text=""))

                # 添加到消息列表
                if parts:
                    gemini_messages.append(ContentDict(role=role, parts=parts))
                    logger.debug(f"Converted message {i}: role={role}, parts_count={len(parts)}")
                elif role == "user":
                    # 用户消息即使为空也要添加
                    gemini_messages.append(ContentDict(role=role, parts=[PartDict(text="")]))

            except Exception as e:
                logger.error(f"Error converting message {i}: {e}")
                continue

        logger.info(f"Converted {len(messages)} OpenAI messages to {len(gemini_messages)} Gemini messages")
        return gemini_messages, system_prompt


class GeminiToOpenAIConverter:
    """
    将 Gemini API 响应格式转换为 OpenAI API 响应格式。
    修复了流式工具调用参数聚合的严重问题。
    """

    def _map_finish_reason(self, reason) -> str:
        """映射Gemini的结束原因到OpenAI格式"""
        if not reason:
            return "stop"

        reason_str = str(reason).upper()
        if "MAX_TOKENS" in reason_str or "LENGTH" in reason_str:
            return "length"
        elif "TOOL_CALLS" in reason_str or "TOOL" in reason_str or "FUNCTION" in reason_str:
            return "tool_calls"
        elif "SAFETY" in reason_str or "BLOCKED" in reason_str:
            return "content_filter"
        elif "STOP" in reason_str or "FINISH" in reason_str:
            return "stop"
        else:
            return "stop"

    def convert_response(self, gemini_response: genai.types.GenerateContentResponse, original_request: ChatCompletionRequest) -> Dict[str, Any]:
        """将Gemini响应转换为OpenAI格式，优化工具调用处理"""
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

                        # 处理工具调用（优化）
                        if hasattr(part, 'function_call') and part.function_call:
                            try:
                                tool_call_id = f"call_{uuid.uuid4().hex}"
                                function_name = part.function_call.name

                                # 更安全的参数处理
                                function_args = {}
                                if hasattr(part.function_call, 'args') and part.function_call.args:
                                    try:
                                        # Gemini的args可能是多种格式
                                        args = part.function_call.args
                                        if isinstance(args, dict):
                                            function_args = args
                                        elif hasattr(args, 'items'):
                                            # 处理protobuf字典类型
                                            function_args = dict(args.items())
                                        else:
                                            function_args = dict(args) if args else {}
                                    except Exception as e:
                                        logger.error(f"Error processing function args: {e}")
                                        function_args = {}

                                tool_calls.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": json.dumps(function_args, ensure_ascii=False)
                                    }
                                })
                                logger.debug(f"Converted function call: {function_name} with args: {list(function_args.keys())}")

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
                # 如果有工具调用但没有文本内容，设置finish_reason为tool_calls
                if not text_content:
                    finish_reason = "tool_calls"

            choice = {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }

            # 处理使用统计（优化）
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(gemini_response, 'usage_metadata') and gemini_response.usage_metadata:
                metadata = gemini_response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(metadata, 'total_token_count', 0)
                }

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [choice],
                "usage": usage,
                "system_fingerprint": f"gemini-{int(time.time())}"
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
        """
        将Gemini流式响应转换为OpenAI格式
        修复：完全重构了工具调用参数的聚合逻辑，使用字典而非字符串拼接
        """
        model = original_request.model
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        # 修复：用字典跟踪工具调用的参数状态，并记录上次发送的字符串以计算增量
        # {tool_call_index: {"id": str, "name": str, "args": dict, "last_sent_str": str}}
        active_tool_calls: Dict[int, Dict] = {}
        tool_call_index_counter = 0
        first_chunk_sent = False
        content_buffer = ""

        try:
            async for chunk in gemini_stream:
                try:
                    if not chunk.candidates:
                        continue
                    
                    candidate = chunk.candidates[0]
                    
                    # 发送初始角色块
                    if not first_chunk_sent:
                        delta = {"role": "assistant"}
                        choice = {"index": 0, "delta": delta}
                        openai_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [choice]
                        }
                        yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                        first_chunk_sent = True

                    delta = {}
                    # 处理内容
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # 处理文本内容（增量发送）
                            if hasattr(part, 'text') and part.text:
                                new_content = part.text
                                # 只发送新增的内容
                                if len(new_content) > len(content_buffer):
                                    delta_content = new_content[len(content_buffer):]
                                    if delta_content:
                                        delta["content"] = delta_content
                                        content_buffer = new_content

                            # 处理工具调用（增量发送）
                            if hasattr(part, 'function_call') and part.function_call:
                                func_call = part.function_call
                                func_name = func_call.name
                                
                                # 查找此工具调用是否已开始
                                current_tool_index = -1
                                for idx, tool in active_tool_calls.items():
                                    if tool["name"] == func_name:
                                        current_tool_index = idx
                                        break

                                # 如果是新的工具调用
                                if current_tool_index == -1:
                                    current_tool_index = tool_call_index_counter
                                    tool_call_id = f"call_{uuid.uuid4().hex}"
                                    active_tool_calls[current_tool_index] = {
                                        "id": tool_call_id,
                                        "name": func_name,
                                        "args": {},
                                        "last_sent_str": ""
                                    }
                                    if "tool_calls" not in delta: delta["tool_calls"] = []
                                    delta["tool_calls"].append({
                                        "index": current_tool_index,
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {"name": func_name, "arguments": ""}
                                    })
                                    tool_call_index_counter += 1
                                
                                # 安全地获取并更新参数
                                if hasattr(func_call, 'args') and func_call.args:
                                    # Gemini的args是部分更新，所以我们需要合并
                                    new_args = dict(func_call.args.items())
                                    tool_state = active_tool_calls[current_tool_index]
                                    tool_state["args"].update(new_args)

                                    # 计算参数字符串的增量
                                    new_full_args_str = json.dumps(tool_state["args"], ensure_ascii=False, sort_keys=True)
                                    last_sent_str = tool_state["last_sent_str"]
                                    
                                    if len(new_full_args_str) > len(last_sent_str):
                                        arg_delta = new_full_args_str[len(last_sent_str):]
                                        tool_state["last_sent_str"] = new_full_args_str

                                        # 将增量添加到delta中
                                        if "tool_calls" not in delta: delta["tool_calls"] = []
                                        
                                        tool_delta_found = False
                                        for tc in delta["tool_calls"]:
                                            if tc.get("index") == current_tool_index:
                                                tc["function"]["arguments"] = arg_delta
                                                tool_delta_found = True
                                                break
                                        
                                        if not tool_delta_found:
                                            delta["tool_calls"].append({
                                                "index": current_tool_index,
                                                "function": {"arguments": arg_delta}
                                            })

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
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason and str(candidate.finish_reason) != "FinishReason.FINISH_REASON_UNSPECIFIED":
                        finish_reason = self._map_finish_reason(candidate.finish_reason)
                        final_choice = {"index": 0, "delta": {}, "finish_reason": finish_reason}
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
                    logger.error(f"Error processing stream chunk: {chunk_error}", exc_info=True)
                    continue

        except Exception as stream_error:
            logger.error(f"Fatal error in stream conversion: {stream_error}", exc_info=True)
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
        finally:
            # 发送流结束标志
            yield "data: [DONE]\n\n"
            logger.info(f"Stream finished for chat ID: {chat_id}")

class APIConfig:
    """持有转换器实例的配置类。"""
    def __init__(self):
        self.openai_to_gemini = OpenAIToGeminiConverter()
        self.gemini_to_openai = GeminiToOpenAIConverter()
        logger.info("🚀 OpenAI-Gemini API Compatibility Layer Initialized (Fixed Version).")
