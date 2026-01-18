import json
import uuid
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from context_manager import ContextManager, TokenBasedContextManager

try:
    from .claude_types import ClaudeRequest, ClaudeMessage, ClaudeTool, ClaudeThinking
except ImportError:
    # Fallback for dynamic loading where relative import might fail
    # We assume claude_types is available in sys.modules or we can import it directly if in same dir
    import sys
    if "v2.claude_types" in sys.modules:
        from v2.claude_types import ClaudeRequest, ClaudeMessage, ClaudeTool
    else:
        # Try absolute import assuming v2 is in path or current dir
        try:
            from claude_types import ClaudeRequest, ClaudeMessage, ClaudeTool, ClaudeThinking
        except ImportError:
            # Last resort: if loaded via importlib in app.py, we might need to rely on app.py injecting it
            # But app.py loads this module.
            pass

logger = logging.getLogger(__name__)

MAX_THINKING_BUDGET = 24576
DEFAULT_THINKING_BUDGET = 20000

def clamp_thinking_budget(value: Optional[int]) -> int:
    """Clamp thinking budget into safe range."""
    if value is None:
        return DEFAULT_THINKING_BUDGET
    return max(256, min(MAX_THINKING_BUDGET, int(value)))

def generate_thinking_prefix(thinking: Optional["ClaudeThinking"]) -> Optional[str]:
    """Return thinking instruction tags when thinking is enabled."""
    if not thinking:
        return None
    if getattr(thinking, "thinking_type", "").lower() != "enabled":
        return None
    budget = clamp_thinking_budget(getattr(thinking, "budget_tokens", None))
    return (
        "<thinking_mode>enabled</thinking_mode>"
        f"<max_thinking_length>{budget}</max_thinking_length>"
    )

def has_thinking_tags(text: str) -> bool:
    """Check whether system prompt already contains thinking tags."""
    if not text:
        return False
    lowered = text.lower()
    return "<thinking_mode>" in lowered or "<max_thinking_length>" in lowered

def get_current_timestamp() -> str:
    """Get current timestamp in Amazon Q format."""
    now = datetime.now().astimezone()
    weekday = now.strftime("%A")
    iso_time = now.isoformat(timespec='milliseconds')
    return f"{weekday}, {iso_time}"

def map_model_name(claude_model: str) -> str:
    """Map Claude model name to Amazon Q model ID."""
    model_lower = claude_model.lower()
    if model_lower.startswith("claude-sonnet-4.5") or model_lower.startswith("claude-sonnet-4-5"):
        return "claude-sonnet-4.5"
    return "claude-sonnet-4"

def extract_text_from_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract text from Claude content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""

def extract_images_from_content(content: Union[str, List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Extract images from Claude content and convert to Amazon Q format."""
    if not isinstance(content, list):
        return None
    
    images = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                fmt = media_type.split("/")[-1] if "/" in media_type else "png"
                images.append({
                    "format": fmt,
                    "source": {
                        "bytes": source.get("data", "")
                    }
                })
    return images if images else None

def convert_tool(tool: ClaudeTool) -> Dict[str, Any]:
    """Convert Claude tool to Amazon Q tool."""
    desc = tool.description or ""
    if len(desc) > 10240:
        desc = desc[:10100] + "\n\n...(Full description provided in TOOL DOCUMENTATION section)"
    
    return {
        "toolSpecification": {
            "name": tool.name,
            "description": desc,
            "inputSchema": {"json": tool.input_schema}
        }
    }

def merge_user_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge consecutive user messages, keeping only the last 2 messages' images."""
    if not messages:
        return {}
    
    all_contents = []
    base_context = None
    base_origin = None
    base_model = None
    all_images = []
    
    for msg in messages:
        content = msg.get("content", "")
        if base_context is None:
            base_context = msg.get("userInputMessageContext", {})
        if base_origin is None:
            base_origin = msg.get("origin", "CLI")
        if base_model is None:
            base_model = msg.get("modelId")
        
        if content:
            all_contents.append(content)
        
        # Collect images from each message
        msg_images = msg.get("images")
        if msg_images:
            all_images.append(msg_images)
    
    result = {
        "content": "\n\n".join(all_contents),
        "userInputMessageContext": base_context or {},
        "origin": base_origin or "CLI",
        "modelId": base_model
    }
    
    # Only keep images from the last 2 messages that have images
    if all_images:
        kept_images = []
        for img_list in all_images[-2:]:  # Take last 2 messages' images
            kept_images.extend(img_list)
        if kept_images:
            result["images"] = kept_images
    
    return result

def process_history(messages: List[ClaudeMessage], max_messages: int = 20) -> List[Dict[str, Any]]:
    """Process history messages to match Amazon Q format (alternating user/assistant).

    Args:
        messages: List of Claude messages
        max_messages: Maximum number of messages to keep (default 20, 0 = unlimited)
    """
    # 智能压缩：已禁用
    if max_messages == 0:
        # 无限制模式，不压缩
        pass
    elif len(messages) > max_messages:
        # 有限制模式，简单截断
        print(f"[Context Truncate] {len(messages)} msgs → {max_messages} msgs")
        messages = messages[-max_messages:]

    history = []
    seen_tool_use_ids = set()
    
    raw_history = []
    
    # First pass: convert individual messages
    for msg in messages:
        if msg.role == "user":
            content = msg.content
            text_content = ""
            tool_results = None
            images = extract_images_from_content(content)
            
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            text_parts.append(block.get("text", ""))
                        elif btype == "tool_result":
                            if tool_results is None:
                                tool_results = []
                            
                            tool_use_id = block.get("tool_use_id")
                            raw_c = block.get("content", [])
                            
                            aq_content = []
                            if isinstance(raw_c, str):
                                aq_content = [{"text": raw_c}]
                            elif isinstance(raw_c, list):
                                for item in raw_c:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            aq_content.append({"text": item.get("text", "")})
                                        elif "text" in item:
                                            aq_content.append({"text": item["text"]})
                                    elif isinstance(item, str):
                                        aq_content.append({"text": item})
                            
                            if not any(i.get("text", "").strip() for i in aq_content):
                                aq_content = [{"text": "Tool use was cancelled by the user"}]
                                
                            # Merge if exists
                            existing = next((r for r in tool_results if r["toolUseId"] == tool_use_id), None)
                            if existing:
                                existing["content"].extend(aq_content)
                            else:
                                tool_results.append({
                                    "toolUseId": tool_use_id,
                                    "content": aq_content,
                                    "status": block.get("status", "success")
                                })
                text_content = "\n".join(text_parts)
            else:
                text_content = extract_text_from_content(content)
            
            user_ctx = {
                "envState": {
                    "operatingSystem": "macos",
                    "currentWorkingDirectory": "/"
                }
            }
            if tool_results:
                user_ctx["toolResults"] = tool_results
                
            u_msg = {
                "content": text_content,
                "userInputMessageContext": user_ctx,
                "origin": "CLI"
            }
            if images:
                u_msg["images"] = images
                
            raw_history.append({"userInputMessage": u_msg})
            
        elif msg.role == "assistant":
            content = msg.content
            text_content = extract_text_from_content(content)
            
            entry = {
                "assistantResponseMessage": {
                    "messageId": str(uuid.uuid4()),
                    "content": text_content
                }
            }
            
            if isinstance(content, list):
                tool_uses = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tid = block.get("id")
                        if tid and tid not in seen_tool_use_ids:
                            seen_tool_use_ids.add(tid)
                            tool_uses.append({
                                "toolUseId": tid,
                                "name": block.get("name"),
                                "input": block.get("input", {})
                            })
                if tool_uses:
                    entry["assistantResponseMessage"]["toolUses"] = tool_uses
            
            raw_history.append(entry)

    # Second pass: merge consecutive user messages
    pending_user_msgs = []
    for item in raw_history:
        if "userInputMessage" in item:
            pending_user_msgs.append(item["userInputMessage"])
        elif "assistantResponseMessage" in item:
            if pending_user_msgs:
                merged = merge_user_messages(pending_user_msgs)
                history.append({"userInputMessage": merged})
                pending_user_msgs = []
            history.append(item)
            
    if pending_user_msgs:
        merged = merge_user_messages(pending_user_msgs)
        history.append({"userInputMessage": merged})
        
    return history

def convert_claude_to_amazonq_request(req: ClaudeRequest, conversation_id: Optional[str] = None, max_history: int = 20) -> Dict[str, Any]:
    """Convert ClaudeRequest to Amazon Q request body.

    Args:
        req: Claude request
        conversation_id: Optional conversation ID
        max_history: Maximum number of history messages to keep (0 = unlimited)
    """
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        
    # 1. Tools
    aq_tools = []
    long_desc_tools = []
    if req.tools:
        for t in req.tools:
            if t.description and len(t.description) > 10240:
                long_desc_tools.append({"name": t.name, "full_description": t.description})
            aq_tools.append(convert_tool(t))
            
    # 2. Current Message (last user message)
    last_msg = req.messages[-1] if req.messages else None
    prompt_content = ""
    tool_results = None
    has_tool_result = False
    images = None
    
    if last_msg and last_msg.role == "user":
        content = last_msg.content
        images = extract_images_from_content(content)
        
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        has_tool_result = True
                        if tool_results is None:
                            tool_results = []
                        
                        tid = block.get("tool_use_id")
                        raw_c = block.get("content", [])
                        
                        aq_content = []
                        if isinstance(raw_c, str):
                            aq_content = [{"text": raw_c}]
                        elif isinstance(raw_c, list):
                            for item in raw_c:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        aq_content.append({"text": item.get("text", "")})
                                    elif "text" in item:
                                        aq_content.append({"text": item["text"]})
                                elif isinstance(item, str):
                                    aq_content.append({"text": item})
                                    
                        if not any(i.get("text", "").strip() for i in aq_content):
                            aq_content = [{"text": "Tool use was cancelled by the user"}]
                            
                        existing = next((r for r in tool_results if r["toolUseId"] == tid), None)
                        if existing:
                            existing["content"].extend(aq_content)
                        else:
                            tool_results.append({
                                "toolUseId": tid,
                                "content": aq_content,
                                "status": block.get("status", "success")
                            })
            prompt_content = "\n".join(text_parts)
        else:
            prompt_content = extract_text_from_content(content)
            
    # 3. Context
    user_ctx = {
        "envState": {
            "operatingSystem": "macos",
            "currentWorkingDirectory": "/"
        }
    }
    if aq_tools:
        user_ctx["tools"] = aq_tools
    if tool_results:
        user_ctx["toolResults"] = tool_results
        
    # 4. Format Content
    formatted_content = ""
    if has_tool_result and not prompt_content:
        formatted_content = ""
    else:
        formatted_content = (
            "--- CONTEXT ENTRY BEGIN ---\n"
            f"Current time: {get_current_timestamp()}\n"
            "--- CONTEXT ENTRY END ---\n\n"
            "--- USER MESSAGE BEGIN ---\n"
            f"{prompt_content}\n"
            "--- USER MESSAGE END ---"
        )
        
    if long_desc_tools:
        docs = []
        for info in long_desc_tools:
            docs.append(f"Tool: {info['name']}\nFull Description:\n{info['full_description']}\n")
        formatted_content = (
            "--- TOOL DOCUMENTATION BEGIN ---\n"
            f"{''.join(docs)}"
            "--- TOOL DOCUMENTATION END ---\n\n"
            f"{formatted_content}"
        )
        
    thinking_prefix = generate_thinking_prefix(req.thinking)

    system_text = ""
    if req.system:
        if isinstance(req.system, str):
            system_text = req.system
        elif isinstance(req.system, list):
            parts = []
            for b in req.system:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
            system_text = "\n".join(parts)

    if thinking_prefix:
        if system_text:
            if not has_thinking_tags(system_text):
                system_text = f"{thinking_prefix}\n{system_text}"
        else:
            system_text = thinking_prefix

    if system_text and formatted_content:
        formatted_content = (
            "--- SYSTEM PROMPT BEGIN ---\n"
            f"{system_text}\n"
            "--- SYSTEM PROMPT END ---\n\n"
            f"{formatted_content}"
        )
            
    # 5. Model
    model_id = map_model_name(req.model)
    
    # 6. User Input Message
    user_input_msg = {
        "content": formatted_content,
        "userInputMessageContext": user_ctx,
        "origin": "CLI",
        "modelId": model_id
    }
    if images:
        user_input_msg["images"] = images
        
    # 7. History
    history_msgs = req.messages[:-1] if len(req.messages) > 1 else []
    aq_history = process_history(history_msgs, max_messages=max_history)
    
    # 8. Final Body
    return {
        "conversationState": {
            "conversationId": conversation_id,
            "history": aq_history,
            "currentMessage": {
                "userInputMessage": user_input_msg
            },
            "chatTriggerType": "MANUAL"
        }
    }
