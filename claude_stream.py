import json
import logging
import importlib.util
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List
import tiktoken

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

try:
    # cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None

def count_tokens(text: str) -> int:
    """Counts tokens with tiktoken."""
    if not text or not ENCODING:
        return 0
    return len(ENCODING.encode(text))

# ------------------------------------------------------------------------------
# Dynamic Loader
# ------------------------------------------------------------------------------

def _load_claude_parser():
    """Dynamically load claude_parser module."""
    base_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("v2_claude_parser", str(base_dir / "claude_parser.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    _parser = _load_claude_parser()
    build_message_start = _parser.build_message_start
    build_content_block_start = _parser.build_content_block_start
    build_content_block_delta = _parser.build_content_block_delta
    build_content_block_stop = _parser.build_content_block_stop
    build_ping = _parser.build_ping
    build_message_stop = _parser.build_message_stop
    build_tool_use_start = _parser.build_tool_use_start
    build_tool_use_input_delta = _parser.build_tool_use_input_delta
    build_thinking_delta = _parser.build_thinking_delta
except Exception as e:
    logger.error(f"Failed to load claude_parser: {e}")
    # Fallback definitions
    def build_message_start(*args, **kwargs): return ""
    def build_content_block_start(*args, **kwargs): return ""
    def build_content_block_delta(*args, **kwargs): return ""
    def build_content_block_stop(*args, **kwargs): return ""
    def build_ping(*args, **kwargs): return ""
    def build_message_stop(*args, **kwargs): return ""
    def build_tool_use_start(*args, **kwargs): return ""
    def build_tool_use_input_delta(*args, **kwargs): return ""
    def build_thinking_delta(*args, **kwargs): return ""

THINKING_START_TAG = "<thinking>"
THINKING_END_TAG = "</thinking>"
QUOTE_CHARS = set("`\"'\\#!@$%^&*()-_=+[]{};:<>,.?/")
CONTEXT_WINDOW_TOKENS = 200_000

def _is_quote_char(buffer: str, pos: int) -> bool:
    if pos < 0 or pos >= len(buffer):
        return False
    return buffer[pos] in QUOTE_CHARS

def find_real_thinking_start_tag(buffer: str) -> Optional[int]:
    search_start = 0
    while True:
        pos = buffer.find(THINKING_START_TAG, search_start)
        if pos == -1:
            return None
        has_quote_before = pos > 0 and _is_quote_char(buffer, pos - 1)
        after_pos = pos + len(THINKING_START_TAG)
        has_quote_after = _is_quote_char(buffer, after_pos)
        if not has_quote_before and not has_quote_after:
            return pos
        search_start = pos + 1

def find_real_thinking_end_tag(buffer: str) -> Optional[int]:
    search_start = 0
    while True:
        pos = buffer.find(THINKING_END_TAG, search_start)
        if pos == -1:
            return None
        has_quote_before = pos > 0 and _is_quote_char(buffer, pos - 1)
        after_pos = pos + len(THINKING_END_TAG)
        has_quote_after = _is_quote_char(buffer, after_pos)
        if has_quote_before or has_quote_after:
            search_start = pos + 1
            continue
        after_content = buffer[after_pos:]
        if len(after_content) < 2:
            return None
        if after_content.startswith("\n\n"):
            return pos
        search_start = pos + 1

def find_real_thinking_end_tag_at_buffer_end(buffer: str) -> Optional[int]:
    search_start = 0
    while True:
        pos = buffer.find(THINKING_END_TAG, search_start)
        if pos == -1:
            return None
        has_quote_before = pos > 0 and _is_quote_char(buffer, pos - 1)
        after_pos = pos + len(THINKING_END_TAG)
        has_quote_after = _is_quote_char(buffer, after_pos)
        if has_quote_before or has_quote_after:
            search_start = pos + 1
            continue
        after_content = buffer[after_pos:]
        if after_content and any(not ch.isspace() for ch in after_content):
            search_start = pos + 1
            continue
        return pos

class ClaudeStreamHandler:
    def __init__(
        self,
        model: str,
        input_tokens: int = 0,
        message_id: Optional[str] = None,
        thinking_enabled: bool = False,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.message_id = message_id or f"msg_{uuid.uuid4()}"
        self.message_start_sent = False
        self.conversation_id: Optional[str] = None
        self.next_block_index = -1
        self.block_states: Dict[int, Dict[str, Any]] = {}
        self.text_block_index: Optional[int] = None
        self.thinking_block_index: Optional[int] = None
        self.tool_blocks: Dict[str, int] = {}
        self.tool_inputs: Dict[str, List[str]] = {}
        self.response_buffer: List[str] = []
        self.thinking_segments: List[str] = []
        self.all_tool_inputs: List[str] = []
        self.thinking_enabled = thinking_enabled
        self.thinking_buffer = ""
        self.in_thinking_block = False
        self.thinking_extracted = False
        self.context_input_tokens: Optional[int] = None

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        if event_type == "initial-response":
            if not self.message_start_sent:
                conv_id = payload.get("conversationId", self.conversation_id or "unknown")
                self.conversation_id = conv_id
                yield build_message_start(self.message_id, self.model, self.input_tokens, conv_id)
                self.message_start_sent = True
                yield build_ping()

        elif event_type == "assistantResponseEvent":
            content = payload.get("content", "")
            async for evt in self._handle_assistant_response(content):
                yield evt

        elif event_type == "toolUseEvent":
            async for evt in self._handle_tool_use(payload):
                yield evt

        elif event_type == "assistantResponseEnd":
            for evt in self._finalize_thinking_buffer():
                yield evt
            for evt in self._close_text_block():
                yield evt
        elif event_type == "contextUsageEvent":
            self._handle_context_usage(payload)

    async def _handle_assistant_response(self, content: str) -> AsyncGenerator[str, None]:
        if not content:
            return
        if self.thinking_enabled:
            for evt in self._process_content_with_thinking(content):
                yield evt
        else:
            for evt in self._emit_text_delta(content):
                yield evt

    def _process_content_with_thinking(self, content: str) -> List[str]:
        emitted: List[str] = []
        self.thinking_buffer += content

        while True:
            if not self.in_thinking_block and not self.thinking_extracted:
                start_pos = find_real_thinking_start_tag(self.thinking_buffer)
                if start_pos is not None:
                    before = self.thinking_buffer[:start_pos]
                    if before:
                        emitted.extend(self._emit_text_delta(before))
                    self.thinking_buffer = self.thinking_buffer[start_pos + len(THINKING_START_TAG):]
                    emitted.extend(self._start_thinking_block())
                    self.in_thinking_block = True
                else:
                    safe_len = max(0, len(self.thinking_buffer) - len(THINKING_START_TAG))
                    if safe_len > 0:
                        chunk = self.thinking_buffer[:safe_len]
                        emitted.extend(self._emit_text_delta(chunk))
                        self.thinking_buffer = self.thinking_buffer[safe_len:]
                    break
            elif self.in_thinking_block:
                end_pos = find_real_thinking_end_tag(self.thinking_buffer)
                if end_pos is not None:
                    thinking_content = self.thinking_buffer[:end_pos]
                    if thinking_content:
                        emitted.extend(self._emit_thinking_delta(thinking_content))
                    self.thinking_buffer = self.thinking_buffer[end_pos + len(THINKING_END_TAG):]
                    emitted.extend(self._finish_thinking_block())
                    self.in_thinking_block = False
                    self.thinking_extracted = True
                else:
                    safe_len = max(0, len(self.thinking_buffer) - len(THINKING_END_TAG))
                    if safe_len > 0:
                        chunk = self.thinking_buffer[:safe_len]
                        emitted.extend(self._emit_thinking_delta(chunk))
                        self.thinking_buffer = self.thinking_buffer[safe_len:]
                    break
            else:
                if self.thinking_buffer:
                    emitted.extend(self._emit_text_delta(self.thinking_buffer))
                    self.thinking_buffer = ""
                break

        return emitted

    async def _handle_tool_use(self, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        tool_use_id = payload.get("toolUseId")
        tool_name = payload.get("name")
        tool_input = payload.get("input", {})
        is_stop = payload.get("stop", False)

        if tool_use_id and tool_name and tool_use_id not in self.tool_blocks:
            for evt in self._finalize_thinking_buffer():
                yield evt
            for evt in self._close_text_block():
                yield evt
            idx = self._next_index()
            self.block_states[idx] = {"type": "tool_use", "stopped": False}
            self.tool_blocks[tool_use_id] = idx
            self.tool_inputs[tool_use_id] = []
            yield build_tool_use_start(idx, tool_use_id, tool_name)

        if tool_use_id and tool_use_id in self.tool_blocks:
            idx = self.tool_blocks[tool_use_id]
            if tool_input:
                fragment = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
                self.tool_inputs.setdefault(tool_use_id, []).append(fragment)
                yield build_tool_use_input_delta(idx, fragment)

            if is_stop:
                collected = "".join(self.tool_inputs.get(tool_use_id, []))
                if collected:
                    self.all_tool_inputs.append(collected)
                yield build_content_block_stop(idx)
                self.block_states[idx]["stopped"] = True
                self.tool_blocks.pop(tool_use_id, None)
                self.tool_inputs.pop(tool_use_id, None)

    def _next_index(self) -> int:
        self.next_block_index += 1
        return self.next_block_index

    def _start_text_block(self) -> List[str]:
        events: List[str] = []
        idx = self.text_block_index
        if idx is not None and self.block_states.get(idx, {}).get("stopped"):
            idx = None
            self.text_block_index = None
        if idx is None:
            idx = self._next_index()
            self.text_block_index = idx
            self.block_states[idx] = {"type": "text", "stopped": False}
            events.append(build_content_block_start(idx, "text"))
        return events

    def _close_text_block(self) -> List[str]:
        events: List[str] = []
        idx = self.text_block_index
        if idx is not None and not self.block_states.get(idx, {}).get("stopped"):
            events.append(build_content_block_stop(idx))
            self.block_states[idx]["stopped"] = True
        self.text_block_index = None
        return events

    def _emit_text_delta(self, text: str) -> List[str]:
        if not text:
            return []
        events = self._start_text_block()
        idx = self.text_block_index
        if idx is not None:
            events.append(build_content_block_delta(idx, text))
        self.response_buffer.append(text)
        return events

    def _start_thinking_block(self) -> List[str]:
        events: List[str] = []
        idx = self.thinking_block_index
        if idx is not None and self.block_states.get(idx, {}).get("stopped"):
            idx = None
            self.thinking_block_index = None
        if idx is None:
            idx = self._next_index()
            self.thinking_block_index = idx
            self.block_states[idx] = {"type": "thinking", "stopped": False}
            events.append(build_content_block_start(idx, "thinking"))
        return events

    def _emit_thinking_delta(self, thinking: str) -> List[str]:
        if not thinking:
            return []
        events = self._start_thinking_block()
        idx = self.thinking_block_index
        if idx is not None:
            events.append(build_thinking_delta(idx, thinking))
        self.thinking_segments.append(thinking)
        return events

    def _finish_thinking_block(self) -> List[str]:
        events: List[str] = []
        idx = self.thinking_block_index
        if idx is not None and not self.block_states.get(idx, {}).get("stopped"):
            events.append(build_thinking_delta(idx, ""))
            events.append(build_content_block_stop(idx))
            self.block_states[idx]["stopped"] = True
        self.thinking_block_index = None
        return events

    def _finalize_thinking_buffer(self) -> List[str]:
        if not self.thinking_enabled or not self.thinking_buffer and not self.in_thinking_block:
            return []
        events: List[str] = []
        if self.in_thinking_block:
            end_pos = find_real_thinking_end_tag_at_buffer_end(self.thinking_buffer)
            if end_pos is not None:
                thinking_content = self.thinking_buffer[:end_pos]
                if thinking_content:
                    events.extend(self._emit_thinking_delta(thinking_content))
                self.thinking_buffer = self.thinking_buffer[end_pos + len(THINKING_END_TAG):]
            if self.thinking_buffer:
                events.extend(self._emit_thinking_delta(self.thinking_buffer))
                self.thinking_buffer = ""
            events.extend(self._finish_thinking_block())
            self.in_thinking_block = False
            self.thinking_extracted = True
        elif not self.thinking_extracted and self.thinking_buffer:
            events.extend(self._emit_text_delta(self.thinking_buffer))
            self.thinking_buffer = ""
        return events

    def _handle_context_usage(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        percentage = (
            payload.get("contextUsagePercentage")
            or payload.get("context_usage_percentage")
            or payload.get("percentage")
        )
        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            return
        if percentage <= 0:
            return
        tokens = int((percentage / 100.0) * CONTEXT_WINDOW_TOKENS)
        if tokens > 0:
            self.context_input_tokens = tokens

    async def finish(self) -> AsyncGenerator[str, None]:
        for evt in self._finalize_thinking_buffer():
            yield evt
        for evt in self._close_text_block():
            yield evt
        for tool_id, idx in list(self.tool_blocks.items()):
            if not self.block_states.get(idx, {}).get("stopped"):
                yield build_content_block_stop(idx)
            self.block_states[idx]["stopped"] = True
            self.tool_blocks.pop(tool_id, None)
            self.tool_inputs.pop(tool_id, None)

        full_text = "".join(self.response_buffer)
        full_thinking = "".join(self.thinking_segments)
        full_tool_input = "".join(self.all_tool_inputs)
        output_tokens = (
            count_tokens(full_text)
            + count_tokens(full_thinking)
            + count_tokens(full_tool_input)
        )
        input_tokens = self.context_input_tokens or self.input_tokens
        yield build_message_stop(input_tokens, output_tokens, "end_turn")
