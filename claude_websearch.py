"""WebSearch tool handling for Claude API compatibility."""
import json
import uuid
import logging
from typing import Optional, Dict, Any, List
from claude_types import ClaudeRequest

logger = logging.getLogger(__name__)


def has_web_search_tool(req: ClaudeRequest) -> bool:
    """Check if request has only websearch tool (pure websearch request)."""
    if not req.tools:
        return False
    return len(req.tools) == 1 and req.tools[0].name == "web_search"


def extract_search_query(req: ClaudeRequest) -> Optional[str]:
    """Extract search query from messages."""
    if not req.messages:
        return None

    first_msg = req.messages[0]
    content = first_msg.content

    # Extract text
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                break
        else:
            return None
    else:
        return None

    # Remove prefix if present
    prefix = "Perform a web search for the query: "
    if text.startswith(prefix):
        text = text[len(prefix):]

    return text.strip() or None


def create_mcp_request(query: str) -> tuple[str, Dict[str, Any]]:
    """Create MCP request for websearch."""
    request_id = f"web_search_{uuid.uuid4().hex[:16]}_{int(uuid.uuid4().int % 1000000)}"
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex[:32]}"

    request = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {"query": query}
        }
    }

    return tool_use_id, request


def generate_websearch_events(
    model: str,
    query: str,
    tool_use_id: str,
    search_results: Optional[Dict[str, Any]],
    input_tokens: int
) -> List[str]:
    """Generate SSE events for websearch response."""
    events = []
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # 1. message_start
    data = json.dumps({
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'model': model,
            'content': [],
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {'input_tokens': input_tokens, 'output_tokens': 0}
        }
    })
    events.append(f"event: message_start\ndata: {data}\n\n")

    # 2. content_block_start (server_tool_use)
    data = json.dumps({
        'type': 'content_block_start',
        'index': 0,
        'content_block': {'id': tool_use_id, 'type': 'server_tool_use', 'name': 'web_search', 'input': {}}
    })
    events.append(f"event: content_block_start\ndata: {data}\n\n")

    # 3. content_block_delta (input_json_delta)
    input_json = json.dumps({"query": query})
    data = json.dumps({
        'type': 'content_block_delta',
        'index': 0,
        'delta': {'type': 'input_json_delta', 'partial_json': input_json}
    })
    events.append(f"event: content_block_delta\ndata: {data}\n\n")

    # 4. content_block_stop
    data = json.dumps({'type': 'content_block_stop', 'index': 0})
    events.append(f"event: content_block_stop\ndata: {data}\n\n")

    # 5. content_block_start (web_search_tool_result)
    search_content = []
    if search_results and "results" in search_results:
        for r in search_results["results"]:
            search_content.append({
                "type": "web_search_result",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "encrypted_content": r.get("snippet", ""),
                "page_age": None
            })

    data = json.dumps({
        'type': 'content_block_start',
        'index': 1,
        'content_block': {'type': 'web_search_tool_result', 'tool_use_id': tool_use_id, 'content': search_content}
    })
    events.append(f"event: content_block_start\ndata: {data}\n\n")

    # 6. content_block_stop
    data = json.dumps({'type': 'content_block_stop', 'index': 1})
    events.append(f"event: content_block_stop\ndata: {data}\n\n")

    # 7. content_block_start (text)
    data = json.dumps({
        'type': 'content_block_start',
        'index': 2,
        'content_block': {'type': 'text', 'text': ''}
    })
    events.append(f"event: content_block_start\ndata: {data}\n\n")

    # 8. content_block_delta (text_delta) - summary
    summary = generate_search_summary(query, search_results)
    for i in range(0, len(summary), 100):
        chunk = summary[i:i+100]
        data = json.dumps({
            'type': 'content_block_delta',
            'index': 2,
            'delta': {'type': 'text_delta', 'text': chunk}
        })
        events.append(f"event: content_block_delta\ndata: {data}\n\n")

    # 9. content_block_stop
    data = json.dumps({'type': 'content_block_stop', 'index': 2})
    events.append(f"event: content_block_stop\ndata: {data}\n\n")

    # 10. message_delta
    output_tokens = (len(summary) + 3) // 4
    data = json.dumps({
        'type': 'message_delta',
        'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
        'usage': {'output_tokens': output_tokens}
    })
    events.append(f"event: message_delta\ndata: {data}\n\n")

    # 11. message_stop
    data = json.dumps({'type': 'message_stop'})
    events.append(f"event: message_stop\ndata: {data}\n\n")

    return events


def generate_search_summary(query: str, results: Optional[Dict[str, Any]]) -> str:
    """Generate search results summary."""
    summary = f'Here are the search results for "{query}":\n\n'

    if results and "results" in results:
        for i, result in enumerate(results["results"], 1):
            summary += f"{i}. **{result.get('title', 'Untitled')}**\n"
            snippet = result.get("snippet", "")
            if snippet:
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                summary += f"   {snippet}\n"
            summary += f"   Source: {result.get('url', '')}\n\n"
    else:
        summary += "No results found.\n"

    summary += "\nPlease note that these are web search results and may not be fully accurate or up-to-date."
    return summary
