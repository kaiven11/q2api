from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class ClaudeMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = ""
    input_schema: Dict[str, Any] = {}
    type: Optional[str] = Field(default=None, alias="type")  # For websearch: "web_search_20250305"
    max_uses: Optional[int] = None  # For websearch tools

    def is_web_search(self) -> bool:
        """Check if this is a websearch tool."""
        return self.type is not None and self.type.startswith("web_search")

class ClaudeThinking(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thinking_type: str = Field(alias="type")
    budget_tokens: Optional[int] = Field(default=20000, alias="budget_tokens")

class ClaudeRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    max_tokens: int = 4096
    temperature: Optional[float] = None
    tools: Optional[List[ClaudeTool]] = None
    stream: bool = False
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    thinking: Optional[ClaudeThinking] = None
