"""
智能上下文管理和压缩模块
参考:
- https://platform.claude.com/cookbook/tool-use-automatic-context-compaction
- https://devblogs.microsoft.com/semantic-kernel/managing-chat-history-for-large-language-models-llms/
"""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class ContextManager:
    """管理对话历史的智能压缩"""

    def __init__(self,
                 max_messages: int = 20,
                 compression_threshold: int = 15,
                 keep_recent: int = 5):
        """
        Args:
            max_messages: 最大消息数量
            compression_threshold: 触发压缩的阈值
            keep_recent: 保留最近N条完整消息
        """
        self.max_messages = max_messages
        self.compression_threshold = compression_threshold
        self.keep_recent = keep_recent

    def compress_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        智能压缩消息历史
        策略: 滑动窗口 + 摘要压缩
        """
        if self.max_messages <= 0:
            # 无限制模式，使用智能压缩
            return self._smart_compress(messages)

        # 限制模式，简单截断
        if len(messages) <= self.max_messages:
            return messages

        return messages[-self.max_messages:]

    def _smart_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """智能压缩：保留最近消息 + 压缩旧消息"""
        if len(messages) <= self.compression_threshold:
            return messages

        # 保留最近的消息
        recent = messages[-self.keep_recent:]
        old = messages[:-self.keep_recent]

        # 压缩旧消息
        compressed = self._compress_old_messages(old)

        return compressed + recent

    def _compress_old_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """压缩旧消息：移除冗余内容"""
        compressed = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 跳过空消息
            if not content:
                continue

            # 压缩长文本
            if isinstance(content, str):
                compressed_content = self._compress_text(content)
                compressed.append({
                    "role": role,
                    "content": compressed_content
                })
            elif isinstance(content, list):
                # 处理多模态内容，只保留文本
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))

                if text_parts:
                    compressed_text = self._compress_text(" ".join(text_parts))
                    compressed.append({
                        "role": role,
                        "content": compressed_text
                    })

        # 进一步压缩：每3条合并为1条摘要
        if len(compressed) > 6:
            return self._merge_messages(compressed)

        return compressed

    def _compress_text(self, text: str, max_length: int = 200) -> str:
        """压缩单条文本"""
        if len(text) <= max_length:
            return text

        # 保留开头和结尾
        return text[:max_length//2] + "...[省略]..." + text[-max_length//2:]

    def _merge_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并多条消息为摘要"""
        merged = []
        batch_size = 3

        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]

            # 合并为一条摘要
            contents = []
            for msg in batch:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                contents.append(f"[{role}]: {content[:100]}")

            merged.append({
                "role": "user",
                "content": "[历史摘要] " + " | ".join(contents)
            })

        return merged


class TokenBasedContextManager(ContextManager):
    """基于Token数量的上下文管理"""

    def __init__(self, max_tokens: int = 100000):
        super().__init__()
        self.max_tokens = max_tokens

    def compress_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于Token数量压缩"""
        total_tokens = self._estimate_tokens(messages)

        if total_tokens <= self.max_tokens:
            return messages

        # 从旧到新逐步移除消息
        compressed = messages.copy()
        while self._estimate_tokens(compressed) > self.max_tokens and len(compressed) > 5:
            # 移除最旧的消息（保留至少5条）
            compressed.pop(0)

        return compressed

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算Token数量（粗略估计：1 token ≈ 4 字符）"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        total += len(text) // 4
        return total
