from random import randint
from typing import Generator

from dify_plugin.entities.model.llm import LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import AssistantPromptMessage, PromptMessageRole


def mock_llm_stream(t: str) -> Generator[LLMResultChunk, None, None]:
    # 模拟LLM token by token 输出
    i = 0
    c = 0
    while i < len(t):
        sub_length = min(randint(i, len(t)), 5)
        delta = t[i: i + sub_length]
        yield LLMResultChunk(
            model="",
            delta=LLMResultChunkDelta(
                index=c,
                message=AssistantPromptMessage(
                    role=PromptMessageRole.ASSISTANT,
                    content=delta
                ))
        )
        i += sub_length
        c += 1
