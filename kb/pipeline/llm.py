"""LLM / Embedding 工厂：DeepSeek via LangChain OpenAI 兼容接口。"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

from langchain_openai import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddingsPrimary
except ImportError:  # 旧环境可能仅有 community 包
    HFEmbeddingsPrimary = None

from langchain_community.embeddings import (
    HuggingFaceEmbeddings as HFEmbeddingsCommunity,
)


def get_deepseek_chat(temperature: float = 0.0) -> Any:
    """构建指向 DeepSeek API 的 LangChain ChatModel。

    Args:
        temperature: 采样温度，评测/判定场景建议为 0。

    Returns:
        已配置 base_url 与 api_key 的 ChatOpenAI 实例。

    Raises:
        RuntimeError: 未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")

    return ChatOpenAI(
        # 评测稳定性：优先 chat；flash/推理模型输出更飘，可用环境变量覆盖
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        temperature=temperature,
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096")),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """中文向量模型；优先 BGE，失败则回退 MiniLM（Chroma 常用）。

    Returns:
        LangChain Embeddings 实例（进程内缓存单例）。
    """
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    kwargs = {
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
    }

    if HFEmbeddingsPrimary is not None:
        try:
            return HFEmbeddingsPrimary(model_name=model_name, **kwargs)
        except Exception:
            pass

    try:
        return HFEmbeddingsCommunity(model_name=model_name, **kwargs)
    except Exception:
        # 最后回退：轻量英文模型，保证链路可跑
        return HFEmbeddingsCommunity(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            **kwargs,
        )
