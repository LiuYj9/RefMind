"""探测 DashScope OpenAI 兼容接口下可用的嵌入/对话模型。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from refmind.config import settings

client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.api_base)

EMBED_CANDIDATES = [
    "text-embedding-v4",
    "text-embedding-v3",
    "text-embedding-v2",
    settings.embedding_model,
]
CHAT_CANDIDATES = [
    settings.llm_model,
    "qwen-plus",
    "qwen3-plus",
    "qwen-turbo",
]

print("== 嵌入模型 ==")
for m in EMBED_CANDIDATES:
    try:
        r = client.embeddings.create(model=m, input="测试文本 hello")
        print(f"  OK   {m}  (dim={len(r.data[0].embedding)})")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {m}: {str(exc)[:110]}")

print("\n== 对话模型 ==")
for m in CHAT_CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "用一句话回答：你好"}],
            max_tokens=20,
        )
        print(f"  OK   {m}: {r.choices[0].message.content[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {m}: {str(exc)[:110]}")
