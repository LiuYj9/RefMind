"""示例：给检索片段添加可观测标签，不改变正文或排序。"""

from __future__ import annotations

from langchain_core.documents import Document

from refmind.plugins import CoreHook, HookEvent, PluginRegistrar

PLUGIN_NAME = "metadata-tag-example"


def _tag_retrieved_documents(event: HookEvent[list[Document]]) -> list[Document]:
    tagged: list[Document] = []
    for document in event.value:
        metadata = dict(document.metadata or {})
        metadata["processed_by_plugin"] = PLUGIN_NAME
        tagged.append(Document(page_content=document.page_content, metadata=metadata))
    return tagged


def register(registrar: PluginRegistrar) -> None:
    registrar.add_hook(CoreHook.AFTER_RETRIEVE, _tag_retrieved_documents)

