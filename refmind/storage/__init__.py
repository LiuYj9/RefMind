"""存储子包：SQLite 持久化（用户组、文档、会话、消息）。

对外统一从本包导入，例如::

    from refmind.storage import init_db, create_group, list_documents
"""

from .connection import connect, init_db
from .models import DocumentRow, Group, Message, Session
from .repository import (
    add_message,
    clear_messages,
    create_document,
    create_group,
    create_session,
    delete_document,
    delete_group,
    delete_session,
    get_document,
    get_group,
    get_session,
    list_documents,
    list_groups,
    list_messages,
    list_sessions,
    rename_session,
    update_document,
)

__all__ = [
    # 连接与初始化
    "connect",
    "init_db",
    # 数据模型
    "Group",
    "DocumentRow",
    "Session",
    "Message",
    # 用户组
    "create_group",
    "get_group",
    "list_groups",
    "delete_group",
    # 文档
    "create_document",
    "update_document",
    "get_document",
    "list_documents",
    "delete_document",
    # 会话
    "create_session",
    "get_session",
    "list_sessions",
    "rename_session",
    "delete_session",
    # 消息
    "add_message",
    "list_messages",
    "clear_messages",
]
