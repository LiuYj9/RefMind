"""存储子包：SQLite 持久化（用户组、文档、会话、消息）。"""

from .connection import connect, init_db
from .models import DocumentRow, Group, LongTermMemory, Message, Session
from .repository import (
    add_message,
    clear_messages,
    create_long_term_memory,
    create_document,
    create_group,
    create_session,
    delete_document,
    delete_group,
    delete_session,
    deactivate_long_term_memory,
    get_document,
    get_group,
    get_session,
    get_long_term_memory,
    list_documents,
    list_groups,
    list_messages,
    list_sessions,
    list_long_term_memories,
    rename_session,
    touch_long_term_memories,
    update_long_term_memory,
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
    "LongTermMemory",
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
    # 用户长期记忆
    "create_long_term_memory",
    "get_long_term_memory",
    "list_long_term_memories",
    "update_long_term_memory",
    "touch_long_term_memories",
    "deactivate_long_term_memory",
]
