"""RefMind —— 基于大语言模型的智能文献知识库助手。

子包划分：
    config   配置与全局设置
    storage  SQLite 持久化（组 / 文档 / 会话 / 消息）
    parsing  PDF 解析（MinerU + PyMuPDF 回退）
    llm      模型工厂、翻译、摘要
    rag      文档处理、混合检索、长对话记忆、LangGraph 对话流水线
    services 跨模块的高层业务编排
"""

__version__ = "0.1.0"
