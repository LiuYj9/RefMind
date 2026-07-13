"""全局配置：从 .env 读取密钥、模型名与路径，暴露为单例 settings。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    """读取布尔型环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_bool(value: object) -> bool:
    """把任意值归一化为布尔（供设置页写回时的类型转换用）。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """读取整型环境变量，解析失败时返回默认值。"""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    """读取浮点型环境变量，解析失败时返回默认值。"""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """应用配置。相对路径均相对于项目根目录。"""

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # 凭证 / 接口地址
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.api_base = os.getenv(
            "API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 模型
        self.llm_model = os.getenv("LLM_MODEL", "qwen3.7-plus")
        # 多模态嵌入模型不支持 OpenAI 兼容接口，默认使用文本嵌入模型
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.llm_temperature = _get_float("LLM_TEMPERATURE", 0.1)
        # 模型降级：主模型不可用时自动切换备选模型，恢复后继自动切回
        self.fallback_llm_model = os.getenv("LLM_FALLBACK_MODEL", "")
        self.fallback_api_base = os.getenv("LLM_FALLBACK_API_BASE", "")
        self.llm_health_check_interval = _get_int("LLM_HEALTH_CHECK_INTERVAL", 60)
        # 熔断器：连续失败 N 次后熔断（Open），冷却后可进入半开状态探测
        self.llm_circuit_failure_threshold = _get_int(
            "LLM_CIRCUIT_FAILURE_THRESHOLD", 3
        )

        # PDF 解析
        self.mineru_binary = os.getenv("MINERU_BINARY_PATH", "mineru")
        self.use_fallback_parser = _get_bool("USE_FALLBACK_PARSER", False)
        # MinerU 后端：pipeline 更通用、模型更小；其余为高精度但更重的方案
        self.mineru_backend = os.getenv("MINERU_BACKEND", "pipeline")
        # MinerU 解析方法：auto / txt / ocr
        self.mineru_method = os.getenv("MINERU_METHOD", "auto")
        # 模型下载源：huggingface / modelscope / local（留空则用 MinerU 默认）
        self.mineru_model_source = os.getenv("MINERU_MODEL_SOURCE", "")

        # 存储路径
        self.chroma_persist_dir = self._resolve(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_data")
        )
        self.database_path = self._resolve(
            os.getenv("DATABASE_PATH", "./data/refmind.db")
        )
        self.upload_dir = self._resolve(os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.parsed_dir = self._resolve(os.getenv("PARSED_DIR", "./data/parsed"))

        # 检索 / 记忆相关参数
        self.retrieval_top_k = _get_int("RETRIEVAL_TOP_K", 5)
        self.memory_max_turns = _get_int("MEMORY_MAX_TURNS", 30)
        self.memory_relevance_threshold = _get_float(
            "MEMORY_RELEVANCE_THRESHOLD", 0.3
        )
        self.chunk_size = _get_int("CHUNK_SIZE", 1000)
        self.chunk_overlap = _get_int("CHUNK_OVERLAP", 200)
        # DashScope 兼容接口单次嵌入请求最多 10 条文本
        self.embedding_batch_size = _get_int("EMBEDDING_BATCH_SIZE", 10)

        # 混合召回 -> 重排 -> 上下文压缩
        # 召回阶段先取较多候选，交给重排精排后再压缩进 Prompt
        self.recall_top_k = _get_int("RECALL_TOP_K", 20)
        self.rerank_enabled = _get_bool("RERANK_ENABLED", True)
        self.rerank_model = os.getenv("RERANK_MODEL", "gte-rerank-v2")
        self.rerank_top_n = _get_int("RERANK_TOP_N", 5)
        self.context_compression_enabled = _get_bool(
            "CONTEXT_COMPRESSION_ENABLED", True
        )
        # 送入 Prompt 的上下文字数上限
        self.context_max_chars = _get_int("CONTEXT_MAX_CHARS", 4000)
        # 去重阈值：块间余弦相似度高于此值视为重复，仅保留排名靠前者
        self.redundancy_threshold = _get_float("REDUNDANCY_THRESHOLD", 0.92)
        # 句级过滤阈值：句子与问题相似度低于此值则从上下文中剔除
        self.sentence_relevance_threshold = _get_float(
            "SENTENCE_RELEVANCE_THRESHOLD", 0.25
        )

        # 受控 multi-agent：规划只负责拆查询，并发只用于检索，答案审校仍受证据约束。
        self.multi_agent_enabled = _get_bool("MULTI_AGENT_ENABLED", True)
        self.multi_agent_max_subqueries = min(
            3, max(1, _get_int("MULTI_AGENT_MAX_SUBQUERIES", 3))
        )
        self.multi_agent_max_workers = min(
            8, max(1, _get_int("MULTI_AGENT_MAX_WORKERS", 3))
        )
        self.multi_agent_retrieval_timeout = min(
            300.0,
            max(1.0, _get_float("MULTI_AGENT_RETRIEVAL_TIMEOUT", 30.0)),
        )
        self.evidence_review_enabled = _get_bool(
            "MULTI_AGENT_EVIDENCE_REVIEW", False
        )
        self.answer_review_enabled = _get_bool("MULTI_AGENT_ANSWER_REVIEW", True)

    def _resolve(self, value: str) -> Path:
        """将相对路径解析为相对于项目根目录的绝对路径。"""
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def ensure_dirs(self) -> None:
        """创建所有存储目录（以及数据库所在目录）。"""
        for path in (
            self.chroma_persist_dir,
            self.upload_dir,
            self.parsed_dir,
            self.database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def group_chroma_dir(self, group_id: int) -> Path:
        """返回某个用户组独立的 Chroma 持久化目录。"""
        path = self.chroma_persist_dir / f"group_{group_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def has_api_key(self) -> bool:
        """是否已配置 API 密钥。"""
        return bool(self.dashscope_api_key)

    # 设置界面可编辑项：环境变量名 -> (属性名, 类型转换)
    _ENV_ATTR_MAP = {
        "DASHSCOPE_API_KEY": ("dashscope_api_key", str),
        "API_BASE": ("api_base", str),
        "LLM_MODEL": ("llm_model", str),
        "LLM_FALLBACK_MODEL": ("fallback_llm_model", str),
        "LLM_FALLBACK_API_BASE": ("fallback_api_base", str),
        "LLM_HEALTH_CHECK_INTERVAL": ("llm_health_check_interval", int),
        "LLM_CIRCUIT_FAILURE_THRESHOLD": ("llm_circuit_failure_threshold", int),
        "EMBEDDING_MODEL": ("embedding_model", str),
        "LLM_TEMPERATURE": ("llm_temperature", float),
        "CHUNK_SIZE": ("chunk_size", int),
        "CHUNK_OVERLAP": ("chunk_overlap", int),
        "RETRIEVAL_TOP_K": ("retrieval_top_k", int),
        "MEMORY_MAX_TURNS": ("memory_max_turns", int),
        "MEMORY_RELEVANCE_THRESHOLD": ("memory_relevance_threshold", float),
        "EMBEDDING_BATCH_SIZE": ("embedding_batch_size", int),
        "RECALL_TOP_K": ("recall_top_k", int),
        "RERANK_ENABLED": ("rerank_enabled", _as_bool),
        "RERANK_MODEL": ("rerank_model", str),
        "RERANK_TOP_N": ("rerank_top_n", int),
        "CONTEXT_COMPRESSION_ENABLED": ("context_compression_enabled", _as_bool),
        "CONTEXT_MAX_CHARS": ("context_max_chars", int),
        "REDUNDANCY_THRESHOLD": ("redundancy_threshold", float),
        "SENTENCE_RELEVANCE_THRESHOLD": ("sentence_relevance_threshold", float),
        "MULTI_AGENT_ENABLED": ("multi_agent_enabled", _as_bool),
        "MULTI_AGENT_MAX_SUBQUERIES": ("multi_agent_max_subqueries", int),
        "MULTI_AGENT_MAX_WORKERS": ("multi_agent_max_workers", int),
        "MULTI_AGENT_RETRIEVAL_TIMEOUT": (
            "multi_agent_retrieval_timeout",
            float,
        ),
        "MULTI_AGENT_EVIDENCE_REVIEW": ("evidence_review_enabled", _as_bool),
        "MULTI_AGENT_ANSWER_REVIEW": ("answer_review_enabled", _as_bool),
        "MINERU_BACKEND": ("mineru_backend", str),
        "MINERU_METHOD": ("mineru_method", str),
        "MINERU_MODEL_SOURCE": ("mineru_model_source", str),
    }

    def apply_and_persist(self, values: dict[str, object]) -> None:
        """更新内存配置并写回 .env，键为环境变量名（见 _ENV_ATTR_MAP）。"""
        from dotenv import set_key

        env_path = self.project_root / ".env"
        env_path.touch(exist_ok=True)

        for env_key, raw in values.items():
            if env_key not in self._ENV_ATTR_MAP:
                continue
            attr, caster = self._ENV_ATTR_MAP[env_key]
            try:
                casted = caster(raw)
            except (TypeError, ValueError):
                continue
            setattr(self, attr, casted)
            set_key(str(env_path), env_key, str(raw), quote_mode="never")

        # 清空缓存，确保新模型 / 检索器立即生效
        from ..llm.factory import reset_model_cache
        from ..rag.retrieval import reset_retrievers

        reset_model_cache()
        reset_retrievers()


# 模块级单例
settings = Settings()
