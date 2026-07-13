"""RefMind 前端。启动：.venv/Scripts/python.exe -m streamlit run app.py。"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from refmind import storage
from refmind.config import settings
from refmind.integrations import MCPManager, mcp_sdk_available
from refmind.llm import get_llm_status, stream_translate
from refmind.plugins import get_plugin_manager
from refmind.rag import (
    RelevantMemory,
    answer_question,
    dashscope_sdk_available,
    get_retriever,
)
from refmind.services import (
    ingest_pdf,
    recover_incomplete_ingestions,
    remove_document,
    remove_group,
)
from refmind.ui import inject_global_css

st.set_page_config(
    page_title="RefMind 文献知识库助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _initialize_runtime() -> dict[str, list[int]]:
    """每个进程只执行一次建库/恢复，避免不同 Streamlit 会话互相清理。"""
    storage.init_db()
    return recover_incomplete_ingestions()


_startup_recovery = _initialize_runtime()


# 会话状态
def _ss_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


_ss_default("current_group_id", None)
_ss_default("current_session_id", None)
_ss_default("uploader_nonce", 0)
_ss_default("lib_input_nonce", 0)
_ss_default("theme", "light")
_ss_default("pending_q", None)
_ss_default("confirm_delete", None)
_ss_default("is_parsing", False)
_ss_default("parse_queue", None)
_ss_default("parse_group_id", None)


def current_group() -> storage.Group | None:
    gid = st.session_state.current_group_id
    return storage.get_group(gid) if gid else None


def get_memory(session_id: int) -> RelevantMemory:
    key = f"memory_{session_id}"
    if key not in st.session_state:
        st.session_state[key] = RelevantMemory(session_id)
    return st.session_state[key]


def _open_in_system(path: str | None) -> tuple[bool, str]:
    """用系统默认程序打开 PDF，失败则打开所在文件夹（仅本地运行有效）。"""
    if not path:
        return False, "missing"
    p = Path(path)
    if p.exists() and hasattr(os, "startfile"):
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True, "file"
        except Exception:  # noqa: BLE001
            pass
    try:
        folder = p.parent
        if folder.exists() and hasattr(os, "startfile"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
            return True, "folder"
    except Exception:  # noqa: BLE001
        pass
    return False, "fail"


# 侧边栏
@st.dialog("删除文献库")
def _confirm_delete_dialog(group: storage.Group) -> None:
    st.write(f"确认删除文献库「{group.name}」吗？")
    st.caption("将同时删除其下全部文档与向量库，操作不可恢复。")
    c_cancel, c_ok = st.columns(2)
    if c_cancel.button("取消", use_container_width=True):
        st.session_state.confirm_delete = None
        st.rerun()
    if c_ok.button("确认删除", type="primary", use_container_width=True):
        try:
            remove_group(group.id)
        except Exception as exc:  # noqa: BLE001
            st.error(f"删除未完成，记录已保留以便重试：{exc}")
            return
        if st.session_state.current_group_id == group.id:
            st.session_state.current_group_id = None
            st.session_state.current_session_id = None
            st.session_state.pending_q = None
        st.session_state.confirm_delete = None
        st.rerun()


def render_sidebar() -> None:
    st.sidebar.markdown("## 📚 RefMind")
    st.sidebar.caption("文献知识库助手")
    if _startup_recovery["failed"]:
        st.sidebar.warning(
            "部分未完成文档仍需清理："
            + ", ".join(str(item) for item in _startup_recovery["failed"])
        )
    elif _startup_recovery["recovered"]:
        st.sidebar.caption(
            f"已恢复清理 {len(_startup_recovery['recovered'])} 条未完成记录。"
        )

    # 模型状态指示器
    status = get_llm_status()
    if status["fallback_model"]:
        state = status["circuit_state"]
        if state == "closed":
            st.sidebar.success(f"🟢 主模型：{status['primary_model']}")
        elif state == "open":
            st.sidebar.warning(
                f"🔴 主模型熔断（{status['failure_count']}/{status['failure_threshold']}）"
            )
            st.sidebar.caption(f"🟡 当前使用备选：{status['fallback_model']}")
        elif state == "half_open":
            st.sidebar.info(f"🟠 正在探测主模型恢复...")
    else:
        state = status["circuit_state"]
        if state == "closed":
            st.sidebar.success(f"🟢 主模型：{status['primary_model']}")
        elif state == "open":
            st.sidebar.warning(f"🔴 {status['primary_model']} 不可用，请检查 API Key 或网络")

    if not settings.has_api_key:
        st.sidebar.warning("未配置 API Key，请在「⚙️ 设置」中填写后使用。")
    if settings.rerank_enabled and not dashscope_sdk_available():
        st.sidebar.warning(
            "未安装 DashScope reranker SDK；当前使用嵌入重排。\n\n"
            "请运行：`.venv\\Scripts\\python.exe -m pip install dashscope`"
        )

    st.sidebar.divider()
    st.sidebar.markdown("### 文献库")

    # 创建文献库：输入名称后回车即可创建（无需按钮）
    new_name = st.sidebar.text_input(
        "创建文献库",
        key=f"new_lib_{st.session_state.lib_input_nonce}",
        placeholder="在此输入名称后回车创建文献库···",
        label_visibility="collapsed",
    )
    if new_name and new_name.strip():
        try:
            group = storage.create_group(new_name.strip())
            st.session_state.current_group_id = group.id
            st.session_state.current_session_id = None
            st.session_state.pending_q = None
        except Exception:  # noqa: BLE001
            st.sidebar.error("创建失败：该名称可能已存在")
        st.session_state.lib_input_nonce += 1
        st.rerun()

    # 文献库列表：单独的框，逐项展示（非下拉框）
    groups = storage.list_groups()
    if groups:
        box = st.sidebar.container(border=True)
        with box:
            for g in groups:
                is_current = g.id == st.session_state.current_group_id
                if is_current:
                    c_name, c_trash = st.columns([5, 1])
                    c_name.button(
                        f"📁 {g.name}",
                        key=f"sel_{g.id}",
                        type="primary",
                        use_container_width=True,
                    )
                    if c_trash.button("🗑", key=f"trash_{g.id}", help="删除该文献库"):
                        st.session_state.confirm_delete = g.id
                        st.rerun()
                else:
                    if st.button(
                        f"📁 {g.name}", key=f"sel_{g.id}", use_container_width=True
                    ):
                        st.session_state.current_group_id = g.id
                        st.session_state.current_session_id = None
                        st.session_state.pending_q = None
                        st.rerun()
    else:
        st.sidebar.caption("还没有文献库，请在上方输入名称创建。")

    group = current_group()
    if group:
        st.sidebar.divider()
        render_upload(group)


def render_upload(group: storage.Group) -> None:
    st.sidebar.markdown("### 上传文献 (PDF)")
    parsing_here = (
        st.session_state.is_parsing and st.session_state.parse_group_id == group.id
    )
    parsing_elsewhere = (
        st.session_state.is_parsing and st.session_state.parse_group_id != group.id
    )

    uploaded = st.sidebar.file_uploader(
        "拖拽或选择 PDF（支持多选）",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{group.id}_{st.session_state.uploader_nonce}",
        label_visibility="collapsed",
        disabled=parsing_here or parsing_elsewhere,
    )

    if parsing_here:
        st.sidebar.button(
            "⏳ 正在解析并入库 ...", disabled=True, use_container_width=True
        )
        _run_parse_job(group)
        return

    if parsing_elsewhere:
        st.sidebar.button(
            "🚀 开始解析并入库", disabled=True, use_container_width=True
        )
        st.sidebar.caption("其他文献库正在解析，请稍候 ...")
        return

    if uploaded and st.sidebar.button("🚀 开始解析并入库", use_container_width=True):
        st.session_state.parse_queue = [(f.name, f.getvalue()) for f in uploaded]
        st.session_state.is_parsing = True
        st.session_state.parse_group_id = group.id
        st.rerun()


def _run_parse_job(group: storage.Group) -> None:
    """在禁用上传控件后的下一轮渲染里执行解析入库。"""
    queue: list[tuple[str, bytes]] = st.session_state.parse_queue or []
    if not queue:
        st.session_state.is_parsing = False
        st.session_state.parse_group_id = None
        return

    existing = {d.filename: d for d in storage.list_documents(group.id)}
    total = len(queue)
    status = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0.0, text="准备中 ...")
    results: list[tuple[str, bool, str]] = []

    try:
        for idx, (filename, data) in enumerate(queue):
            previous = existing.get(filename)
            if previous is not None:
                status.warning(
                    f"♻️ 「{filename}」已存在，正在安全构建新版本 ..."
                )
            else:
                status.info(f"📄 正在处理：{filename}")

            # 每个上传任务使用独立临时文件，避免不同浏览器会话的同名 PDF 相互覆盖。
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(data)
                tmp = Path(handle.name)

            def _cb(p: float, msg: str, _idx=idx, _n=total, _name=filename):
                progress_bar.progress(
                    (_idx + p) / _n, text=f"[{_idx + 1}/{_n}] {_name} · {msg}"
                )

            try:
                new_document = ingest_pdf(
                    group.id, tmp, filename, progress=_cb
                )
                # 新版本 ready 后才删除旧记录；新入库失败时旧文档保持可用。
                cleanup_warning = ""
                if previous is not None:
                    try:
                        remove_document(previous.id)
                    except Exception as exc:  # noqa: BLE001
                        cleanup_warning = f"旧版本待重试清理：{exc}"
                existing[filename] = new_document
                results.append((filename, True, cleanup_warning))
            except Exception as exc:  # noqa: BLE001
                results.append((filename, False, str(exc)[:200]))
            finally:
                tmp.unlink(missing_ok=True)
    finally:
        progress_bar.empty()
        status.empty()
        st.session_state.is_parsing = False
        st.session_state.parse_queue = None
        st.session_state.parse_group_id = None

    for name, ok, err in results:
        if ok:
            st.sidebar.success(f"✅ {name} 已入库")
            if err:
                st.sidebar.warning(f"⚠️ {name}：{err}")
        else:
            st.sidebar.error(f"❌ {name}：{err}")

    if any(ok for _, ok, _ in results):
        sid = st.session_state.current_session_id
        if not sid or storage.get_session(sid) is None:
            sid = storage.create_session(group.id).id
            st.session_state.current_session_id = sid
        storage.add_message(sid, "assistant", "当前文档解析已完成，你有什么问题呢？")
        st.session_state.uploader_nonce += 1
        st.rerun()


# 对话
def render_chat(group: storage.Group) -> None:
    sessions = storage.list_sessions(group.id)

    col1, col2, col3 = st.columns([4, 1.2, 1.2])
    with col1:
        if sessions:
            labels = {f"{s.name} (#{s.id})": s.id for s in sessions}
            keys = list(labels.keys())
            cur = st.session_state.current_session_id
            index = 0
            if cur in labels.values():
                index = list(labels.values()).index(cur)
            chosen = st.selectbox("对话会话", keys, index=index, key="session_select")
            st.session_state.current_session_id = labels[chosen]
        else:
            st.caption("点击右侧「新建对话」开始。")
    with col2:
        st.write("")
        if st.button("➕ 新建对话", use_container_width=True):
            session = storage.create_session(group.id)
            st.session_state.current_session_id = session.id
            st.session_state.pending_q = None
            st.rerun()
    with col3:
        st.write("")
        if sessions and st.button("🗑️ 删除会话", use_container_width=True):
            if st.session_state.current_session_id:
                storage.delete_session(st.session_state.current_session_id)
                st.session_state.current_session_id = None
                st.rerun()

    session_id = st.session_state.current_session_id
    if not session_id:
        return

    messages = storage.list_messages(session_id)
    pending = st.session_state.pending_q
    pending_text = (
        pending["text"] if pending and pending.get("sid") == session_id else None
    )

    if not messages and not pending_text:
        doc_n = len(storage.list_documents(group.id))
        st.markdown(
            '<div class="refmind-hero">'
            '<div class="logo">📚</div>'
            "<h2>RefMind 文献助手</h2>"
            f"<p>已就绪，当前文献库共 {doc_n} 篇文献。在下方输入问题，我会基于这些文献作答。</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in messages:
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # 先渲染这一轮问答，输入框留到最后，保证它始终在回答下方
    if pending_text:
        with st.chat_message("user"):
            st.markdown(pending_text)
        with st.chat_message("assistant"):
            with st.spinner("检索并生成回答中 ..."):
                memory = get_memory(session_id)
                try:
                    result = answer_question(pending_text, group.id, memory=memory)
                except Exception as exc:  # noqa: BLE001
                    st.error("问答流程发生未收敛异常，请检查项目运行环境。")
                    st.code(f"{type(exc).__name__}: {exc}")
                    st.caption(
                        "请使用 `.venv\\Scripts\\python.exe -m streamlit run app.py` "
                        "启动，并执行 `.venv\\Scripts\\python.exe -m pip check`。"
                    )
                    st.session_state.pending_q = None
                    return
                if result.get("service_failed"):
                    st.error(result["answer"])
                    st.caption(
                        "建议使用项目虚拟环境重启："
                        "`.venv\\Scripts\\python.exe -m streamlit run app.py`"
                    )
                else:
                    st.markdown(result["answer"])
                queries = result.get("queries") or []
                if result.get("used_multi_agent") and len(queries) > 1:
                    st.caption("已并行检索：" + " · ".join(queries))
                if result.get("degraded"):
                    # 增强环节失败不会影响基线答案，但把状态展示出来便于排障。
                    with st.expander("⚠️ 本轮部分增强已安全降级"):
                        for warning in result.get("warnings") or ():
                            st.caption(warning)
                docs = result.get("documents") or []
                if docs:
                    with st.expander(f"📎 参考来源（{len(docs)} 个片段）"):
                        for i, d in enumerate(docs, start=1):
                            meta = d.metadata or {}
                            head = (
                                f"**[{i}] {meta.get('filename', '未知')} · 第 "
                                f"{meta.get('page', '?')} 页**"
                            )
                            if meta.get("section"):
                                head += f" · {meta['section']}"
                            if meta.get("rerank_score") is not None:
                                head += f" · 相关度 {meta['rerank_score']}"
                            st.markdown(head)
                            st.caption(d.page_content[:300] + " ...")
        st.session_state.pending_q = None

    prompt = st.chat_input("给 RefMind 发消息 ...")
    if prompt:
        st.session_state.pending_q = {"sid": session_id, "text": prompt}
        st.rerun()


# 文档库
def render_documents(group: storage.Group) -> None:
    docs = storage.list_documents(group.id)
    if not docs:
        st.info("当前文献库还没有文档，请在左侧上传 PDF。")
        return

    st.caption(f"共 {len(docs)} 篇文献 · 点击标题可用系统默认阅读器打开")
    for doc in docs:
        if st.button(f"📄 {doc.filename}", key=f"open_{doc.id}", use_container_width=True):
            ok, how = _open_in_system(doc.original_path)
            if ok and how == "file":
                st.toast("已在系统默认 PDF 阅读器中打开", icon="📖")
            elif ok and how == "folder":
                st.toast("无法直接打开 PDF，已打开其所在文件夹", icon="📂")
            else:
                st.warning(f"无法打开，文件路径：{doc.original_path}")

        status_cls = "ready" if doc.status == "ready" else "warn"
        st.markdown(
            f'<div class="refmind-card">'
            f'<span class="refmind-pill {status_cls}">{doc.status}</span>'
            f'<span class="refmind-pill">{doc.num_chunks} 块</span>'
            f'<div class="refmind-summary">{(doc.summary or "（暂无摘要）")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("删除该文档", key=f"del_doc_{doc.id}"):
            try:
                remove_document(doc.id)
            except Exception as exc:  # noqa: BLE001
                st.error(f"删除未完成，记录已保留以便重试：{exc}")
            else:
                st.rerun()


# 翻译
def render_translation(group: storage.Group) -> None:
    st.markdown("#### 文本翻译")
    src_text = st.text_area("输入待翻译文本", height=220, placeholder="粘贴任意文本 ...")
    c1, c2 = st.columns([1, 2])
    with c1:
        target = st.radio("目标语言", ["中文", "English"], horizontal=True)
    with c2:
        use_ctx = st.checkbox(
            "结合当前文献库提升术语准确性", value=True,
            help="翻译前从当前文献库已上传文献中检索相关片段，作为术语 / 风格参考。",
        )

    if st.button("翻译", type="primary", disabled=not settings.has_api_key):
        if not src_text.strip():
            st.info("请输入待翻译文本。")
            return
        context = ""
        if use_ctx:
            try:
                retriever = get_retriever(group.id)
                if retriever is not None:
                    hits = retriever.invoke(src_text[:600])
                    context = "\n\n".join(d.page_content for d in hits[:4])
            except Exception:  # noqa: BLE001
                context = ""
        if context:
            st.caption("已结合当前文献库片段进行术语对齐。")
        with st.chat_message("assistant"):
            try:
                st.write_stream(stream_translate(src_text, target, context))
            except Exception as exc:  # noqa: BLE001
                st.error(f"翻译失败：{exc}")


# 设置
def render_settings() -> None:
    st.markdown("#### ⚙️ 模型与参数设置")
    st.caption("保存后立即生效并写入 .env 持久化。")

    plugin_manager = get_plugin_manager()
    mcp_manager = MCPManager.from_environment()
    st.caption(
        f"扩展状态：{len(plugin_manager.plugins)} 个插件 · "
        f"{len(mcp_manager.servers)} 个 MCP 服务 · "
        f"MCP SDK {'已安装' if mcp_sdk_available() else '未安装（可选）'}"
    )
    extension_errors = [failure.message for failure in plugin_manager.failures]
    extension_errors.extend(mcp_manager.configuration_errors)
    if extension_errors:
        with st.expander("扩展加载诊断"):
            for error in extension_errors:
                st.caption(error)
    if st.button(
        "🔌 探测 MCP 能力",
        disabled=not bool(mcp_manager.servers),
        help="只读取服务的 tools/resources/prompts，不会把外部内容加入答案。",
    ):
        with st.spinner("正在连接已配置的 MCP 服务 ..."):
            probes = asyncio.run(mcp_manager.probe_all())
        st.json([asdict(probe) for probe in probes])

    backends = [
        "pipeline",
        "vlm-auto-engine",
        "hybrid-auto-engine",
        "vlm-http-client",
        "hybrid-http-client",
    ]
    methods = ["auto", "txt", "ocr"]
    sources = ["", "huggingface", "modelscope", "local"]

    def _idx(options, value, default=0):
        return options.index(value) if value in options else default

    with st.form("settings_form"):
        st.markdown("**对话 / 嵌入 API**")
        api_key = st.text_input(
            "API Key (DASHSCOPE_API_KEY)", value=settings.dashscope_api_key,
            type="password",
        )
        api_base = st.text_input("API_BASE", value=settings.api_base)
        col_a, col_b = st.columns(2)
        with col_a:
            llm_model = st.text_input("对话模型 LLM_MODEL", value=settings.llm_model)
        with col_b:
            embedding_model = st.text_input(
                "嵌入模型 EMBEDDING_MODEL", value=settings.embedding_model
            )
        multimodal_model = st.text_input(
            "图片摘要 / 携图回答模型 MULTIMODAL_LLM_MODEL",
            value=settings.multimodal_llm_model,
            help="仅在入库图片摘要和检索命中图片后的回答中调用。",
        )
        col_c, col_d = st.columns(2)
        with col_c:
            temperature = st.number_input(
                "对话温度 LLM_TEMPERATURE", 0.0, 2.0,
                value=float(settings.llm_temperature), step=0.1,
            )
        with col_d:
            embedding_batch = st.number_input(
                "嵌入批大小 (DashScope ≤ 10)", 1, 10,
                value=int(settings.embedding_batch_size),
            )

        st.markdown("**检索与记忆（影响论文阅读与记忆能力）**")
        col_e, col_f = st.columns(2)
        with col_e:
            top_k = st.number_input(
                "检索 Top-K", 1, 20, value=int(settings.retrieval_top_k)
            )
            chunk_size = st.number_input(
                "分块大小 CHUNK_SIZE", 200, 4000,
                value=int(settings.chunk_size), step=100,
            )
            mem_turns = st.number_input(
                "记忆轮数 MEMORY_MAX_TURNS", 1, 200, value=int(settings.memory_max_turns)
            )
        with col_f:
            chunk_overlap = st.number_input(
                "分块重叠 CHUNK_OVERLAP", 0, 1000,
                value=int(settings.chunk_overlap), step=50,
            )
            mem_thr = st.number_input(
                "记忆相关性阈值", 0.0, 1.0,
                value=float(settings.memory_relevance_threshold), step=0.05,
            )

        st.markdown("**召回 · 重排 · 上下文压缩**")
        col_j, col_k = st.columns(2)
        with col_j:
            recall_top_k = st.number_input(
                "召回候选数 RECALL_TOP_K", 5, 100,
                value=int(settings.recall_top_k),
            )
            rerank_enabled = st.checkbox(
                "启用重排 (rerank)", value=bool(settings.rerank_enabled),
                help="混合召回后用重排模型精排；未装 dashscope 时回退嵌入相似度。",
            )
            rerank_model = st.text_input(
                "重排模型 RERANK_MODEL", value=settings.rerank_model
            )
            rerank_top_n = st.number_input(
                "重排保留数 RERANK_TOP_N", 1, 30, value=int(settings.rerank_top_n)
            )
        with col_k:
            compression_enabled = st.checkbox(
                "启用上下文压缩", value=bool(settings.context_compression_enabled),
                help="去重复分块 + 句级过滤 + 字数预算，降低冗余与 token 消耗。",
            )
            context_max_chars = st.number_input(
                "上下文字数上限 CONTEXT_MAX_CHARS", 500, 20000,
                value=int(settings.context_max_chars), step=500,
            )
            redundancy_threshold = st.number_input(
                "去重相似度阈值", 0.5, 1.0,
                value=float(settings.redundancy_threshold), step=0.01,
            )
            sentence_threshold = st.number_input(
                "句级相关性阈值", 0.0, 1.0,
                value=float(settings.sentence_relevance_threshold), step=0.05,
            )

        st.markdown("**Multi-agent 研究编排**")
        col_ma, col_review = st.columns(2)
        with col_ma:
            multi_agent_enabled = st.checkbox(
                "启用多智能体检索",
                value=bool(settings.multi_agent_enabled),
                help="复杂问题拆成少量子查询并行召回；任一环节失败会自动回退单查询。",
            )
            max_subqueries = st.number_input(
                "最大子查询数", 1, 3,
                value=int(settings.multi_agent_max_subqueries),
            )
            max_workers = st.number_input(
                "并行检索线程数", 1, 8,
                value=int(settings.multi_agent_max_workers),
            )
            retrieval_timeout = st.number_input(
                "并行检索等待上限（秒）", 1.0, 300.0,
                value=float(settings.multi_agent_retrieval_timeout),
            )
        with col_review:
            evidence_review = st.checkbox(
                "启用 LLM 证据审查",
                value=bool(settings.evidence_review_enabled),
                help="在重排之后再次过滤仅关键词重合的片段，会增加一次模型调用。",
            )
            answer_review = st.checkbox(
                "启用答案审校",
                value=bool(settings.answer_review_enabled),
                help="依据最终证据最小化修正无依据断言，失败时保留原答案。",
            )

        st.markdown("**PDF 解析 (MinerU)**")
        col_g, col_h, col_i = st.columns(3)
        with col_g:
            backend = st.selectbox("后端", backends, index=_idx(backends, settings.mineru_backend))
        with col_h:
            method = st.selectbox("方法", methods, index=_idx(methods, settings.mineru_method))
        with col_i:
            model_source = st.selectbox(
                "模型源", sources, index=_idx(sources, settings.mineru_model_source)
            )

        st.caption("⚠️ 修改嵌入模型后，建议重新上传文献以重建向量索引（避免维度不一致）。")
        submitted = st.form_submit_button("💾 保存设置", type="primary")

    if submitted:
        settings.apply_and_persist(
            {
                "DASHSCOPE_API_KEY": api_key,
                "API_BASE": api_base,
                "LLM_MODEL": llm_model,
                "MULTIMODAL_LLM_MODEL": multimodal_model,
                "EMBEDDING_MODEL": embedding_model,
                "LLM_TEMPERATURE": temperature,
                "EMBEDDING_BATCH_SIZE": embedding_batch,
                "RETRIEVAL_TOP_K": top_k,
                "CHUNK_SIZE": chunk_size,
                "CHUNK_OVERLAP": chunk_overlap,
                "MEMORY_MAX_TURNS": mem_turns,
                "MEMORY_RELEVANCE_THRESHOLD": mem_thr,
                "RECALL_TOP_K": recall_top_k,
                "RERANK_ENABLED": rerank_enabled,
                "RERANK_MODEL": rerank_model,
                "RERANK_TOP_N": rerank_top_n,
                "CONTEXT_COMPRESSION_ENABLED": compression_enabled,
                "CONTEXT_MAX_CHARS": context_max_chars,
                "REDUNDANCY_THRESHOLD": redundancy_threshold,
                "SENTENCE_RELEVANCE_THRESHOLD": sentence_threshold,
                "MULTI_AGENT_ENABLED": multi_agent_enabled,
                "MULTI_AGENT_MAX_SUBQUERIES": max_subqueries,
                "MULTI_AGENT_MAX_WORKERS": max_workers,
                "MULTI_AGENT_RETRIEVAL_TIMEOUT": retrieval_timeout,
                "MULTI_AGENT_EVIDENCE_REVIEW": evidence_review,
                "MULTI_AGENT_ANSWER_REVIEW": answer_review,
                "MINERU_BACKEND": backend,
                "MINERU_METHOD": method,
                "MINERU_MODEL_SOURCE": model_source,
            }
        )
        st.success("设置已保存并生效。")


def main() -> None:
    inject_global_css(st.session_state.theme)
    render_sidebar()

    cd = st.session_state.get("confirm_delete")
    if cd:
        target = storage.get_group(cd)
        if target:
            _confirm_delete_dialog(target)
        else:
            st.session_state.confirm_delete = None

    # 右上角主题开关
    st.markdown('<div class="theme-switch-row"></div>', unsafe_allow_html=True)
    _, t_col = st.columns([11, 1])
    with t_col:
        is_dark = st.toggle(
            "主题",
            value=st.session_state.theme == "dark",
            key="theme_switch",
            label_visibility="collapsed",
            help="切换白天 / 夜间主题",
        )
    if is_dark != (st.session_state.theme == "dark"):
        st.session_state.theme = "dark" if is_dark else "light"
        st.rerun()

    group = current_group()
    if group is None:
        st.title("📚 RefMind 文献知识库助手")
        st.markdown(
            "基于大语言模型的智能文献知识库：高精度 PDF 解析、混合检索 RAG 问答、"
            "文档翻译与摘要、按文献库隔离的知识库。"
        )
        st.info("👈 请在左侧创建或选择一个文献库开始使用。")
        return

    st.title(f"📚 {group.name}")
    st.caption(f"当前文献库共 {len(storage.list_documents(group.id))} 篇文献")

    tab_chat, tab_docs, tab_trans, tab_set = st.tabs(
        ["💬 对话", "📄 文档库", "🌐 翻译", "⚙️ 设置"]
    )
    with tab_chat:
        render_chat(group)
    with tab_docs:
        render_documents(group)
    with tab_trans:
        render_translation(group)
    with tab_set:
        render_settings()


if __name__ == "__main__":
    main()
