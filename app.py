"""RefMind —— Streamlit 前端。

启动方式::

    streamlit run app.py

（已在 .streamlit/config.toml 中将默认端口设为 8888）
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from refmind import storage
from refmind.config import settings
from refmind.llm import stream_translate
from refmind.rag import RelevantMemory, answer_question, get_retriever
from refmind.services import ingest_pdf, remove_document, remove_group
from refmind.ui import inject_global_css

st.set_page_config(
    page_title="RefMind 文献知识库助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

storage.init_db()


# --------------------------------------------------------------------------- #
# 会话状态
# --------------------------------------------------------------------------- #
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
    """用系统默认程序打开 PDF；失败则打开其所在文件夹。

    仅在本地运行（服务端=用户机器）时有效。
    """
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


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
@st.dialog("删除文献库")
def _confirm_delete_dialog(group: storage.Group) -> None:
    """删除文献库的二次确认弹窗。"""
    st.write(f"确认删除文献库「{group.name}」吗？")
    st.caption("将同时删除其下全部文档与向量库，操作不可恢复。")
    c_cancel, c_ok = st.columns(2)
    if c_cancel.button("取消", use_container_width=True):
        st.session_state.confirm_delete = None
        st.rerun()
    if c_ok.button("确认删除", type="primary", use_container_width=True):
        remove_group(group.id)
        if st.session_state.current_group_id == group.id:
            st.session_state.current_group_id = None
            st.session_state.current_session_id = None
            st.session_state.pending_q = None
        st.session_state.confirm_delete = None
        st.rerun()


def render_sidebar() -> None:
    st.sidebar.markdown("## 📚 RefMind")
    st.sidebar.caption("文献知识库助手")

    if not settings.has_api_key:
        st.sidebar.warning("未配置 API Key，请在「⚙️ 设置」中填写后使用。")

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
        except Exception:  # noqa: BLE001 - 名称重复等
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
    """执行解析任务（在禁用按钮/上传框后的下一轮渲染中运行）。"""
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
            if filename in existing:
                status.warning(
                    f"♻️ 「{filename}」已存在，正在删除旧文件并重新解析 ..."
                )
                remove_document(existing[filename].id)
            else:
                status.info(f"📄 正在处理：{filename}")

            tmp = Path(tempfile.gettempdir()) / filename
            tmp.write_bytes(data)

            def _cb(p: float, msg: str, _idx=idx, _n=total, _name=filename):
                progress_bar.progress(
                    (_idx + p) / _n, text=f"[{_idx + 1}/{_n}] {_name} · {msg}"
                )

            try:
                ingest_pdf(group.id, tmp, filename, progress=_cb)
                results.append((filename, True, ""))
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


# --------------------------------------------------------------------------- #
# 对话标签页
# --------------------------------------------------------------------------- #
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

    # 1) 渲染历史消息
    for msg in messages:
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # 2) 若有待回答的问题，先渲染问答（确保输入框位于回答下方）
    if pending_text:
        with st.chat_message("user"):
            st.markdown(pending_text)
        with st.chat_message("assistant"):
            with st.spinner("检索并生成回答中 ..."):
                memory = get_memory(session_id)
                try:
                    result = answer_question(pending_text, group.id, memory=memory)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"生成回答失败：{exc}")
                    st.session_state.pending_q = None
                    return
                st.markdown(result["answer"])
                docs = result.get("documents") or []
                if docs:
                    with st.expander(f"📎 参考来源（{len(docs)} 个片段）"):
                        for i, d in enumerate(docs, start=1):
                            meta = d.metadata or {}
                            st.markdown(
                                f"**[{i}] {meta.get('filename', '未知')} · 第 "
                                f"{meta.get('page', '?')} 页**"
                            )
                            st.caption(d.page_content[:300] + " ...")
        st.session_state.pending_q = None

    # 3) 输入框始终最后渲染 —— 永远位于回答下方
    prompt = st.chat_input("给 RefMind 发消息 ...")
    if prompt:
        st.session_state.pending_q = {"sid": session_id, "text": prompt}
        st.rerun()


# --------------------------------------------------------------------------- #
# 文档库标签页
# --------------------------------------------------------------------------- #
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
            remove_document(doc.id)
            st.rerun()


# --------------------------------------------------------------------------- #
# 翻译标签页（可结合历史输入文献对齐术语）
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 设置标签页
# --------------------------------------------------------------------------- #
def render_settings() -> None:
    st.markdown("#### ⚙️ 模型与参数设置")
    st.caption("保存后立即生效并写入 .env 持久化。")

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
                "EMBEDDING_MODEL": embedding_model,
                "LLM_TEMPERATURE": temperature,
                "EMBEDDING_BATCH_SIZE": embedding_batch,
                "RETRIEVAL_TOP_K": top_k,
                "CHUNK_SIZE": chunk_size,
                "CHUNK_OVERLAP": chunk_overlap,
                "MEMORY_MAX_TURNS": mem_turns,
                "MEMORY_RELEVANCE_THRESHOLD": mem_thr,
                "MINERU_BACKEND": backend,
                "MINERU_METHOD": method,
                "MINERU_MODEL_SOURCE": model_source,
            }
        )
        st.success("设置已保存并生效。")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_global_css(st.session_state.theme)
    render_sidebar()

    # 待删除确认弹窗
    cd = st.session_state.get("confirm_delete")
    if cd:
        target = storage.get_group(cd)
        if target:
            _confirm_delete_dialog(target)
        else:
            st.session_state.confirm_delete = None

    # 右上角主题滑块（无文字，左=白天 右=夜间，原生动画）
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
