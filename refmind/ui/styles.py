"""注入全局 CSS，支持白天/夜间主题。"""

from __future__ import annotations

import streamlit as st

# 两套主题的配色变量
_THEMES = {
    "light": {
        "bg": "#ffffff",
        "bg2": "#f1f5f9",
        "text": "#0f172a",
        "muted": "#64748b",
        "border": "#e2e8f0",
        "assistant": "#f1f5f9",
        "assistant_text": "#0f172a",
        "card": "#ffffff",
        "input_bg": "#ffffff",
    },
    "dark": {
        "bg": "#0f172a",
        "bg2": "#1e293b",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "border": "#334155",
        "assistant": "#1e293b",
        "assistant_text": "#e2e8f0",
        "card": "#1e293b",
        "input_bg": "#1e293b",
    },
}


def _css(theme: str) -> str:
    c = _THEMES.get(theme, _THEMES["light"])
    return f"""
<style>
:root {{
    --bg: {c['bg']};
    --bg2: {c['bg2']};
    --text: {c['text']};
    --muted: {c['muted']};
    --border: {c['border']};
    --assistant: {c['assistant']};
    --assistant-text: {c['assistant_text']};
    --card: {c['card']};
    --input-bg: {c['input_bg']};
    --primary: #2563eb;
}}

html, body, [class*="css"] {{
    font-family: "Inter", "PingFang SC", "Microsoft YaHei", -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

/* ===== 主区与侧边栏共用相同背景色 ===== */
[data-testid="stAppViewContainer"], .stApp {{ background: var(--bg); }}
[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{
    background: var(--bg);
    border-right: 1px solid var(--border);
}}
.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
section[data-testid="stSidebar"] * {{ color: var(--text); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: var(--text); }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}

.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 7rem;
    max-width: 900px;
}}
h1 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 1.6rem; }}

/* ===== 输入控件 ===== */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: var(--input-bg) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    border-radius: 10px;
}}
[data-testid="stFileUploaderDropzone"] {{
    background-color: var(--bg2);
    border: 1px dashed var(--border);
    border-radius: 10px;
}}

/* ===== 按钮 ===== */
.stButton > button, .stForm button, .stDownloadButton > button {{
    border-radius: 10px;
    font-weight: 600;
    background: var(--bg2);
    color: var(--text);
    border: 1px solid var(--border);
    transition: all .15s ease;
}}
.stButton > button:hover {{ transform: translateY(-1px); border-color: var(--primary); }}
.stButton > button[kind="primary"], .stForm button[kind="primaryFormSubmit"] {{
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
}}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
    height: 42px; padding: 0 16px; border-radius: 10px 10px 0 0;
    font-weight: 600; color: var(--muted);
}}
.stTabs [aria-selected="true"] {{ color: var(--primary); }}

/* ===== 聊天气泡 ===== */
[data-testid="stChatMessage"] {{
    background: transparent;
    padding: 2px 0;
    gap: 12px;
    min-width: 0;
    max-width: 100%;
}}
[data-testid="stChatMessage"] > div:last-child,
[data-testid="stChatMessageContent"] {{
    border-radius: 16px;
    padding: 4px 16px;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
}}
/*
 * 论文文件名常含连续下划线，浏览器会把它视为一个不可分割的长单词。
 * 在 Markdown 文本边界强制提供断行点，避免正文逃出聊天气泡。
 */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: break-word;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]
    :is(p, li, a, em, strong, blockquote, td, th) {{
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: break-word;
}}
/* 代码块和宽表格保留横向滚动，不能靠裁剪或逐字符换行破坏可读性。 */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {{
    display: block;
    max-width: 100%;
    box-sizing: border-box;
    overflow-x: auto;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {{
    white-space: pre;
    overflow-wrap: normal;
    word-break: normal;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] img {{
    max-width: 100%;
    height: auto;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    > div:last-child,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {{
    background: var(--assistant);
    border: 1px solid var(--border);
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:last-child * {{
    color: var(--assistant-text) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ flex-direction: row-reverse; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    > div:last-child,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {{
    background: var(--primary);
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:last-child * {{
    color: #ffffff !important;
}}

/* ===== 底部聊天输入框 ===== */
[data-testid="stChatInput"] {{
    border-radius: 14px; border: 1px solid var(--border);
    background: var(--input-bg);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.10);
}}
[data-testid="stChatInput"] textarea {{ color: var(--text) !important; }}

/* DeepSeek 风格的一体式输入区：原生 chat_input + 可取消选择的 GS pill。 */
.st-key-refmind_composer {{
    max-width: 900px;
    margin: 0 auto .35rem;
    padding: .18rem .45rem .38rem;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--input-bg);
    box-shadow: 0 7px 24px rgba(15, 23, 42, .12);
    transition: border-color .15s ease, box-shadow .15s ease;
}}
.st-key-refmind_composer:focus-within {{
    border-color: var(--primary);
    box-shadow: 0 8px 28px rgba(37, 99, 235, .16);
}}
.st-key-refmind_composer [data-testid="stChatInput"] {{
    border: 0 !important;
    border-radius: 14px !important;
    background: transparent !important;
    box-shadow: none !important;
}}
.st-key-refmind_composer [data-testid="stPills"] {{
    margin-left: .25rem;
}}
.st-key-refmind_composer [data-testid="stPills"] button {{
    min-height: 30px;
    padding: 2px 12px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--bg2);
    color: var(--muted);
    font-weight: 650;
}}
.st-key-refmind_composer [data-testid="stPills"] button[aria-pressed="true"],
.st-key-refmind_composer [data-testid="stPills"] button[aria-selected="true"] {{
    border-color: var(--primary);
    background: rgba(37, 99, 235, .13);
    color: var(--primary);
}}

/* ===== 空状态欢迎区 ===== */
.refmind-hero {{ text-align: center; padding: 56px 20px 28px; color: var(--muted); }}
.refmind-hero .logo {{ font-size: 46px; }}
.refmind-hero h2 {{ color: var(--text); margin: 10px 0 6px; }}
.refmind-hero p {{ color: var(--muted); margin: 0; }}

/* ===== 文档卡 ===== */
.refmind-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; margin-bottom: 8px;
}}
.refmind-pill {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; background: rgba(37,99,235,.12);
    color: var(--primary); margin-right: 6px;
}}
.refmind-pill.ready {{ background: rgba(5,150,105,.14); color: #10b981; }}
.refmind-pill.warn {{ background: rgba(180,83,9,.16); color: #f59e0b; }}
.refmind-summary {{ margin-top: 8px; color: var(--muted); line-height: 1.7; }}

/* ===== 文献库列表（侧边栏的独立框） ===== */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: var(--border) !important;
    border-radius: 12px;
    background: var(--bg2);
}}
/* 列表内按钮左对齐，像清单条目 */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] .stButton > button {{
    justify-content: flex-start;
    text-align: left;
}}
/* 当前库行内的删除按钮：默认半透明，悬停高亮 */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stHorizontalBlock"] > div:last-child .stButton > button {{
    opacity: 0.5;
    transition: opacity .15s ease, transform .15s ease;
    padding: 0 6px;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stHorizontalBlock"]:hover > div:last-child .stButton > button {{
    opacity: 1;
    transform: scale(1.1);
}}

/* ===== 右上角主题滑块 ===== */
.theme-switch-row {{ height: 0; }}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_global_css(theme: str = "light") -> None:
    st.markdown(_css(theme), unsafe_allow_html=True)
