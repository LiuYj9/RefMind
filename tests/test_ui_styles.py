"""聊天布局中长连续文本的 CSS 回归测试。"""

from __future__ import annotations

import unittest

from refmind.ui.styles import _css


class ChatOverflowStyleTests(unittest.TestCase):
    def test_chat_bubble_allows_flex_item_to_shrink(self) -> None:
        css = _css("light")

        self.assertIn('data-testid="stChatMessage"', css)
        self.assertIn('data-testid="stChatMessageContent"', css)
        self.assertRegex(
            css,
            r'(?s)\[data-testid="stChatMessageContent"\]\s*\{'
            r"[^}]*min-width:\s*0[^}]*max-width:\s*100%"
            r"[^}]*box-sizing:\s*border-box",
        )
        self.assertNotIn("calc(100% - 52px)", css)

    def test_long_markdown_tokens_can_wrap_inside_bubble(self) -> None:
        css = _css("light")

        self.assertRegex(
            css,
            r'(?s)\[data-testid="stChatMessage"\]\s+'
            r'\[data-testid="stMarkdownContainer"\]\s*\{'
            r"[^}]*overflow-wrap:\s*anywhere[^}]*word-break:\s*break-word",
        )

    def test_preformatted_content_scrolls_instead_of_being_clipped(self) -> None:
        css = _css("dark")

        self.assertIn("overflow-x: auto", css)
        self.assertIn("white-space: pre", css)
        self.assertIn("word-break: normal", css)


if __name__ == "__main__":
    unittest.main()
