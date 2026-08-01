"""
Small, intentionally-scoped Org-mode -> HTML converter.

Not a full Org parser — it covers the practical subset that READMEs
actually use: headings, #+BEGIN_SRC/#+END_SRC code blocks (syntax
highlighted via Pygments), [[url][text]] links, "- " bullet lists, and
plain paragraphs.
"""

import re
from html import escape

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

LINK_RE = re.compile(r"\[\[([^\]]+?)\](?:\[([^\]]+?)\])?\]")
HEADING_RE = re.compile(r"^(\*+)\s+(.*)$")
SRC_BEGIN_RE = re.compile(r"^\s*#\+BEGIN_SRC\s*(\S+)?", re.IGNORECASE)
SRC_END_RE = re.compile(r"^\s*#\+END_SRC", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*-\s+(.*)$")


def _inline(text: str) -> str:
    escaped = escape(text)

    def repl(match: re.Match) -> str:
        url = escape(match.group(1))
        label = escape(match.group(2)) if match.group(2) else url
        return f'<a href="{url}" target="_blank" rel="noopener">{label}</a>'

    return LINK_RE.sub(repl, escaped)


def _highlight_src(code: str, lang: str | None) -> str:
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except ClassNotFound:
        lexer = TextLexer()
    formatter = HtmlFormatter(nowrap=False, cssclass="highlight")
    return highlight(code, lexer, formatter)


def render_org_to_html(text: str) -> str:
    html_parts: list[str] = []
    in_src = False
    src_lang: str | None = None
    src_lines: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    for line in text.splitlines():
        if in_src:
            if SRC_END_RE.match(line):
                html_parts.append(_highlight_src("\n".join(src_lines), src_lang))
                in_src, src_lang, src_lines = False, None, []
            else:
                src_lines.append(line)
            continue

        begin_match = SRC_BEGIN_RE.match(line)
        if begin_match:
            close_list()
            in_src, src_lang = True, begin_match.group(1)
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            close_list()
            level = min(len(heading_match.group(1)), 6)
            html_parts.append(f"<h{level}>{_inline(heading_match.group(2))}</h{level}>")
            continue

        bullet_match = BULLET_RE.match(line)
        if bullet_match:
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{_inline(bullet_match.group(1))}</li>")
            continue

        close_list()

        if line.strip() == "":
            continue

        html_parts.append(f"<p>{_inline(line)}</p>")

    close_list()
    return "\n".join(html_parts)
