#!/usr/bin/env python3
"""
author: Daniel Quigley
contact: dquigleydev@gmail.com

pdf structure tree check
logical structure tree from (tagged) pdfs

    python pdftree.py <file.pdf>
    python pdftree.py <file.pdf> --json output.json

TODO: borrow overelaf's functionality (except on output text, not code) to identify
        line in particular
"""

import sys
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import pikepdf
except ImportError:
    sys.exit("install pikepdf: pip install pikepdf --break-system-packages")


# data types
@dataclass
class StructNode:
    tag: str
    resolved_tag: Optional[str] = None   # set when tag maps via RoleMap
    title: Optional[str] = None
    alt_text: Optional[str] = None
    actual_text: Optional[str] = None
    lang: Optional[str] = None
    page: Optional[int] = None
    attributes: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


# extract
def extract_struct_elem(elem, pdf, page_map: dict,
                        role_map: dict, depth: int = 0) -> Optional[StructNode]:
    if depth > 500:
        return None  # guard against deep trees
    if not isinstance(elem, pikepdf.Dictionary):
        return None

    s_type = elem.get("/S")
    if s_type is None:
        return None

    raw_tag = str(s_type).lstrip("/")
    # resolve through RoleMap if tag is non-standard
    resolved = role_map.get(raw_tag, raw_tag)
    node = StructNode(tag=raw_tag, resolved_tag=resolved if resolved != raw_tag else None)

    if "/T" in elem:
        node.title = str(elem["/T"])
    if "/Alt" in elem:
        node.alt_text = str(elem["/Alt"])
    if "/ActualText" in elem:
        node.actual_text = str(elem["/ActualText"])
    if "/Lang" in elem:
        node.lang = str(elem["/Lang"])

    if "/Pg" in elem:
        try:
            page_ref = elem["/Pg"]
            node.page = page_map.get(page_ref.objgen)
        except Exception:
            pass

    if "/A" in elem:
        try:
            attrs = elem["/A"]
            if isinstance(attrs, pikepdf.Dictionary):
                for k, v in attrs.items():
                    node.attributes[str(k).lstrip("/")] = str(v)
            elif isinstance(attrs, pikepdf.Array):
                for attr_dict in attrs:
                    if isinstance(attr_dict, pikepdf.Dictionary):
                        for k, v in attr_dict.items():
                            node.attributes[str(k).lstrip("/")] = str(v)
        except Exception:
            pass

    if "/K" in elem:
        kids = elem["/K"]
        if isinstance(kids, pikepdf.Array):
            for kid in kids:
                child = extract_struct_elem(kid, pdf, page_map, role_map, depth + 1)
                if child:
                    node.children.append(child)
        elif isinstance(kids, pikepdf.Dictionary):
            child = extract_struct_elem(kids, pdf, page_map, role_map, depth + 1)
            if child:
                node.children.append(child)
        # integer refs to marked content are leaf nodes: skip

    return node


def _build_page_map(pdf) -> dict:
    """build {objgen to page_number}"""
    return {page.obj.objgen: i + 1 for i, page in enumerate(pdf.pages)}


def _extract_role_map(struct_root) -> dict:
    """
    return {non-standard-tag to standard-tag} from /RoleMap
    return empty dict if none declared
    """
    role_map = {}
    rm = struct_root.get("/RoleMap")
    if rm and isinstance(rm, pikepdf.Dictionary):
        for k, v in rm.items():
            role_map[k.lstrip("/")] = str(v).lstrip("/")
    return role_map


def extract_structure_tree(pdf_path: str) -> tuple[Optional[StructNode], dict]:
    """return (tree, info), metadata about pdf"""
    info = {"tagged": False, "has_tree": False, "role_map": {}, "page_count": 0}

    with pikepdf.open(pdf_path) as pdf:
        root = pdf.Root
        info["page_count"] = len(pdf.pages)

        if "/MarkInfo" in root:
            mark_info = root["/MarkInfo"]
            info["tagged"] = bool(mark_info.get("/Marked", False))

        if "/StructTreeRoot" not in root:
            return None, info

        info["has_tree"] = True
        struct_root = root["/StructTreeRoot"]

        role_map = _extract_role_map(struct_root)
        info["role_map"] = role_map

        page_map = _build_page_map(pdf)
        root_node = StructNode(tag="StructTreeRoot")

        if "/K" in struct_root:
            kids = struct_root["/K"]
            if isinstance(kids, pikepdf.Array):
                for kid in kids:
                    child = extract_struct_elem(kid, pdf, page_map, role_map)
                    if child:
                        root_node.children.append(child)
            elif isinstance(kids, pikepdf.Dictionary):
                child = extract_struct_elem(kids, pdf, page_map, role_map)
                if child:
                    root_node.children.append(child)

        return root_node, info


# serialization
def node_to_dict(node: StructNode) -> dict:
    d = {"tag": node.tag}
    if node.resolved_tag:
        d["resolvedTag"] = node.resolved_tag
    if node.title:
        d["title"] = node.title
    if node.alt_text:
        d["alt"] = node.alt_text
    if node.actual_text:
        d["actualText"] = node.actual_text
    if node.lang:
        d["lang"] = node.lang
    if node.page:
        d["page"] = node.page
    if node.attributes:
        d["attributes"] = node.attributes
    if node.children:
        d["children"] = [node_to_dict(c) for c in node.children]
    return d


# output to terminal
def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"

COLORS = {
    "info":  _rgb(46, 230, 246),    # #2ee6f6
    "ok":    _rgb(39, 242, 35),     # #27f223
    "warn":  _rgb(253, 179, 32),    # #fdb320
    "reset": "\033[0m",
    "dim":   _rgb(109, 115, 117),   # #6d7375
    "bold":  "\033[1m",
}

_MUTED    = _rgb(109, 115, 117)   # #6d7375 root, unknown
_GREEN    = _rgb(39, 242, 35)     # #27f223 lists, figures
_ORANGE   = _rgb(253, 179, 32)    # #fdb320 headings
_CYAN     = _rgb(46, 230, 246)    # #2ee6f6 tables, inline, links
_PINK     = _rgb(249, 195, 214)   # #f9c3d6 structural containers, block text

_TAG_COLORS = {
    # structural
    "document": _PINK, "part": _PINK, "sect": _PINK,
    "div": _PINK, "article": _PINK, "aside": _PINK,
    # headings
    "h": _ORANGE, "h1": _ORANGE, "h2": _ORANGE, "h3": _ORANGE,
    "h4": _ORANGE, "h5": _ORANGE, "h6": _ORANGE,
    # block text
    "p": _PINK, "blockquote": _PINK, "caption": _PINK,
    # inline
    "span": _CYAN, "link": _CYAN, "annot": _CYAN,
    "reference": _CYAN, "note": _CYAN,
    # lists
    "l": _GREEN, "li": _GREEN, "lbl": _GREEN, "lbody": _GREEN,
    # tables
    "table": _CYAN, "tr": _CYAN, "th": _CYAN,
    "td": _CYAN, "thead": _CYAN, "tbody": _CYAN,
    # figures, media
    "figure": _GREEN, "formula": _GREEN,
    # root
    "structtreeroot": _MUTED,
}

def _tag_color(tag: str) -> str:
    return _TAG_COLORS.get(tag.lower(), "\033[37m")  # default plain white


def _tree_lines(node: StructNode, prefix: str = "", is_last: bool = True) -> list[str]:
    """recursively build coloured tree lines with box-drawing connectors"""
    r = COLORS["reset"]
    dim = COLORS["dim"]
    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "

    tc = _tag_color(node.tag)
    tag_str = f"{tc}{node.tag}{r}"
    # show RoleMap resolution inline
    if node.resolved_tag:
        rc = _tag_color(node.resolved_tag)
        tag_str += f" {dim}→{r} {rc}{node.resolved_tag}{r}"

    meta = []
    if node.page:
        meta.append(f"{dim}p.{node.page}{r}")
    if node.title:
        meta.append(f"{dim}title=\"{node.title}\"{r}")
    if node.alt_text:
        clipped = node.alt_text[:40] + ("…" if len(node.alt_text) > 40 else "")
        meta.append(f"{dim}alt=\"{clipped}\"{r}")
    if node.lang:
        meta.append(f"{dim}lang={node.lang}{r}")
    if node.actual_text:
        clipped = node.actual_text[:40] + ("…" if len(node.actual_text) > 40 else "")
        meta.append(f"{dim}\"{clipped}\"{r}")

    meta_str = "  " + "  ".join(meta) if meta else ""
    lines = [f"{prefix}{connector}{tag_str}{meta_str}"]

    child_prefix = prefix + extension
    for i, child in enumerate(node.children):
        last = i == len(node.children) - 1
        lines.extend(_tree_lines(child, child_prefix, last))

    return lines


def _tree_stats(node: StructNode) -> dict:
    """return stats dict with total, max_depth, tag_counts, headings, nonstandard"""
    _STANDARD = {
        "document", "part", "art", "sect", "div", "blockquote", "caption",
        "toc", "toci", "index", "nonstruct", "private",
        "p", "h", "h1", "h2", "h3", "h4", "h5", "h6",
        "l", "li", "lbl", "lbody",
        "table", "tr", "th", "td", "thead", "tbody", "tfoot",
        "span", "quote", "note", "reference", "bibentry", "code",
        "figure", "formula", "form",
        "link", "annot", "ruby", "rb", "rt", "rp", "warichu", "wt", "wp",
        "structtreeroot",
    }

    counts: dict = {}
    depths: list = []
    headings: list = []
    nonstandard: set = set()

    def walk(n, depth):
        counts[n.tag] = counts.get(n.tag, 0) + 1
        depths.append(depth)
        tag_lower = n.tag.lower()
        if tag_lower in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            headings.append(int(tag_lower[1]))
        elif tag_lower == "h":
            headings.append(0)
        # not in the standard set AND
        # not resolved to standard tag via RoleMap
        resolved_lower = (n.resolved_tag or "").lower()
        if tag_lower not in _STANDARD and resolved_lower not in _STANDARD:
            nonstandard.add(n.tag)
        for c in n.children:
            walk(c, depth + 1)

    walk(node, 0)
    return {
        "total":       len(depths),
        "max_depth":   max(depths) if depths else 0,
        "tag_counts":  counts,
        "headings":    headings,
        "nonstandard": nonstandard,
    }


def print_tree(node: StructNode):
    stats = _tree_stats(node)
    b   = COLORS["bold"]
    r   = COLORS["reset"]
    dim = COLORS["dim"]
    warn = COLORS["warn"]

    top_tags = sorted(stats["tag_counts"].items(), key=lambda x: -x[1])[:8]
    tag_dist = "  ".join(f"{dim}{tag}{r} {n}" for tag, n in top_tags)

    print(f"  {b}nodes:{r} {stats['total']}   {b}depth:{r} {stats['max_depth']}")
    print(f"  {tag_dist}")

    # heading sequence check
    headings = stats["headings"]
    if headings:
        skips = []
        for i in range(1, len(headings)):
            if headings[i] > headings[i-1] + 1:
                skips.append(f"h{headings[i-1]}→h{headings[i]}")
        h1_count = headings.count(1)
        if h1_count == 0:
            print(f"  {warn}no H1 found{r}")
        elif h1_count > 1:
            print(f"  {warn}{h1_count}× H1 (expected one){r}")
        if skips:
            print(f"  {warn}heading level skip(s): {', '.join(skips)}{r}")

    # non-standard tags not covered by RoleMap
    if stats["nonstandard"]:
        ns = ", ".join(sorted(stats["nonstandard"]))
        print(f"  {warn}unmapped non-standard tag(s): {ns}{r}")

    print()

    children = node.children if node.children else [node]
    for i, child in enumerate(children):
        last = i == len(children) - 1
        for line in _tree_lines(child, prefix="  ", is_last=last):
            print(line)


def status(level: str, msg: str):
    c = COLORS.get(level, "")
    r = COLORS["reset"]
    print(f"  {c}{msg}{r}")


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


# entry point
def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python pdftree.py <file.pdf> [--json output.json]")

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        sys.exit(f"file not found: {pdf_path}")

    output_dir = Path("access_output")
    output_dir.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem

    tree, info = extract_structure_tree(pdf_path)

    # to .txt
    import io as _io
    buf = _io.StringIO()

    def emit(line: str = ""):
        """print to terminal and capture in buffer simultaneously"""
        print(line)
        buf.write(line + "\n")

    def emit_status(level: str, msg: str):
        c = COLORS.get(level, "")
        r = COLORS["reset"]
        emit(f"  {c}{msg}{r}")

    # status header
    if info["tagged"]:
        emit_status("ok", "pdf is tagged (marked)")
    emit_status("info", f"{info['page_count']} page(s)")
    if info["role_map"]:
        mappings = "  ".join(f"{k}→{v}" for k, v in sorted(info["role_map"].items()))
        emit_status("info", f"rolemap: {mappings}")

    if not info["has_tree"]:
        sys.exit("no structure tree found; pdf may not be tagged")
    if not tree:
        sys.exit("no structure tree to display")

    json_output = None
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--json" and i + 1 < len(args):
            json_output = args[i + 1]

    #tree output
    emit(f"\n")
    emit(f"  structure tree: {pdf_path}")

    # capture print_tree output
    import io as _io2
    old_stdout = sys.stdout
    sys.stdout = tree_buf = _io2.StringIO()
    print_tree(tree)
    sys.stdout = old_stdout
    tree_output = tree_buf.getvalue()
    print(tree_output, end="")
    buf.write(tree_output)

    emit()

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(node_to_dict(tree), f, indent=2, ensure_ascii=False)
        emit_status("info", f"json saved to: {json_output}")

    # write report
    txt_path = output_dir / f"pdftree_{stem}_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(strip_ansi(buf.getvalue()))
    emit_status("info", f"report written to: {txt_path}")

    emit()


if __name__ == "__main__":
    main()
