#!/usr/bin/env python3
"""
author: Daniel Quigley
contact: dquigleydev@gmail.com

pdf text accessibility checker
check whether pdf content is usable
  is pdf tagged?
  extractable text or is it a scanned image?
  figures missing alt text?
  document language declared?
  untagged content?
  empty or short alt texts?
  title metadata exist?

    python textaccess.py <file.pdf>

"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass, field

try:
    import pikepdf
except ImportError:
    sys.exit("install pikepdf: pip install pikepdf --break-system-packages")

try:
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    import io as _io
    HAS_PDFMINER = True
    _PDFMINER_ERR = None
except ImportError as e:
    HAS_PDFMINER = False
    _PDFMINER_ERR = str(e)


# namespace uris (pdf 2.0, iso 32000-2)
NS_PDF17  = "http://iso.org/pdf/ssn"
NS_PDF20  = "http://iso.org/pdf2/ssn"
NS_MATHML = "http://www.w3.org/1998/Math/MathML"

# standard structure types, pdf 1.7 (classic default namespace)
STANDARD_17 = {
    "document", "part", "art", "sect", "div", "blockquote", "caption",
    "toc", "toci", "index", "nonstruct", "private",
    "p", "h", "h1", "h2", "h3", "h4", "h5", "h6",
    "l", "li", "lbl", "lbody",
    "table", "tr", "th", "td", "thead", "tbody", "tfoot",
    "span", "quote", "note", "reference", "bibentry", "code",
    "figure", "formula", "form",
    "link", "annot", "ruby", "rb", "rt", "rp", "warichu", "wt", "wp",
}

# standard structure types, pdf 2.0 namespace; Hn is open-ended (h7, h8, ...)
# and handled by _is_hn below
STANDARD_20 = {
    "document", "documentfragment", "part", "div", "aside", "nonstruct",
    "p", "h", "title", "fenote", "sub", "lbl", "em", "strong", "span",
    "l", "li", "lbody",
    "table", "tr", "th", "td", "thead", "tbody", "tfoot", "caption",
    "figure", "formula", "form", "artifact",
    "link", "annot", "ruby", "rb", "rt", "rp", "warichu", "wt", "wp",
    "index", "toc", "toci",
}

_HN_RE = re.compile(r"^h\d+$")


def _is_hn(tag: str) -> bool:
    return tag == "h" or bool(_HN_RE.match(tag))


def _is_standard(tag: str, ns_uri) -> bool:
    """standard type test relative to the element's (resolved) namespace"""
    t = tag.lower()
    if ns_uri == NS_MATHML:
        return True
    if ns_uri == NS_PDF20:
        return t in STANDARD_20 or _is_hn(t)
    # classic default namespace and explicit pdf 1.7 namespace
    return t in STANDARD_17


def _ns_uri(ns_obj):
    """uri string of a namespace dictionary, or None"""
    if isinstance(ns_obj, pikepdf.Dictionary) and "/NS" in ns_obj:
        return str(ns_obj["/NS"])
    return None


def _build_role_maps(pdf):
    """collect classic /RoleMap and pdf 2.0 per-namespace /RoleMapNS mappings

    returns (classic, ns_maps):
      classic: {name: name}, the pdf 1.x role map on the structure tree root
      ns_maps: {namespace uri: {name: (target name, target namespace uri or None)}}
        target namespace None means the default standard namespace
    """
    classic = {}
    ns_maps = {}
    root = pdf.Root
    if "/StructTreeRoot" not in root:
        return classic, ns_maps
    st = root["/StructTreeRoot"]

    rm = st.get("/RoleMap")
    if isinstance(rm, pikepdf.Dictionary):
        for k, v in rm.items():
            classic[str(k).lstrip("/")] = str(v).lstrip("/")

    namespaces = st.get("/Namespaces")
    if isinstance(namespaces, pikepdf.Array):
        for ns in namespaces:
            if not isinstance(ns, pikepdf.Dictionary):
                continue
            uri = _ns_uri(ns)
            if uri is None:
                continue
            mapping = {}
            rmns = ns.get("/RoleMapNS")
            if isinstance(rmns, pikepdf.Dictionary):
                for k, v in rmns.items():
                    name = str(k).lstrip("/")
                    if isinstance(v, pikepdf.Array) and len(v) >= 1:
                        target = str(v[0]).lstrip("/")
                        tgt_uri = _ns_uri(v[1]) if len(v) > 1 else None
                        mapping[name] = (target, tgt_uri)
                    else:
                        mapping[name] = (str(v).lstrip("/"), None)
            ns_maps[uri] = mapping
    return classic, ns_maps


def _resolve_type(elem, classic, ns_maps):
    """resolve an element's /S through role maps to (tag, namespace uri)

    follows /RoleMapNS for namespaced elements and classic /RoleMap
    otherwise, with a cycle guard; stops at the first standard type or
    when no further mapping applies
    """
    tag = str(elem.get("/S", "")).lstrip("/")
    ns_uri = _ns_uri(elem.get("/NS"))
    seen = set()
    while tag and not _is_standard(tag, ns_uri):
        key = (tag, ns_uri)
        if key in seen:
            break
        seen.add(key)
        if ns_uri is not None and ns_uri in ns_maps and tag in ns_maps[ns_uri]:
            tag, ns_uri = ns_maps[ns_uri][tag]
        elif ns_uri is None and tag in classic:
            tag = classic[tag]
        else:
            break
    return tag, ns_uri


def _extract_text(pdf_path: str) -> str:
    """extract all text from pdf using pdfminer lower-level api"""
    output = _io.StringIO()
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(pdf_path, "rb") as f:
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
    device.close()
    return output.getvalue()


# data types
@dataclass
class Issue:
    level: str      # "error" | "warning" | "info"
    check: str
    detail: str


@dataclass
class AuditResult:
    path: str
    issues: list[Issue] = field(default_factory=list)

    def add(self, level, check, detail):
        self.issues.append(Issue(level, check, detail))

    def summary(self):
        errors   = sum(1 for i in self.issues if i.level == "error")
        warnings = sum(1 for i in self.issues if i.level == "warning")
        infos    = sum(1 for i in self.issues if i.level == "info")
        return errors, warnings, infos


# checks
def check_tagging(pdf, result):
    root = pdf.Root
    marked = False

    if "/MarkInfo" in root:
        mark_info = root["/MarkInfo"]
        marked = bool(mark_info.get("/Marked", False))

    if not marked:
        result.add("error", "tagging",
                   "pdf is not tagged: screen readers cannot determine reading order or structure")
    else:
        result.add("info", "tagging", "pdf is tagged (marked).")

    return marked


def check_language(pdf, result):
    root = pdf.Root
    lang = root.get("/Lang")
    if not lang or not str(lang).strip():
        result.add("error", "language",
                   "no document language declared (/Lang missing from document catalog)")
    else:
        result.add("info", "language", f"document language: {str(lang).strip()}")


def _xmp_dc_title(pdf) -> str:
    """dc:title from xmp metadata, empty string if absent"""
    try:
        xmp = pdf.open_metadata()
        v = xmp.get("{http://purl.org/dc/elements/1.1/}title", "")
        if isinstance(v, (list, tuple)):
            v = v[0] if v else ""
        return str(v).strip()
    except Exception:
        return ""


def check_metadata(pdf, result):
    """document info dict: pdf version, page count, encryption, font encoding"""
    info  = pdf.docinfo
    root  = pdf.Root
    pages = pdf.pages

    # document information dictionary
    def get(key):
        v = info.get(key, "")
        return str(v).strip() if v else ""

    title    = get("/Title")
    author   = get("/Author")
    subject  = get("/Subject")
    keywords = get("/Keywords")
    creator  = get("/Creator")    # authoring tool (e.g. LaTeX (intended inspection tool), Word)
    producer = get("/Producer")   # pdf library (e.g. Distiller, pikepdf)
    created  = get("/CreationDate")
    modified = get("/ModDate")

    if not title:
        # pdf 2.0 deprecates the info dictionary; the authoritative title
        # location is xmp dc:title, so consult it before warning
        xmp_title = _xmp_dc_title(pdf)
        if xmp_title:
            result.add("info", "metadata: title",
                       f"{xmp_title} (xmp dc:title; info dictionary /Title absent, "
                       "deprecated in pdf 2.0)")
        else:
            result.add("warning", "metadata: title",
                       "no /Title in info dictionary and no dc:title in xmp: "
                       "screen readers announce title when a document opens")
    else:
        result.add("info", "metadata: title", title)

    # note: technically author/subject not wcag requirements
    if author:
        result.add("info", "metadata: author", author)
    if subject:
        result.add("info", "metadata: subject", subject)
    if keywords:
        result.add("info", "metadata: keywords", keywords)

    if creator:
        result.add("info", "metadata: creator", creator)
    if producer:
        result.add("info", "metadata: producer", producer)

    if created:
        result.add("info", "metadata: created", _parse_pdf_date(created))
    if modified:
        result.add("info", "metadata: modified", _parse_pdf_date(modified))

    # pdf version
    try:
        version = f"{pdf.pdf_version}"
        # PDF/UA-1 is defined over pdf 1.7; PDF/UA-2 over pdf 2.0
        major, minor = version.split(".")
        if int(major) < 1 or (int(major) == 1 and int(minor) < 7):
            result.add("warning", "pdf version",
                       f"pdf {version}: PDF/UA requires pdf 1.7 (UA-1) or pdf 2.0 (UA-2)")
        else:
            result.add("info", "pdf version", f"pdf {version}")
    except Exception:
        pass

    # simple page count
    result.add("info", "page count", f"{len(pages)} page(s)")

    # check for encryption
    if pdf.is_encrypted:
        result.add("warning", "encryption",
                   "document is encrypted; some assistive technologies cannot "
                   "read encrypted pdfs (depending on permission flags)")
    else:
        result.add("info", "encryption", "document is not encrypted")

    # xmp metadata may carry richer accessibility metadata (pdfuaid:part, dc:title etc.)
    try:
        xmp = pdf.open_metadata()
        xmp_entries = {}
        namespaces_of_interest = {
            "http://www.aiim.org/pdfua/ns/id/":      "pdf/ua",
            "http://purl.org/dc/elements/1.1/":      "dc",
            "http://ns.adobe.com/pdf/1.3/":          "pdf",
            "http://ns.adobe.com/xap/1.0/":          "xmp",
        }
        for key in xmp:
            for ns, prefix in namespaces_of_interest.items():
                if key.startswith("{" + ns + "}"):
                    local = key[len(ns)+2:]
                    xmp_entries[f"{prefix}:{local}"] = str(xmp[key])

        if xmp_entries:
            for k, v in sorted(xmp_entries.items()):
                result.add("info", f"xmp; {k}", v)

        # explicit PDF/UA conformance check
        pdfua_key = "{http://www.aiim.org/pdfua/ns/id/}part"
        if pdfua_key in xmp:
            result.add("info", "pdf/ua",
                       f"document declares PDF/UA-{xmp[pdfua_key]} conformance")
        else:
            result.add("info", "pdf/ua",
                       "no PDF/UA conformance declaration found in XMP metadata")
    except Exception:
        pass

    # font encoding coverage check whether embedded fonts declare ToUnicode tables
    try:
        fonts_total    = 0
        fonts_no_unicode = 0
        seen = set()

        for page in pages:
            resources = page.get("/Resources")
            if not resources or not isinstance(resources, pikepdf.Dictionary):
                continue
            font_dict = resources.get("/Font")
            if not font_dict or not isinstance(font_dict, pikepdf.Dictionary):
                continue
            for font_key, font_ref in font_dict.items():
                try:
                    font_obj = font_ref if isinstance(font_ref, pikepdf.Dictionary) \
                               else pdf.get_object(font_ref.objgen)
                    if not isinstance(font_obj, pikepdf.Dictionary):
                        continue
                    name = str(font_obj.get("/BaseFont", font_key))
                    if name in seen:
                        continue
                    seen.add(name)
                    fonts_total += 1
                    if "/ToUnicode" not in font_obj:
                        fonts_no_unicode += 1
                except Exception:
                    pass

        if fonts_total > 0:
            if fonts_no_unicode:
                result.add("warning", "font encoding",
                           f"{fonts_no_unicode}/{fonts_total} font(s) have no ToUnicode table "
                           "glyphs in these fonts may not map correctly to unicode text, "
                           "making content unreadable by screen readers")
            else:
                result.add("info", "font encoding",
                           f"all {fonts_total} font(s) have ToUnicode tables")
    except Exception:
        pass


def _parse_pdf_date(raw: str) -> str:
    """parse pdf date string (D:YYYYMMDDHHmmSS) -> readable form"""
    s = raw.lstrip("D:").rstrip("Z").replace("'", "")
    m = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})?", s)
    if m:
        y, mo, d, h, mi = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        return f"{y}-{mo}-{d} {h}:{mi}"
    return raw


def check_extractable_text(pdf_path, pdf, result):
    if not HAS_PDFMINER:
        detail = f" ({_PDFMINER_ERR})" if _PDFMINER_ERR else ""
        result.add("info", "extractable text",
                   f"pdfminer.six not importable{detail}: skipping text extraction check")
        return

    text = _extract_text(pdf_path)
    stripped = text.strip()

    if not stripped:
        result.add("error", "extractable text",
                   "no extractable text found; the pdf is likely a scanned image! "
                   "ocr must be applied before it is accessible (not supported here)")
        return

    page_count = len(pdf.pages)
    char_count = len(stripped)
    word_count = len(stripped.split())
    chars_per_page = char_count / page_count if page_count else 0

    # encoding, glyph mapping check counts characters that signal broken ToUnicode maps:
    # U+FFFD  replacement character (unmapped glyph)
    # U+F000-U+F8FF  private use area (font-internal codepoints leaking out)
    # C0/C1 control characters (except tab, newline, carriage return)
    def is_suspect(c):
        cp = ord(c)
        if cp == 0xFFFD:
            return True
        if 0xE000 <= cp <= 0xF8FF:   # private use area
            return True
        if cp < 0x09 or (0x0E <= cp <= 0x1F) or (0x80 <= cp <= 0x9F):
            return True
        return False

    suspect_chars = sum(1 for c in stripped if is_suspect(c))
    suspect_ratio = suspect_chars / char_count if char_count else 0

    if suspect_ratio > 0.05:
        result.add("error", "extractable text: encoding",
                   f"{suspect_chars} suspect characters ({suspect_ratio:.0%} of text); "
                   "likely missing or broken ToUnicode tables in one or more fonts; "
                   "screen readers will read garbled text even though extraction succeeds")
    elif suspect_ratio > 0.01:
        result.add("warning", "extractable text: encoding",
                   f"{suspect_chars} suspect characters ({suspect_ratio:.0%} of text); "
                   "some glyphs may not map correctly to unicode")

    # volume check
    if chars_per_page < 50:
        result.add("warning", "extractable text",
                   f"{word_count} words / {char_count} characters across {page_count} page(s) "
                   f"({chars_per_page:.0f} chars/page); "
                   "some pages may be scanned images without ocr")
    else:
        result.add("info", "extractable text",
                   f"{word_count} words / {char_count} characters across {page_count} page(s) "
                   f"({chars_per_page:.0f} chars/page)")

    # take the first 120 non-whitespace-run characters as quick spot-check
    sample = " ".join(stripped.split())[:120]
    if len(" ".join(stripped.split())) > 120:
        sample += "..."
    result.add("info", "extractable text", f'"{sample}"')


def _table_has_summary(elem) -> bool:
    """table summary lives in the /A attribute object(s), not on the element;
    accept the legacy direct key as well"""
    if "/Summary" in elem:
        return True
    attrs = elem.get("/A")
    if attrs is None:
        return False
    attr_list = list(attrs) if isinstance(attrs, pikepdf.Array) else [attrs]
    for a in attr_list:
        if isinstance(a, pikepdf.Dictionary) and "/Summary" in a:
            return True
    return False


def _table_has_th(elem, classic, ns_maps, depth=0) -> bool:
    """search for TH cells through THead/TBody/TFoot/TR nesting"""
    if depth > 4 or not isinstance(elem, pikepdf.Dictionary):
        return False
    kids = elem.get("/K")
    if kids is None:
        return False
    kid_list = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
    for kid in kid_list:
        if not isinstance(kid, pikepdf.Dictionary):
            continue
        tag, _ = _resolve_type(kid, classic, ns_maps)
        t = tag.lower()
        if t == "th":
            return True
        if t in {"thead", "tbody", "tfoot", "tr"}:
            if _table_has_th(kid, classic, ns_maps, depth + 1):
                return True
    return False


def check_structure_inventory(pdf, result, classic, ns_maps):
    """walk structure tree; audit figures, tables, lists, headings, links, forms

    element types are resolved through /RoleMap and /RoleMapNS before
    classification, so namespaced source types (e.g. latex 'section')
    are counted under the standard types they map to
    """
    if "/StructTreeRoot" not in pdf.Root:
        return

    # each entry is [total, issues] where issues is sub-dictionary
    counts = {
        "figure":  {"total": 0, "missing_alt": 0, "empty_alt": 0},
        "table":   {"total": 0, "missing_summary": 0, "missing_headers": 0},
        "list":    {"total": 0},
        "heading": {"total": 0, "empty": 0},
        "link":    {"total": 0, "missing_alt": 0},
        "form":    {"total": 0},
        "formula": {"total": 0, "missing_alt": 0},
    }

    def walk(elem):
        if not isinstance(elem, pikepdf.Dictionary):
            return
        s_type = elem.get("/S")
        if s_type:
            resolved, _ = _resolve_type(elem, classic, ns_maps)
            tag = resolved.lower()

            # figures
            if tag == "figure":
                counts["figure"]["total"] += 1
                alt = elem.get("/Alt")
                if alt is None:
                    counts["figure"]["missing_alt"] += 1
                elif len(str(alt).strip()) < 4:
                    counts["figure"]["empty_alt"] += 1

            # tables
            elif tag == "table":
                counts["table"]["total"] += 1
                if not _table_has_summary(elem):
                    # a caption child also declares purpose
                    has_caption = False
                    kids = elem.get("/K")
                    if kids:
                        kid_list = list(kids) if isinstance(kids, pikepdf.Array) else [kids]
                        for kid in kid_list:
                            if isinstance(kid, pikepdf.Dictionary):
                                kt, _ = _resolve_type(kid, classic, ns_maps)
                                if kt.lower() == "caption":
                                    has_caption = True
                    if not has_caption:
                        counts["table"]["missing_summary"] += 1
                if not _table_has_th(elem, classic, ns_maps):
                    counts["table"]["missing_headers"] += 1

            # lists
            elif tag == "l":
                counts["list"]["total"] += 1

            # headings (pdf 2.0 allows Hn beyond h6)
            elif _is_hn(tag):
                counts["heading"]["total"] += 1
                # note: with no actual-text and no children with text is empty
                actual = elem.get("/ActualText")
                kids = elem.get("/K")
                has_content = bool(actual) or bool(kids)
                if not has_content:
                    counts["heading"]["empty"] += 1

            # links
            elif tag == "link":
                counts["link"]["total"] += 1
                alt = elem.get("/Alt")
                actual = elem.get("/ActualText")
                kids = elem.get("/K")
                if not alt and not actual and not kids:
                    counts["link"]["missing_alt"] += 1

            # forms
            elif tag == "form":
                counts["form"]["total"] += 1

            # formulas
            elif tag == "formula":
                counts["formula"]["total"] += 1
                alt = elem.get("/Alt")
                if not alt or len(str(alt).strip()) < 2:
                    counts["formula"]["missing_alt"] += 1

        kids = elem.get("/K")
        if kids is None:
            return
        if isinstance(kids, pikepdf.Array):
            for kid in kids:
                if isinstance(kid, pikepdf.Dictionary):
                    walk(kid)
        elif isinstance(kids, pikepdf.Dictionary):
            walk(kids)

    struct_root = pdf.Root["/StructTreeRoot"]
    root_kids = struct_root.get("/K")
    if root_kids:
        if isinstance(root_kids, pikepdf.Array):
            for kid in root_kids:
                walk(kid)
        elif isinstance(root_kids, pikepdf.Dictionary):
            walk(root_kids)

    # figures
    f = counts["figure"]
    if f["total"] == 0:
        result.add("info", "figures", "no Figure elements found")
    else:
        if f["missing_alt"]:
            result.add("error", "figures",
                       f"{f['missing_alt']}/{f['total']} Figure element(s) have no /Alt text")
        if f["empty_alt"]:
            result.add("warning", "figures",
                       f"{f['empty_alt']}/{f['total']} Figure element(s) have empty or "
                       "suspiciously short /Alt text")
        if not f["missing_alt"] and not f["empty_alt"]:
            result.add("info", "figures",
                       f"all {f['total']} Figure element(s) have /Alt text")

    # tables
    t = counts["table"]
    if t["total"] == 0:
        result.add("info", "tables", "no Table elements found")
    else:
        if t["missing_headers"]:
            result.add("warning", "tables",
                       f"{t['missing_headers']}/{t['total']} Table(s) have no TH header cells; "
                       "screen readers cannot associate data cells with column/row headers")
        if t["missing_summary"]:
            result.add("warning", "tables",
                       f"{t['missing_summary']}/{t['total']} Table(s) have no /Summary "
                       "or Caption child; purpose is undeclared")
        if not t["missing_headers"] and not t["missing_summary"]:
            result.add("info", "tables",
                       f"all {t['total']} Table(s) have headers and a caption or summary")

    # lists
    l = counts["list"]
    if l["total"] == 0:
        result.add("info", "lists", "no List (L) elements found")
    else:
        result.add("info", "lists", f"{l['total']} List element(s) found")

    # headings
    h = counts["heading"]
    if h["total"] == 0:
        result.add("warning", "headings",
                   "no heading elements (H, H1-H6) found; "
                   "documents without headings have no navigable structure for AT users")
    else:
        if h["empty"]:
            result.add("warning", "headings",
                       f"{h['empty']}/{h['total']} heading element(s) appear to have no text content")
        else:
            result.add("info", "headings", f"{h['total']} heading element(s) found")

    # links
    lk = counts["link"]
    if lk["total"] > 0:
        if lk["missing_alt"]:
            result.add("warning", "links",
                       f"{lk['missing_alt']}/{lk['total']} Link element(s) have no accessible name "
                       "(/Alt, /ActualText, or child content)")
        else:
            result.add("info", "links", f"{lk['total']} Link element(s) found, all have accessible names")

    # forms
    if counts["form"]["total"] > 0:
        result.add("info", "forms",
                   f"{counts['form']['total']} Form element(s) found; "
                   "verify that all form fields have associated labels in the structure tree")

    # formulas (formulae?)
    fm = counts["formula"]
    if fm["total"] > 0:
        if fm["missing_alt"]:
            result.add("warning", "formulas",
                       f"{fm['missing_alt']}/{fm['total']} Formula element(s) have no /Alt text; "
                       "mathematical content is inaccessible without a text alternative")
        else:
            result.add("info", "formulas",
                       f"all {fm['total']} Formula element(s) have /Alt text")


def check_untagged_content(pdf, result):
    """heuristic: pages with content streams but no BDC/BMC markers likely have untagged content"""
    untagged_pages = []

    for i, page in enumerate(pdf.pages):
        contents = page.get("/Contents")
        if contents is None:
            continue
        try:
            if isinstance(contents, pikepdf.Array):
                stream_bytes = b"".join(bytes(pdf.get_object(ref)) for ref in contents)
            else:
                stream_bytes = bytes(contents.read_bytes())
            # look for marked content sequences: BDC/BMC operators
            if not re.search(rb'\bBDC\b|\bBMC\b', stream_bytes):
                untagged_pages.append(i + 1)
        except Exception:
            pass

    if untagged_pages:
        sample = untagged_pages[:5]
        more = f" (and {len(untagged_pages) - 5} more)" if len(untagged_pages) > 5 else ""
        result.add("warning", "untagged content",
                   f"{len(untagged_pages)} page(s) appear to have no marked content sequences: "
                   f"pp. {', '.join(str(p) for p in sample)}{more}")
    else:
        result.add("info", "untagged content",
                   "all pages with content appear to use marked content sequences")


def check_structure_types(pdf, result, classic, ns_maps):
    """flag structure types that remain non-standard after role map resolution

    pdf 2.0 documents carry per-namespace role maps (/Namespaces with
    /RoleMapNS); a type is only a problem if it resolves to nothing
    standard through either mechanism
    """
    if "/StructTreeRoot" not in pdf.Root:
        return

    unresolved = set()
    source_namespaces = set()

    def walk(elem):
        if not isinstance(elem, pikepdf.Dictionary):
            return
        s_type = elem.get("/S")
        if s_type:
            uri = _ns_uri(elem.get("/NS"))
            if uri:
                source_namespaces.add(uri)
            tag, final_uri = _resolve_type(elem, classic, ns_maps)
            if not _is_standard(tag, final_uri):
                origin = str(s_type).lstrip("/")
                if uri:
                    unresolved.add(f"{origin} ({uri})")
                else:
                    unresolved.add(origin)
        kids = elem.get("/K")
        if isinstance(kids, pikepdf.Array):
            for kid in kids:
                if isinstance(kid, pikepdf.Dictionary):
                    walk(kid)
        elif isinstance(kids, pikepdf.Dictionary):
            walk(kids)

    struct_root = pdf.Root["/StructTreeRoot"]
    kids = struct_root.get("/K")
    if kids:
        if isinstance(kids, pikepdf.Array):
            for kid in kids:
                walk(kid)
        elif isinstance(kids, pikepdf.Dictionary):
            walk(kids)

    if source_namespaces:
        result.add("info", "structure namespaces",
                   f"structure element namespace(s): {', '.join(sorted(source_namespaces))}")

    if unresolved:
        result.add("warning", "structure types",
                   f"structure type(s) not resolvable to standard types through "
                   f"/RoleMap or /RoleMapNS: {', '.join(sorted(unresolved))}; "
                   "AT cannot interpret these")
    else:
        result.add("info", "structure types",
                   "all structure types are standard or resolve to standard types "
                   "through role maps")


# report
COLORS = {
    "error":   "\033[91m",
    "warning": "\033[93m",
    "info":    "\033[94m",
    "reset":   "\033[0m",
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _report_lines(result: AuditResult) -> list[str]:
    """report lines with embedded ANSI color codes"""
    lines = []
    lines.append(f"\n")
    lines.append(f"\tpdf text accessibility: {result.path}")
    lines.append(f"\n")
    
    for level in ("error", "warning", "info"):
        issues = [i for i in result.issues if i.level == level]
        for issue in issues:
            c = COLORS[level]
            r = COLORS["reset"]
            lines.append(f"\t{c}[{level}] {issue.check}{r}")
            lines.append(f"\t\t{issue.detail}")
            lines.append("")

    errors, warnings, infos = result.summary()
    lines.append(
        f"\t{COLORS['error']}errors: {errors}{COLORS['reset']}  "
        f"{COLORS['warning']}warnings: {warnings}{COLORS['reset']}  "
        f"{COLORS['info']}info: {infos}{COLORS['reset']}\n"
    )
    return lines


def print_report(result: AuditResult):
    for line in _report_lines(result):
        print(line)


def write_report(result: AuditResult, path: str):
    lines = [strip_ansi(l) for l in _report_lines(result)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python textaccess.py <file.pdf>")

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        sys.exit(f"file not found: {pdf_path}")
    if Path(pdf_path).suffix.lower() != ".pdf":
        sys.exit(f"expected a .pdf file, got: {pdf_path}")

    result = AuditResult(path=pdf_path)

    try:
        with pikepdf.open(pdf_path) as pdf:
            check_tagging(pdf, result)
            check_language(pdf, result)
            check_metadata(pdf, result)
            classic, ns_maps = _build_role_maps(pdf)
            check_structure_inventory(pdf, result, classic, ns_maps)
            check_untagged_content(pdf, result)
            check_structure_types(pdf, result, classic, ns_maps)
            # inside the with block: check_extractable_text touches pdf.pages,
            # which is invalid on a closed pdf
            check_extractable_text(pdf_path, pdf, result)
    except pikepdf.PdfError as e:
        sys.exit(f"could not open pdf: {e}")

    print_report(result)

    output_dir = Path("access_output")
    output_dir.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem
    txt_path = output_dir / f"textaccess_{stem}_report.txt"
    write_report(result, str(txt_path))
    print(f"\treport written to: {txt_path}\n")

    errors, _, _ = result.summary()
    sys.exit(1 if errors else 0)
