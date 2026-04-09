#!/usr/bin/env python3
"""
author: Daniel Quigley
contact: dquigleydev@gmail.com

wcag 2.1/2.2 aa accessibility audit
NOTE: only for static sites; interaction-dependent criteria marked n/a in matrix

    python ally.py https://example.com
    python ally.py https://example.com/page1 https://example.com/page2
    python ally.py --matrix https://example.com/page1 https://example.com/page2

TODO: traverse drop-down menus? could be overwhelming if not reported to separate output locations....
"""

import sys
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup, Tag
except ImportError:
    sys.exit("install deps: pip install requests beautifulsoup4")


# data types
@dataclass
class Issue:
    level: str          # "error" | "warning" | "info"
    criterion: str      # e.g. "1.1.1"
    title: str
    detail: str
    element: str = ""   # snippet of offending HTML


@dataclass
class Pass:
    criterion: str
    title: str


@dataclass
class AuditResult:
    url: str
    issues: list[Issue] = field(default_factory=list)
    passes: list[Pass]  = field(default_factory=list)
    externals: dict[str, list[str]] = field(default_factory=lambda: {
        "stylesheets": [],
        "scripts": [],
        "iframes": [],
        "fonts": [],
        "media": [],
    })

    def add(self, level, criterion, title, detail, element=""):
        self.issues.append(Issue(level, criterion, title, detail, element))

    def passed(self, criterion, title):
        self.passes.append(Pass(criterion, title))

    def summary(self):
        errors   = sum(1 for i in self.issues if i.level == "error")
        warnings = sum(1 for i in self.issues if i.level == "warning")
        infos    = sum(1 for i in self.issues if i.level == "info")
        return errors, warnings, infos


# helpers
def snippet(tag: Tag, max_len: int = 120) -> str:
    s = str(tag)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def visible_text(tag: Tag) -> str:
    return tag.get_text(separator=" ", strip=True)


def is_decorative(tag: Tag) -> bool:
    return tag.get("alt") == ""


def has_aria_label(tag: Tag) -> bool:
    return bool(tag.get("aria-label") or tag.get("aria-labelledby"))


def parse_inline_color(style: str, prop: str) -> Optional[str]:
    m = re.search(rf"{prop}\s*:\s*([^;]+)", style or "")
    return m.group(1).strip() if m else None


def _is_in_tab_order(el: Tag) -> bool:
    """return true if element participates in the natural tab order"""
    tabindex = el.get("tabindex")
    if tabindex == "-1":
        return False
    if el.name in {"a", "button", "input", "select", "textarea"}:
        return True
    if tabindex is not None:
        return True
    return False


def _name_check(el: Tag, soup, mechanisms: list[str]) -> tuple[bool, list[str]]:
    """
    try each accessible name mechanism in order; first success short-circuits.
    returns (found: bool, failure_reasons: list[str]); if found is True the list is empty.

    mechanisms (ordered):
        "text_content"    ; visible text inside the element
        "alt"             ; alt attribute (images, input[type=image], area)
        "svg_title"       ; first <title> child of an <svg>
        "wrapped_label"   ; element is a descendant of a <label>
        "explicit_label"  ; <label for="id"> matching element's id
        "aria_label"      ; aria-label attribute present and non-empty
        "aria_labelledby" ; aria-labelledby resolves to at least one non-empty element
        "title_attr"      ; title attribute present and non-empty
        "presentation"    ; role="none" or role="presentation" (decorative opt-out)
    """
    all_ids = {tag["id"] for tag in soup.find_all(id=True)}
    failures = []

    for mech in mechanisms:
        if mech == "text_content":
            if visible_text(el).strip():
                return True, []
            failures.append("no text content")

        elif mech == "alt":
            alt = el.get("alt")
            if alt is not None and alt.strip():
                return True, []
            failures.append(
                "alt attribute missing"
                if alt is None
                else "alt attribute is present but empty"
            )

        elif mech == "svg_title":
            t = el.find("title")
            if t and visible_text(t).strip():
                return True, []
            failures.append(
                "no <title> child element"
                if not t
                else "<title> child element is empty"
            )

        elif mech == "wrapped_label":
            if el.find_parent("label"):
                return True, []
            failures.append("not wrapped in a <label> element")

        elif mech == "explicit_label":
            el_id = el.get("id")
            if el_id and soup.find("label", attrs={"for": el_id}):
                return True, []
            failures.append(
                "element has no id, so <label for=...> cannot reference it"
                if not el_id
                else f'no <label for="{el_id}"> found in document'
            )

        elif mech == "aria_label":
            val = el.get("aria-label")
            if val and val.strip():
                return True, []
            failures.append(
                "aria-label attribute absent"
                if val is None
                else "aria-label attribute is empty"
            )

        elif mech == "aria_labelledby":
            val = el.get("aria-labelledby", "").strip()
            if not val:
                failures.append("aria-labelledby attribute absent")
            else:
                refs    = val.split()
                missing = [r for r in refs if r not in all_ids]
                empty   = [r for r in refs
                           if r in all_ids
                           and not visible_text(soup.find(id=r)).strip()]
                valid   = [r for r in refs if r not in missing and r not in empty]
                if valid:
                    return True, []
                parts = []
                if missing:
                    parts.append(f"references non-existent id(s): {', '.join(missing)}")
                if empty:
                    parts.append(f"references empty element(s): {', '.join(empty)}")
                failures.append("aria-labelledby present but " + "; ".join(parts))

        elif mech == "title_attr":
            val = el.get("title")
            if val and val.strip():
                return True, []
            failures.append(
                "title attribute absent"
                if val is None
                else "title attribute is empty"
            )

        elif mech == "presentation":
            role = el.get("role", "")
            if role in {"none", "presentation"}:
                return True, []
            failures.append(
                'default semantics not overridden with role="none" or role="presentation" '
                "(required if element is purely decorative)"
            )

    return False, failures


def _fmt_failures(failures: list[str]) -> str:
    """format name-mechanism failures as indented bullet list"""
    return "\n\t· ".join(failures)


def _url_label(url: str) -> str:
    """short path label used as matrix column header"""
    path = urlparse(url).path.rstrip("/")
    return path or "/"


def _sc_codes(criterion: str) -> list[str]:
    return re.findall(r"\d+\.\d+\.\d+", criterion)


def _criterion_status(result: Optional[AuditResult], sc: str) -> str:
    """return E/W/P/I/— for a given success criterion code from an AuditResult"""
    if result is None:
        return "—"
    for issue in result.issues:
        if issue.level == "error" and sc in _sc_codes(issue.criterion):
            return "E"
    for issue in result.issues:
        if issue.level == "warning" and sc in _sc_codes(issue.criterion):
            return "W"
    for p in result.passes:
        if sc in _sc_codes(p.criterion):
            return "P"
    return "I"


# individual checks
def check_page_title(soup, result):
    title = soup.find("title")
    if not title:
        result.add("error", "2.4.2", "page title",
                   "<title> element is absent from <head>; "
                   "every page must have a descriptive title")
    elif not title.get_text(strip=True):
        result.add("error", "2.4.2", "page title",
                   f"<title> element is present but empty; "
                   "title must contain meaningful text describing the page",
                   snippet(title))


def check_language(soup, result):
    html = soup.find("html")
    if not html:
        result.add("error", "3.1.1", "language of page",
                   "<html> element not found; lang attribute cannot be set")
        return
    lang = html.get("lang")
    if lang is None:
        result.add("error", "3.1.1", "language of page",
                   "lang attribute is absent from <html>; "
                   "add lang=\"en\" (or the appropriate BCP 47 code) so AT can select the correct voice")
    elif not lang.strip():
        result.add("error", "3.1.1", "language of page",
                   "lang attribute is present on <html> but is empty; "
                   "provide a valid BCP 47 language code, e.g. lang=\"en\"")


def check_images(soup, result):
    for img in soup.find_all("img"):
        if img.get("role") in {"presentation", "none"}:
            continue
        if img.get("aria-hidden") == "true":
            continue
        alt = img.get("alt")
        if alt is None or (alt.strip() == "" and not has_aria_label(img)):
            # image has no accessible name via any mechanism
            _, failures = _name_check(img, soup,
                ["alt", "aria_label", "aria_labelledby", "title_attr", "presentation"])
            result.add("error", "1.1.1", "non-text content",
                       "image has no accessible name:\n\t· " + _fmt_failures(failures),
                       snippet(img))
        elif alt is not None and alt.strip().lower() in {
            "image", "photo", "picture", "graphic", "icon"
        }:
            result.add("warning", "1.1.1", "non-text content",
                       f'alt text "{alt}" is likely non-descriptive; '
                       "describe what the image conveys, not just its type",
                       snippet(img))


def check_form_labels(soup, result):
    inputs = soup.find_all(["input", "select", "textarea"])
    for inp in inputs:
        itype = inp.get("type", "text").lower()
        if itype in {"hidden", "submit", "reset", "button", "image"}:
            continue
        found, failures = _name_check(inp, soup,
            ["wrapped_label", "explicit_label", "aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("error", "1.3.1, 4.1.2", "form label missing",
                       f'<{inp.name} type="{itype}"> has no accessible label:\n\t· '
                       + _fmt_failures(failures),
                       snippet(inp))


def check_buttons(soup, result):
    for btn in soup.find_all("button"):
        found, failures = _name_check(btn, soup,
            ["text_content", "aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("error", "4.1.2", "button name",
                       "button has no accessible name:\n\t· " + _fmt_failures(failures),
                       snippet(btn))


def check_links(soup, result):
    for a in soup.find_all("a", href=True):
        found, failures = _name_check(a, soup,
            ["text_content", "aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("error", "2.4.4", "link purpose",
                       "link has no accessible name:\n\t· " + _fmt_failures(failures),
                       snippet(a))
        else:
            # accessible name exists ; check if it's non-descriptive
            name = (a.get("aria-label") or a.get("title") or visible_text(a) or "").strip()
            if name.lower() in {"click here", "here", "read more", "more", "link", "learn more"}:
                result.add("warning", "2.4.4", "link purpose",
                           f'link text "{name}" is non-descriptive out of context',
                           snippet(a))


def check_headings(soup, result):
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        result.add("warning", "1.3.1", "info and relationships",
                   "no heading elements found; "
                   "use <h1>-<h6> to communicate page structure semantically")
        return

    h1s = soup.find_all("h1")
    if len(h1s) == 0:
        result.add("error", "1.3.1", "heading structure",
                   "no <h1> element found; every page should have exactly one <h1> "
                   "identifying its primary topic")
    elif len(h1s) > 1:
        h1_texts = [f'"{visible_text(h)[:50] or "(empty)"}"' for h in h1s]
        result.add("warning", "1.3.1", "heading structure",
                   f"{len(h1s)} <h1> elements found; only one is appropriate per page\n"
                   "\t· h1 texts: " + ", ".join(h1_texts))

    levels = [int(h.name[1]) for h in headings]
    for i in range(1, len(levels)):
        skip = levels[i] - levels[i - 1]
        if skip > 1:
            prev_text = visible_text(headings[i - 1])[:50] or "(empty)"
            curr_text = visible_text(headings[i])[:50] or "(empty)"
            skipped   = ", ".join(f"h{l}" for l in range(levels[i - 1] + 1, levels[i]))
            result.add("warning", "1.3.1", "heading hierarchy",
                       f"heading level skips from h{levels[i-1]} to h{levels[i]} "
                       f"(skipped: {skipped})\n"
                       f"\t· h{levels[i-1]}: \"{prev_text}\"\n"
                       f"\t· h{levels[i]}: \"{curr_text}\"",
                       snippet(headings[i]))


def check_iframes(soup, result):
    for frame in soup.find_all("iframe"):
        found, failures = _name_check(frame, soup,
            ["title_attr", "aria_label", "aria_labelledby"])
        if not found:
            result.add("error", "4.1.2", "iframe title",
                       "iframe has no accessible name:\n\t· " + _fmt_failures(failures),
                       snippet(frame))


def check_skip_link(soup, result):
    links = soup.find_all("a", href=True)
    first_few = links[:5]
    has_skip = any(
        "#" in a.get("href", "") and
        any(kw in visible_text(a).lower() for kw in ("skip", "jump", "main content"))
        for a in first_few
    )
    if not has_skip:
        found_labels = [
            f'"{visible_text(a) or a.get("aria-label", "") or "(no text)"}" -> {a.get("href", "")}'
            for a in first_few
        ]
        detail = (
            "no skip-navigation link detected in the first 5 links of the page; "
            "a skip link must appear before repeated navigation and anchor to the main content region\n"
            "\tfirst links found:\n\t· " + "\n\t· ".join(found_labels)
            if found_labels else
            "no skip-navigation link detected; page has no links at all near the top"
        )
        result.add("warning", "2.4.1", "bypass blocks", detail)


def check_tables(soup, result):
    for table in soup.find_all("table"):
        headers   = table.find_all("th")
        caption   = table.find("caption")
        aria_lbl  = has_aria_label(table)
        problems  = []

        if not headers:
            problems.append(
                "no <th> elements ; data relationships are not programmatically determinable; "
                "add <th scope=\"col\"> for column headers and <th scope=\"row\"> for row headers"
            )
        else:
            scope_missing = [th for th in headers if not th.get("scope") and not th.get("id")]
            if scope_missing:
                problems.append(
                    f"{len(scope_missing)} <th> element(s) missing scope attribute "
                    "(use scope=\"col\", \"row\", \"colgroup\", or \"rowgroup\")"
                )

        if not caption and not aria_lbl:
            problems.append(
                "no <caption> element and no aria-label/aria-labelledby ; "
                "table has no accessible name to distinguish it from other tables"
            )

        if problems:
            result.add("warning", "1.3.1", "table structure",
                       "table has structural accessibility issues:\n\t· "
                       + "\n\t· ".join(problems),
                       snippet(table)[:80])


def check_aria_roles(soup, result):
    landmarks = {
        "main": soup.find("main") or soup.find(attrs={"role": "main"}),
        "navigation": soup.find("nav") or soup.find(attrs={"role": "navigation"}),
    }
    if not landmarks["main"]:
        result.add("warning", "1.3.1, 4.1.2", "landmark: main",
                   "no <main> or role='main' landmark found")
    if not landmarks["navigation"]:
        result.add("info", "1.3.1", "landmark: navigation",
                   "no <nav> element found; may be fine for relatively simple pages")

    valid_roles = {
        "alert","alertdialog","application","article","banner","button","cell",
        "checkbox","columnheader","combobox","complementary","contentinfo",
        "definition","dialog","directory","document","feed","figure","form",
        "grid","gridcell","group","heading","img","link","list","listbox",
        "listitem","log","main","marquee","math","menu","menubar","menuitem",
        "menuitemcheckbox","menuitemradio","navigation","none","note","option",
        "presentation","progressbar","radio","radiogroup","region","row",
        "rowgroup","rowheader","scrollbar","search","searchbox","separator",
        "slider","spinbutton","status","switch","tab","table","tablist",
        "tabpanel","term","textbox","timer","toolbar","tooltip","tree",
        "treegrid","treeitem",
    }
    for el in soup.find_all(attrs={"role": True}):
        for role in el["role"].split():
            if role not in valid_roles:
                result.add("error", "4.1.2", "invalid aria role",
                           f'role="{role}" is not a valid aria role',
                           snippet(el))


def check_viewport(soup, result):
    meta = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)})
    if not meta:
        return
    content = meta.get("content", "")
    if re.search(r"user-scalable\s*=\s*no", content, re.I):
        result.add("error", "1.4.4", "viewport scalability",
                   "meta viewport sets user-scalable=no, preventing user text resize",
                   snippet(meta))
    m = re.search(r"maximum-scale\s*=\s*([\d.]+)", content, re.I)
    if m and float(m.group(1)) < 2:
        result.add("error", "1.4.4", "viewport scalability",
                   f"meta viewport sets maximum-scale={m.group(1)}; must be >= 2 to allow text resize",
                   snippet(meta))


def check_duplicate_ids(soup, result):
    from collections import Counter
    ids = [tag["id"] for tag in soup.find_all(id=True)]
    counts = Counter(ids)
    for id_, count in counts.items():
        if count < 2:
            continue
        owners = soup.find_all(id=id_)
        tags   = ", ".join(f"<{el.name}>" for el in owners)
        result.add("error", "4.1.1", "duplicate id",
                   f'id="{id_}" appears {count} times ({tags}); '
                   "id values must be unique ; aria-labelledby, aria-describedby, "
                   "<label for=...>, and fragment links targeting this id will resolve unpredictably")


def check_dangling_aria_refs(soup, result):
    all_ids = {tag["id"] for tag in soup.find_all(id=True)}
    attrs = ["aria-labelledby", "aria-describedby", "aria-controls", "aria-owns"]
    for el in soup.find_all(True):
        for attr in attrs:
            val = el.get(attr, "").strip()
            if not val:
                continue
            for ref_id in val.split():
                if ref_id not in all_ids:
                    result.add("error", "4.1.2", "dangling aria reference",
                               f'{attr}="{ref_id}" references an id that does not exist in dom',
                               snippet(el))


def check_positive_tabindex(soup, result):
    positive = []
    for el in soup.find_all(tabindex=True):
        try:
            val = int(el["tabindex"])
        except ValueError:
            continue
        if val > 0:
            positive.append((val, el))

    if not positive:
        return

    positive.sort(key=lambda x: x[0])
    for val, el in positive:
        name = (
            visible_text(el)[:40]
            or el.get("aria-label", "")[:40]
            or el.get("value", "")[:40]
            or "(no accessible name)"
        )
        result.add("warning", "2.4.3", "positive tabindex",
                   f"tabindex={val} on <{el.name}> (\"{name}\") "
                   f"creates an explicit focus sequence that overrides document order; "
                   f"all tabindex={val} elements are visited before tabindex=0 elements, "
                   f"which typically breaks the visual reading order ; use tabindex=\"0\" "
                   f"or tabindex=\"-1\" only",
                   snippet(el))


def check_aria_hidden_focusable(soup, result):
    for el in soup.find_all(attrs={"aria-hidden": "true"}):
        self_focusable = _is_in_tab_order(el)
        focusable_child = el.find(lambda t: isinstance(t, Tag) and _is_in_tab_order(t))

        if not self_focusable and not focusable_child:
            continue

        if self_focusable:
            reason = (
                f"the element itself (<{el.name}>"
                + (f' type="{el.get("type")}"' if el.get("type") else "")
                + ") is focusable via the natural tab order"
            )
        else:
            reason = (
                f"a focusable descendant exists: <{focusable_child.name}>"
                + (f' type="{focusable_child.get("type")}"' if focusable_child.get("type") else "")
                + (f' id="{focusable_child.get("id")}"' if focusable_child.get("id") else "")
                + " remains in the tab order despite the container being hidden from AT"
            )

        result.add("error", "4.1.2", "aria-hidden on focusable element",
                   'aria-hidden="true" hides this element from assistive technology '
                   "but does not remove it from keyboard focus:\n"
                   f"\t· {reason}\n"
                   "\t· keyboard users can reach it; screen reader users will get no announcement\n"
                   "\t· fix: move aria-hidden to a purely decorative wrapper, "
                   "or add tabindex=\"-1\" to the focusable element(s) inside",
                   snippet(el))


def check_video_captions(soup, result):
    for video in soup.find_all("video"):
        tracks = video.find_all("track")
        present_kinds = [t.get("kind", "").lower() for t in tracks if t.get("kind")]
        has_captions  = any(k in {"captions", "subtitles"} for k in present_kinds)
        if not has_captions:
            detail = (
                'required: <track kind="captions"> or <track kind="subtitles">; '
                + (
                    f"tracks present with kind(s): {', '.join(present_kinds)} ; "
                    "none of these satisfy the captions requirement"
                    if present_kinds
                    else "no <track> elements found at all"
                )
            )
            result.add("error", "1.2.2", "video captions", detail, snippet(video))


def check_video_description(soup, result):
    """1.2.5 prerecorded video needs an audio description track"""
    for video in soup.find_all("video"):
        tracks        = video.find_all("track")
        present_kinds = [t.get("kind", "").lower() for t in tracks if t.get("kind")]
        has_description = "descriptions" in present_kinds
        if not has_description:
            detail = (
                'required: <track kind="descriptions"> for prerecorded video with meaningful visual content; '
                + (
                    f"tracks present with kind(s): {', '.join(present_kinds)} ; "
                    "none provide audio description"
                    if present_kinds
                    else "no <track> elements found at all"
                )
            )
            result.add("warning", "1.2.5", "audio description", detail, snippet(video))


def check_audio_description_alternative(soup, result):
    """1.2.3 prerecorded synchronized media needs audio description or full text alternative"""
    _TEXT_ALT_KEYWORDS = ("transcript", "text alternative", "audio description", "described")
    for video in soup.find_all("video"):
        tracks          = video.find_all("track")
        present_kinds   = [t.get("kind", "").lower() for t in tracks if t.get("kind")]
        has_description = "descriptions" in present_kinds
        parent_text     = visible_text(video.parent).lower() if video.parent else ""
        matched_kw      = next((kw for kw in _TEXT_ALT_KEYWORDS if kw in parent_text), None)

        if not has_description and not matched_kw:
            track_note = (
                f"tracks present: {', '.join(present_kinds)} ; none are descriptions"
                if present_kinds else "no <track> elements present"
            )
            result.add("warning", "1.2.3", "audio description or media alternative",
                       "prerecorded synchronized media has no description track "
                       "and no adjacent text alternative:\n"
                       f"\t· {track_note}\n"
                       f"\t· adjacent text checked for keywords: "
                       f"{', '.join(_TEXT_ALT_KEYWORDS)} ; none found\n"
                       "\t· add <track kind=\"descriptions\"> or a full text transcript nearby",
                       snippet(video))


def check_svg_accessibility(soup, result):
    for svg in soup.find_all("svg"):
        role = svg.get("role", "")
        if role in {"presentation", "none"}:
            continue
        if svg.get("aria-hidden") == "true":
            continue
        found, failures = _name_check(svg, soup,
            ["svg_title", "aria_label", "aria_labelledby", "presentation"])
        if not found:
            result.add("warning", "1.1.1", "svg accessible name",
                       "inline <svg> has no accessible name:\n\t· " + _fmt_failures(failures),
                       snippet(svg))


def check_meta_refresh(soup, result):
    for meta in soup.find_all("meta", attrs={"http-equiv": re.compile(r"^refresh$", re.I)}):
        content = meta.get("content", "")
        m = re.match(r"(\d+)", content.strip())
        if m:
            delay = int(m.group(1))
            if delay == 0:
                result.add("warning", "2.2.1", "meta refresh (redirect)",
                           "instant meta-refresh redirect; ensure destination page is accessible",
                           snippet(meta))
            else:
                result.add("error", "2.2.1", "meta refresh (auto-reload)",
                           f"page auto-refreshes after {delay}s; users cannot pause, stop, or extend this",
                           snippet(meta))


_AUTOCOMPLETE_HINTS = {
    "name": "name", "fname": "given-name", "firstname": "given-name",
    "lname": "family-name", "lastname": "family-name", "email": "email",
    "phone": "tel", "telephone": "tel", "mobile": "tel",
    "address": "street-address", "city": "address-level2",
    "zip": "postal-code", "postcode": "postal-code",
    "country": "country", "cardnumber": "cc-number", "cc-number": "cc-number",
    "bday": "bday", "birthday": "bday",
}

def check_autocomplete(soup, result):
    for inp in soup.find_all("input"):
        itype = inp.get("type", "text").lower()
        if itype in {"hidden", "submit", "reset", "button", "checkbox", "radio"}:
            continue
        if inp.get("autocomplete"):
            continue
        candidate_attrs = {
            "id":          inp.get("id", ""),
            "name":        inp.get("name", ""),
            "placeholder": inp.get("placeholder", ""),
            "aria-label":  inp.get("aria-label", ""),
        }
        for keyword, purpose in _AUTOCOMPLETE_HINTS.items():
            matched_attr = next(
                (attr for attr, val in candidate_attrs.items() if keyword in val.lower()), None
            )
            if matched_attr:
                result.add("warning", "1.3.5", "autocomplete purpose",
                           f'input appears to collect "{keyword}" data '
                           f'(matched in {matched_attr}="{candidate_attrs[matched_attr]}") '
                           f'but has no autocomplete attribute; '
                           f'add autocomplete="{purpose}" so browsers and AT can assist users',
                           snippet(inp))
                break


def check_empty_labels(soup, result):
    for label in soup.find_all("label"):
        text = "".join(
            t for t in label.strings
            if t.strip() and t.parent.name not in {"input", "select", "textarea"}
        ).strip()
        if not text and not label.get("aria-label"):
            for_id = label.get("for")
            if for_id:
                target = soup.find(id=for_id)
                association = (
                    f'associated with element id="{for_id}" '
                    + (f"(<{target.name}>)" if target else "(id not found in DOM ; dangling reference)")
                )
            else:
                wrapped = label.find(["input", "select", "textarea"])
                association = (
                    f"implicitly wraps <{wrapped.name} type=\"{wrapped.get('type','text')}\">"
                    if wrapped else "no for attribute and no wrapped control found"
                )
            result.add("warning", "1.3.1, 4.1.2", "empty label",
                       f"<label> has no visible text and no aria-label; "
                       f"the associated control will have no accessible name\n"
                       f"\t· {association}",
                       snippet(label))


def check_moving_content(soup, result):
    for tag in soup.find_all(["marquee", "blink"]):
        result.add("error", "2.2.2", "moving, blinking content",
                   f"<{tag.name}> creates uncontrollable moving content; remove it",
                   snippet(tag))


_INTERACTIVE_ROLES = {
    "button", "link", "checkbox", "radio", "textbox", "combobox",
    "listbox", "menuitem", "menuitemcheckbox", "menuitemradio",
    "option", "switch", "tab",
}

def check_label_in_name(soup, result):
    """2.5.3 visible text must appear as substring of aria-label on interactive elements"""
    for el in soup.find_all(True):
        is_interactive = (
            el.name in {"a", "button", "input", "select", "textarea"} or
            el.get("role", "") in _INTERACTIVE_ROLES
        )
        if not is_interactive:
            continue
        aria_label = el.get("aria-label", "").strip()
        if not aria_label:
            continue
        text = visible_text(el).strip()
        if not text or len(text) > 80:
            continue
        if text.lower() not in aria_label.lower():
            result.add("warning", "2.5.3", "label in name",
                       f'visible text "{text}" is absent from aria-label "{aria_label}"; '
                       "speech-input users activating by visible label will fail",
                       snippet(el))


_STATUS_CLASS_RE = re.compile(
    r"\b(alert|toast|notification|snackbar|banner|message|status|error|success|warning|info)\b",
    re.I,
)

def check_status_messages(soup, result):
    """4.1.3 notification-like containers need role=status/alert or aria-live"""
    live_roles  = {"alert", "status", "log", "marquee", "timer"}
    live_values = {"polite", "assertive", "off"}
    for el in soup.find_all(True):
        cls    = " ".join(el.get("class", []))
        el_id  = el.get("id", "")
        if not _STATUS_CLASS_RE.search(cls + " " + el_id):
            continue
        role      = el.get("role", "")
        aria_live = el.get("aria-live", "")
        if role in live_roles or aria_live:
            continue

        matched = _STATUS_CLASS_RE.search(cls + " " + el_id).group(0)
        problems = []
        if not role:
            problems.append(
                f'role attribute absent; for notifications add role="status" (polite) '
                f'or role="alert" (assertive) ; valid live roles: {", ".join(sorted(live_roles))}'
            )
        else:
            problems.append(
                f'role="{role}" is not a live region role; '
                f"valid live roles: {', '.join(sorted(live_roles))}"
            )
        if not aria_live:
            problems.append(
                'aria-live attribute absent; alternatively add aria-live="polite" or aria-live="assertive"'
            )
        result.add("warning", "4.1.3", "status message",
                   f'element matched notification pattern "{matched}" '
                   f'(class/id: "{(cls or el_id).strip()}") '
                   "but dynamically injected content won't be announced to screen readers:\n"
                   "\t· " + "\n\t· ".join(problems),
                   snippet(el))


def check_language_parts(soup, result):
    """3.1.2 inline lang overrides must be valid BCP 47"""
    _COMMON_MISTAKES = {
        "english": "en", "french": "fr", "german": "de", "spanish": "es",
        "italian": "it", "portuguese": "pt", "dutch": "nl", "japanese": "ja",
        "chinese": "zh", "korean": "ko", "arabic": "ar", "russian": "ru",
        "en-uk": "en-GB",
    }
    html_el = soup.find("html")
    for el in soup.find_all(attrs={"lang": True}):
        if el == html_el:
            continue
        lang = el.get("lang", "").strip()
        if not re.match(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$", lang):
            suggestion = _COMMON_MISTAKES.get(lang.lower())
            detail = (
                f'lang="{lang}" on <{el.name}> is not a valid BCP 47 tag'
                + (f'; did you mean lang="{suggestion}"?' if suggestion else
                   "; use a 2-3 letter primary subtag, e.g. lang=\"fr\" for French")
            )
            result.add("warning", "3.1.2", "language of parts", detail, snippet(el))


def check_audit_limitations(soup, result):
    """detect page characteristics that degrade reliability; reports as info"""
    html_text = str(soup)

    frameworks = []
    if soup.find(id="root") or soup.find(id="app") or soup.find(attrs={"data-reactroot": True}):
        frameworks.append("react (generic root div)")
    if "__NEXT_DATA__" in html_text:
        frameworks.append("next.js")
    if "gatsby" in html_text.lower() or soup.find(id="___gatsby"):
        frameworks.append("gatsby")
    if "remix" in html_text.lower() and "__remixContext" in html_text:
        frameworks.append("remix")
    ng = soup.find(attrs={"ng-version": True}) or soup.find("app-root")
    if ng or re.search(r"ng-version", html_text):
        frameworks.append("angular")
    if soup.find(id="__nuxt") or "__NUXT__" in html_text:
        frameworks.append("nuxt.js")
    elif soup.find(attrs={"data-v-app": True}) or "__vue_app__" in html_text:
        frameworks.append("vue")
    if re.search(r"__sveltekit|sveltekit:data", html_text, re.I):
        frameworks.append("sveltekit")
    elif soup.find(attrs={"data-svelte": True}):
        frameworks.append("svelte")

    if frameworks:
        result.add("info", "—", "js framework detected",
                   f"page uses {', '.join(frameworks)}; body content is likely js-rendered; "
                   "most audit checks ran against a partially-populated dom")

    body = soup.find("body")
    body_text_len = len(body.get_text(strip=True)) if body else 0
    script_tags = len(soup.find_all("script", src=True))
    if body_text_len < 200 and script_tags >= 3 and not frameworks:
        result.add("info", "—", "likely js-rendered content",
                   f"body text is sparse ({body_text_len} chars) with {script_tags} external scripts; "
                   "dom may be populated after page load; audit coverage is probably incomplete!")

    stylesheets = soup.find_all("link", rel=lambda r: r and "stylesheet" in r)
    result.externals["stylesheets"] = [s.get("href", "(no href)") for s in stylesheets]
    if stylesheets:
        result.add("info", "1.4.3", "contrast check incomplete",
                   f"{len(stylesheets)} external stylesheet(s) found; color contrast cannot be "
                   "evaluated without a rendering engine; use axe-core or browser devtools")

    result.externals["scripts"] = [s.get("src") for s in soup.find_all("script", src=True)]

    iframes = soup.find_all("iframe")
    result.externals["iframes"] = [f.get("src", "(no src)") for f in iframes]
    if iframes:
        srcs = result.externals["iframes"][:3]
        result.add("info", "—", "iframe content not audited",
                   f"{len(iframes)} iframe(s) found; subtrees are not fetched or checked; "
                   f"sources: {', '.join(srcs)}")

    result.externals["fonts"] = [
        s.get("href", "(no href)")
        for s in soup.find_all("link", rel=lambda r: r and "preload" in r)
        if s.get("as") == "font"
    ] + [
        s.get("href", "(no href)")
        for s in soup.find_all("link", rel=lambda r: r and "stylesheet" in r)
        if s.get("href") and ("fonts.googleapis" in s.get("href", "") or
                               "typekit" in s.get("href", "") or
                               "fonts.bunny" in s.get("href", ""))
    ]

    media_srcs = []
    for el in soup.find_all(["video", "audio"]):
        src = el.get("src")
        if src:
            media_srcs.append(src)
        for source in el.find_all("source"):
            s = source.get("src")
            if s:
                media_srcs.append(s)
    result.externals["media"] = media_srcs

    if soup.find("noscript"):
        result.add("info", "—", "js dependency confirmed",
                   "<noscript> element present; interactive states are not audited here")

    result.add("info", "2.1.1, 2.1.2, 2.4.3, 2.4.7",
               "keyboard and focus not tested",
               "focus order, keyboard operability, focus visibility, and focus traps require "
               "interaction simulation (e.g., playwright + axe-core); not covered here")


def check_autoplay(soup, result):
    for media in soup.find_all(["video", "audio"]):
        if media.get("autoplay") is None or media.get("muted") is not None:
            continue
        has_controls = media.get("controls") is not None
        problems = [
            "muted attribute absent ; audio will play automatically on page load"
        ]
        if not has_controls:
            problems.append(
                "controls attribute absent ; user has no pause/stop mechanism in the player itself; "
                "a mechanism to pause or stop audio is required within 3 seconds"
            )
        result.add("error", "1.4.2", "audio control",
                   f"<{media.name}> autoplays with audio:\n\t· "
                   + "\n\t· ".join(problems),
                   snippet(media))


def check_color_contrast_hints(soup, result):
    for el in soup.find_all(style=True):
        style = el.get("style", "")
        color = parse_inline_color(style, "color")
        bg    = parse_inline_color(style, "background-color")
        if color and bg:
            result.add("info", "1.4.3", "contrast (inline styles only)",
                       f"element has inline color:{color} + background:{bg}; "
                       "verify contrast ratio >= 4.5:1 (normal text), 3:1 (large text)",
                       snippet(el))


def check_target_size(soup, result):
    for el in soup.find_all(["a", "button"]):
        style = el.get("style", "")
        w = parse_inline_color(style, "width")
        h = parse_inline_color(style, "height")
        if w and h:
            try:
                wv = float(re.sub(r"[^\d.]", "", w))
                hv = float(re.sub(r"[^\d.]", "", h))
                if wv < 24 or hv < 24:
                    result.add("warning", "2.5.8", "target size",
                               f"interactive element appears small ({w} x {h}); "
                               "minimum 24x24 css px required (44x44 recommended)",
                               snippet(el))
            except ValueError:
                pass


def check_image_buttons(soup, result):
    """1.1.1 <input type="image"> needs alt text"""
    for inp in soup.find_all("input", attrs={"type": re.compile(r"^image$", re.I)}):
        found, failures = _name_check(inp, soup,
            ["alt", "aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("error", "1.1.1", "image button alt text",
                       '<input type="image"> has no accessible name:\n\t· '
                       + _fmt_failures(failures),
                       snippet(inp))


def check_audio(soup, result):
    """1.2.1 prerecorded audio-only content needs a text transcript"""
    for audio in soup.find_all("audio"):
        if audio.get("autoplay") is not None and audio.get("muted") is not None:
            continue
        tracks        = audio.find_all("track")
        present_kinds = [t.get("kind", "").lower() for t in tracks if t.get("kind")]
        has_transcript = any(k in {"captions", "descriptions", "subtitles"} for k in present_kinds)
        parent_text    = visible_text(audio.parent).lower() if audio.parent else ""
        has_transcript_link = "transcript" in parent_text

        if not has_transcript and not has_transcript_link:
            track_note = (
                f"tracks present with kind(s): {', '.join(present_kinds)} ; "
                "none satisfy the transcript requirement"
                if present_kinds
                else "no <track> elements found"
            )
            result.add("warning", "1.2.1", "audio transcript",
                       "prerecorded audio-only content requires a text alternative:\n"
                       f"\t· {track_note}\n"
                       '\t· no adjacent text containing "transcript" found\n'
                       '\t· add <track kind="captions"> or link to a text transcript nearby',
                       snippet(audio))


def check_image_maps(soup, result):
    """1.1.1 <area> elements in image maps need alt text"""
    for area in soup.find_all("area"):
        if area.get("href") is None and area.get("role") in {"presentation", "none"}:
            continue
        found, failures = _name_check(area, soup,
            ["alt", "aria_label", "aria_labelledby"])
        if not found:
            result.add("error", "1.1.1", "image map area alt",
                       "<area> in image map has no accessible name:\n\t· "
                       + _fmt_failures(failures),
                       snippet(area))


def check_role_img(soup, result):
    """1.1.1 elements with role='img' need an accessible name"""
    for el in soup.find_all(attrs={"role": True}):
        if "img" not in el["role"].split():
            continue
        if el.get("aria-hidden") == "true":
            continue
        found, failures = _name_check(el, soup,
            ["aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("error", "1.1.1", "role=img without name",
                       f'<{el.name} role="img"> has no accessible name:\n\t· '
                       + _fmt_failures(failures),
                       snippet(el))


def check_object_fallback(soup, result):
    """1.1.1 <object> elements need text alternative or title"""
    for obj in soup.find_all("object"):
        found, failures = _name_check(obj, soup,
            ["text_content", "aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("warning", "1.1.1", "object fallback",
                       "<object> has no accessible name or fallback text:\n\t· "
                       + _fmt_failures(failures),
                       snippet(obj))


def check_fieldsets(soup, result):
    """1.3.1 <fieldset> should have a <legend>"""
    for fs in soup.find_all("fieldset"):
        legend = fs.find("legend")
        if not legend:
            controls = fs.find_all(["input", "select", "textarea"])
            ctrl_summary = (
                f"{len(controls)} control(s) inside: "
                + ", ".join(
                    f'<{c.name} type="{c.get("type","text")}">'
                    if c.name == "input" else f"<{c.name}>"
                    for c in controls[:4]
                )
                + (" ..." if len(controls) > 4 else "")
            ) if controls else "no controls found inside"
            result.add("warning", "1.3.1", "fieldset without legend",
                       f"<fieldset> has no <legend> element; "
                       f"grouped controls will have no announced group label\n"
                       f"\t· {ctrl_summary}",
                       snippet(fs))
        elif not visible_text(legend).strip():
            result.add("warning", "1.3.1", "fieldset empty legend",
                       "<fieldset> has a <legend> element but it contains no visible text; "
                       "the group label will be announced as empty",
                       snippet(legend))


def check_details_summary(soup, result):
    """4.1.2 <details> needs a non-empty <summary>"""
    for details in soup.find_all("details"):
        summary = details.find("summary")
        if not summary:
            result.add("warning", "4.1.2", "details without summary",
                       "<details> has no <summary> element; "
                       "browsers insert a default 'details' label which is non-descriptive",
                       snippet(details))
        else:
            found, failures = _name_check(summary, soup,
                ["text_content", "aria_label", "aria_labelledby"])
            if not found:
                result.add("warning", "4.1.2", "details empty summary",
                           "<summary> has no accessible name:\n\t· "
                           + _fmt_failures(failures),
                           snippet(summary))


def check_progress_meter(soup, result):
    """4.1.2 <progress> and <meter> need accessible names"""
    for el in soup.find_all(["progress", "meter"]):
        found, failures = _name_check(el, soup,
            ["wrapped_label", "explicit_label", "aria_label", "aria_labelledby", "title_attr"])
        if not found:
            result.add("warning", "4.1.2", f"{el.name} accessible name",
                       f"<{el.name}> has no accessible name:\n\t· "
                       + _fmt_failures(failures),
                       snippet(el))


def check_language_validity(soup, result):
    """3.1.1 verify lang attribute is plausible BCP 47"""
    _COMMON_MISTAKES = {
        "english": "en", "french": "fr", "german": "de", "spanish": "es",
        "italian": "it", "portuguese": "pt", "dutch": "nl", "japanese": "ja",
        "chinese": "zh", "korean": "ko", "arabic": "ar", "russian": "ru",
        "en-uk": "en-GB", "en-us": "en-US",
    }
    html = soup.find("html")
    if not html:
        return
    lang = html.get("lang", "").strip()
    if not lang:
        return
    if not re.match(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$", lang):
        suggestion = _COMMON_MISTAKES.get(lang.lower())
        detail = (
            f'lang="{lang}" is not a valid BCP 47 language tag'
            + (f'; did you mean lang="{suggestion}"?' if suggestion else
               "; BCP 47 primary subtags are 2-3 letters (e.g. \"en\", \"fr\", \"zh\"); "
               "region subtags follow a hyphen (e.g. \"en-GB\", \"zh-Hant\")")
        )
        result.add("warning", "3.1.1", "language tag format", detail, snippet(html))


def check_duplicate_adjacent_links(soup, result):
    """2.4.4 adjacent same-href links create redundant tab stops"""
    links = soup.find_all("a", href=True)
    for i in range(len(links) - 1):
        a, b = links[i], links[i + 1]
        if a.parent != b.parent:
            continue
        href_a = a.get("href", "").strip()
        href_b = b.get("href", "").strip()
        if not href_a or href_a != href_b:
            continue
        name_a = (a.get("aria-label") or visible_text(a) or "(no accessible name)").strip()
        name_b = (b.get("aria-label") or visible_text(b) or "(no accessible name)").strip()
        if name_a or name_b:
            result.add("warning", "2.4.4", "adjacent duplicate links",
                       f"two adjacent sibling links point to the same destination, "
                       f"creating a redundant tab stop:\n"
                       f"\t· link 1: \"{name_a[:60]}\"\n"
                       f"\t· link 2: \"{name_b[:60]}\"\n"
                       f"\t· href: {href_a}\n"
                       f"\t· consider wrapping the image and text in a single <a>",
                       snippet(a))


def check_multiple_ways(soup, result):
    """2.4.5 more than one way to locate a page within the site"""
    found    = []
    missing  = []

    if (soup.find("input", attrs={"type": re.compile(r"^search$", re.I)}) or
            soup.find(attrs={"role": "search"})):
        found.append("search field (input[type=search] or role=search)")
    else:
        missing.append("search field ; no input[type=search] or role=search found")

    if any("sitemap" in (a.get("href", "") + " " + visible_text(a)).lower()
           for a in soup.find_all("a", href=True)):
        found.append("sitemap link")
    else:
        missing.append("sitemap link ; no link containing \"sitemap\" in href or text found")

    if soup.find("nav") or soup.find(attrs={"role": "navigation"}):
        found.append("navigation landmark (<nav> or role=navigation)")
    else:
        missing.append("navigation landmark ; no <nav> or role=navigation found")

    if len(found) >= 2:
        return  # two or more mechanisms present ; passes

    detail = (
        "fewer than two page-location mechanisms found; "
        "multi-page sites require at least two ways to locate any page\n"
        + (("\t· present: " + "; ".join(found) + "\n") if found else "")
        + "\t· missing: " + "\n\t· ".join(missing)
    )
    result.add("warning", "2.4.5", "multiple ways", detail)


def check_headings_labels(soup, result):
    """2.4.6 headings and labels are descriptive"""
    generic = {"untitled", "heading", "title", "section", "content", "page", "text"}
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = visible_text(h).strip()
        if not text and not has_aria_label(h):
            result.add("warning", "2.4.6", "headings and labels",
                       f"<{h.name}> is empty",
                       snippet(h))
        elif text.lower() in generic:
            result.add("warning", "2.4.6", "headings and labels",
                       f'<{h.name}> text "{text}" is likely non-descriptive',
                       snippet(h))


def check_error_identification(soup, result):
    """3.3.1 forms should have infrastructure to identify input errors"""
    for form in soup.find_all("form"):
        inputs = [
            i for i in form.find_all(["input", "select", "textarea"])
            if i.get("type", "text").lower() not in
               {"hidden", "submit", "reset", "button", "image"}
        ]
        if not inputs:
            continue

        checks = {
            "aria-invalid attribute":         bool(form.find(attrs={"aria-invalid": True})),
            "role=alert element":             bool(form.find(attrs={"role": re.compile(r"^alert$", re.I)})),
            "aria-live region":               bool(form.find(attrs={"aria-live": True})),
            'class matching "error"/"invalid"': bool(form.find(class_=re.compile(r"\berror\b|\binvalid\b", re.I))),
        }

        if any(checks.values()):
            continue

        detail = (
            f"form with {len(inputs)} input(s) has no detectable error-identification pattern; "
            "the following were searched and not found:\n"
            + "\n".join(f"\t· {name}: absent" for name in checks)
        )
        result.add("warning", "3.3.1", "error identification", detail, snippet(form)[:80])
        break


def check_labels_instructions(soup, result):
    """3.3.2 required fields should indicate they are required"""
    page_text = soup.get_text(" ", strip=True).lower()
    has_global_note = bool(re.search(r"required field|fields marked|denotes required|\* =", page_text))
    for inp in soup.find_all(["input", "select", "textarea"]):
        itype = inp.get("type", "text").lower()
        if itype in {"hidden", "submit", "reset", "button", "image"}:
            continue
        is_required = (
            inp.get("required") is not None or
            inp.get("aria-required") in {"true", "True"}
        )
        if not is_required:
            continue
        inp_id     = inp.get("id")
        label      = soup.find("label", attrs={"for": inp_id}) if inp_id else inp.find_parent("label")
        label_text = (visible_text(label) + " " + (label.get("aria-label", "") if label else "")).lower()
        aria_label = inp.get("aria-label", "").lower()
        indicated  = (
            "required" in label_text or
            "required" in aria_label or
            "*" in label_text or
            has_global_note
        )
        if not indicated:
            label_display = f'"{visible_text(label)}"' if label else "no label found"
            required_via  = (
                'required attribute' if inp.get("required") is not None
                else 'aria-required="true"'
            )
            result.add("warning", "3.3.2", "labels or instructions",
                       f"input is marked required ({required_via}) but has no visible required indicator:\n"
                       f"\t· label text: {label_display}\n"
                       f"\t· checked for: word \"required\", asterisk (*) in label, "
                       f"or page-level note (\"required field\", \"fields marked\", etc.)\n"
                       f"\t· global note on page: {'found' if has_global_note else 'not found'}",
                       snippet(inp))
            break


def check_error_suggestion(soup, result):
    """3.3.3 inputs with format constraints should provide correction hints"""
    for inp in soup.find_all("input"):
        itype = inp.get("type", "text").lower()
        if itype in {"hidden", "submit", "reset", "button", "image", "checkbox", "radio"}:
            continue
        pattern = inp.get("pattern")
        if not pattern:
            continue
        hints_present = {
            "aria-describedby": bool(inp.get("aria-describedby")),
            "title":            bool(inp.get("title", "").strip()),
            "placeholder":      bool(inp.get("placeholder", "").strip()),
        }
        if not any(hints_present.values()):
            result.add("warning", "3.3.3", "error suggestion",
                       f"input has pattern=\"{pattern}\" but no mechanism to describe the expected format:\n"
                       + "\n".join(
                           f"\t· {name}: {'present' if found else 'absent'}"
                           for name, found in hints_present.items()
                       )
                       + "\n\t· add at least one: aria-describedby pointing to a format hint, "
                       "a title attribute, or a descriptive placeholder",
                       snippet(inp))


_SENSITIVE_FORM_RE = re.compile(
    r"\b(payment|checkout|purchase|order|credit.?card|subscribe|billing|"
    r"legal|agreement|contract|terms)\b", re.I
)

def check_error_prevention(soup, result):
    """3.3.4 legal/financial forms should be reversible, checked, or confirmed"""
    for form in soup.find_all("form"):
        form_text = visible_text(form)
        match = _SENSITIVE_FORM_RE.search(form_text)
        if not match:
            continue
        has_checkbox = bool(form.find("input", attrs={"type": re.compile(r"^checkbox$", re.I)}))
        confirm_match = re.search(r"\b(confirm|review|agree|accept)\b", form_text, re.I)
        if not has_checkbox and not confirm_match:
            result.add("warning", "3.3.4", "error prevention",
                       f"form appears to handle legal/financial data "
                       f"(triggered by keyword: \"{match.group(0)}\") "
                       f"but has no detectable confirmation or review step:\n"
                       f"\t· no checkbox (for agree/confirm) found\n"
                       f"\t· no text matching confirm/review/agree/accept found\n"
                       f"\t· add a confirmation checkbox or review step before submission",
                       snippet(form)[:80])


_CAPTCHA_RE = re.compile(r"captcha|recaptcha|hcaptcha", re.I)

def check_accessible_auth(soup, result):
    """3.3.8 authentication must not rely solely on cognitive function tests"""
    _ALT_SIGNALS = ("audio", "alternative", "accessible", "bypass", "skip")
    for el in soup.find_all(True):
        cls    = " ".join(el.get("class", []))
        el_id  = el.get("id", "")
        match  = _CAPTCHA_RE.search(cls + " " + el_id)
        if not match:
            continue
        el_html   = str(el)
        found_alt = next((s for s in _ALT_SIGNALS if s in el_html.lower()), None)
        if not found_alt:
            result.add("warning", "3.3.8", "accessible authentication",
                       f"captcha detected (matched \"{match.group(0)}\" in "
                       f"{'class' if match.group(0) in cls else 'id'}) "
                       f"with no apparent accessible alternative:\n"
                       f"\t· searched surrounding markup for signals: "
                       f"{', '.join(_ALT_SIGNALS)} ; none found\n"
                       f"\t· authentication must not rely solely on a cognitive function test; "
                       f"provide an audio captcha, a fallback login method, or a support contact",
                       snippet(el))
        break


# fetch, orchestrate
def fetch(url: str) -> tuple[str, str]:
    headers = {"User-Agent": "a11y-audit/1.0 (accessibility checker)"}
    r = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    r.raise_for_status()
    return r.text, r.url


def audit(url: str) -> AuditResult:
    html, final_url = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    result = AuditResult(url=final_url)

    check_audit_limitations(soup, result)  # run first, not tracked for passes

    _CHECKS = [
        (check_page_title,                    "2.4.2",         "page title"),
        (check_language,                       "3.1.1",         "language of page"),
        (check_language_validity,              "3.1.1",         "language tag format"),
        (check_language_parts,                 "3.1.2",         "language of parts"),
        (check_images,                         "1.1.1",         "non-text content (images)"),
        (check_form_labels,                    "1.3.1, 4.1.2",  "form labels"),
        (check_buttons,                        "4.1.2",         "button names"),
        (check_links,                          "2.4.4",         "link purpose"),
        (check_label_in_name,                  "2.5.3",         "label in name"),
        (check_headings,                       "1.3.1",         "heading structure"),
        (check_headings_labels,                "2.4.6",         "headings and labels"),
        (check_iframes,                        "4.1.2",         "iframe titles"),
        (check_skip_link,                      "2.4.1",         "bypass blocks"),
        (check_tables,                         "1.3.1",         "table structure"),
        (check_aria_roles,                     "4.1.2",         "aria roles"),
        (check_autoplay,                       "1.4.2",         "autoplay"),
        (check_color_contrast_hints,           "1.4.3",         "contrast (inline only)"),
        (check_target_size,                    "2.5.8",         "target size"),
        (check_viewport,                       "1.4.4",         "viewport scalability"),
        (check_duplicate_ids,                  "4.1.1",         "duplicate ids"),
        (check_dangling_aria_refs,             "4.1.2",         "aria references"),
        (check_positive_tabindex,              "2.4.3",         "focus order (tabindex)"),
        (check_aria_hidden_focusable,          "4.1.2",         "aria-hidden on focusable"),
        (check_video_captions,                 "1.2.2",         "video captions"),
        (check_video_description,              "1.2.5",         "audio description"),
        (check_audio_description_alternative,  "1.2.3",         "audio description or media alternative"),
        (check_svg_accessibility,              "1.1.1",         "svg accessible names"),
        (check_meta_refresh,                   "2.2.1",         "meta refresh"),
        (check_autocomplete,                   "1.3.5",         "autocomplete purpose"),
        (check_empty_labels,                   "1.3.1, 4.1.2",  "empty labels"),
        (check_moving_content,                 "2.2.2",         "moving content"),
        (check_image_buttons,                  "1.1.1",         "image button alt text"),
        (check_audio,                          "1.2.1",         "audio transcripts"),
        (check_image_maps,                     "1.1.1",         "image map areas"),
        (check_role_img,                       "1.1.1",         "role=img names"),
        (check_object_fallback,                "1.1.1",         "object fallback"),
        (check_fieldsets,                      "1.3.1",         "fieldset legends"),
        (check_details_summary,                "4.1.2",         "details/summary"),
        (check_progress_meter,                 "4.1.2",         "progress/meter names"),
        (check_status_messages,                "4.1.3",         "status messages"),
        (check_duplicate_adjacent_links,       "2.4.4",         "adjacent duplicate links"),
        (check_multiple_ways,                  "2.4.5",         "multiple ways"),
        (check_error_identification,           "3.3.1",         "error identification"),
        (check_labels_instructions,            "3.3.2",         "labels or instructions"),
        (check_error_suggestion,               "3.3.3",         "error suggestion"),
        (check_error_prevention,               "3.3.4",         "error prevention"),
        (check_accessible_auth,                "3.3.8",         "accessible authentication"),
    ]

    for fn, criterion, title in _CHECKS:
        before = sum(1 for i in result.issues if i.level in {"error", "warning"})
        fn(soup, result)
        after = sum(1 for i in result.issues if i.level in {"error", "warning"})
        if after == before:
            result.passed(criterion, title)

    return result


# wcag criterion registry ; drives the matrix
# (sc, level, name, static)
# static=False: requires rendering or interaction; always "n/a" in matrix
_WCAG_CRITERIA = [
    ("1.1.1",  "A",  "Non-text content",                 True),
    ("1.2.1",  "A",  "Audio-only / video-only",          True),
    ("1.2.2",  "A",  "Captions (prerecorded)",           True),
    ("1.2.3",  "A",  "Audio description or alt",         True),
    ("1.2.4",  "AA", "Captions (live)",                  False),
    ("1.2.5",  "AA", "Audio description (prerecorded)",  True),
    ("1.3.1",  "A",  "Info and relationships",           True),
    ("1.3.2",  "A",  "Meaningful sequence",              False),
    ("1.3.3",  "A",  "Sensory characteristics",          False),
    ("1.3.4",  "AA", "Orientation",                      False),
    ("1.3.5",  "AA", "Identify input purpose",           True),
    ("1.4.1",  "A",  "Use of color",                     False),
    ("1.4.2",  "A",  "Audio control",                    True),
    ("1.4.3",  "AA", "Contrast (minimum)",               False),
    ("1.4.4",  "AA", "Resize text",                      True),
    ("1.4.5",  "AA", "Images of text",                   False),
    ("1.4.10", "AA", "Reflow",                           False),
    ("1.4.11", "AA", "Non-text contrast",                False),
    ("1.4.12", "AA", "Text spacing",                     False),
    ("1.4.13", "AA", "Content on hover or focus",        False),
    ("2.1.1",  "A",  "Keyboard",                         False),
    ("2.1.2",  "A",  "No keyboard trap",                 False),
    ("2.1.4",  "A",  "Character key shortcuts",          False),
    ("2.2.1",  "A",  "Timing adjustable",                True),
    ("2.2.2",  "A",  "Pause, stop, hide",                True),
    ("2.3.1",  "A",  "Three flashes or below threshold", False),
    ("2.4.1",  "A",  "Bypass blocks",                    True),
    ("2.4.2",  "A",  "Page titled",                      True),
    ("2.4.3",  "A",  "Focus order",                      True),   # partial: tabindex only
    ("2.4.4",  "A",  "Link purpose",                     True),
    ("2.4.5",  "AA", "Multiple ways",                    True),
    ("2.4.6",  "AA", "Headings and labels",              True),
    ("2.4.7",  "AA", "Focus visible",                    False),
    ("2.4.11", "AA", "Focus not obscured",               False),
    ("2.5.1",  "A",  "Pointer gestures",                 False),
    ("2.5.2",  "A",  "Pointer cancellation",             False),
    ("2.5.3",  "AA", "Label in name",                    True),
    ("2.5.4",  "A",  "Motion actuation",                 False),
    ("2.5.7",  "AA", "Dragging movements",               False),
    ("2.5.8",  "AA", "Target size (minimum)",            True),
    ("3.1.1",  "A",  "Language of page",                 True),
    ("3.1.2",  "AA", "Language of parts",                True),
    ("3.2.1",  "A",  "On focus",                         False),
    ("3.2.2",  "A",  "On input",                         False),
    ("3.2.3",  "AA", "Consistent navigation",            False),  # cross-page
    ("3.2.4",  "AA", "Consistent identification",        False),  # cross-page
    ("3.2.6",  "A",  "Consistent help",                  False),  # cross-page
    ("3.3.1",  "A",  "Error identification",             True),
    ("3.3.2",  "A",  "Labels or instructions",           True),
    ("3.3.3",  "AA", "Error suggestion",                 True),
    ("3.3.4",  "AA", "Error prevention",                 True),
    ("3.3.7",  "A",  "Redundant entry",                  False),
    ("3.3.8",  "AA", "Accessible authentication",        True),
    ("4.1.1",  "A",  "Parsing",                          True),
    ("4.1.2",  "A",  "Name, role, value",                True),
    ("4.1.3",  "AA", "Status messages",                  True),
]


# report
_WCAG_LEVEL = {
    "1.1.1": "A",   "1.2.1": "A",   "1.2.2": "A",   "1.2.3": "A",
    "1.2.4": "AA",  "1.2.5": "AA",
    "1.3.1": "A",   "1.3.2": "A",   "1.3.3": "A",   "1.3.4": "AA",  "1.3.5": "AA",
    "1.4.1": "A",   "1.4.2": "A",   "1.4.3": "AA",  "1.4.4": "AA",
    "1.4.5": "AA",  "1.4.10": "AA", "1.4.11": "AA", "1.4.12": "AA", "1.4.13": "AA",
    "2.1.1": "A",   "2.1.2": "A",   "2.1.4": "A",
    "2.2.1": "A",   "2.2.2": "A",   "2.3.1": "A",
    "2.4.1": "A",   "2.4.2": "A",   "2.4.3": "A",   "2.4.4": "A",
    "2.4.5": "AA",  "2.4.6": "AA",  "2.4.7": "AA",  "2.4.11": "AA",
    "2.5.1": "A",   "2.5.2": "A",   "2.5.3": "AA",  "2.5.4": "A",
    "2.5.7": "AA",  "2.5.8": "AA",
    "3.1.1": "A",   "3.1.2": "AA",
    "3.2.1": "A",   "3.2.2": "A",   "3.2.3": "AA",  "3.2.4": "AA",  "3.2.6": "A",
    "3.3.1": "A",   "3.3.2": "A",   "3.3.3": "AA",  "3.3.4": "AA",
    "3.3.7": "A",   "3.3.8": "AA",
    "4.1.1": "A",   "4.1.2": "A",   "4.1.3": "AA",
}


def _level_tag(criterion: str) -> str:
    codes = re.findall(r"\d+\.\d+\.\d+", criterion)
    levels = [_WCAG_LEVEL.get(c, "") for c in codes]
    if "AA" in levels:
        return "aa"
    if "A" in levels:
        return "a"
    return ""


COLORS = {
    "error":   "\033[91m",
    "warning": "\033[93m",
    "info":    "\033[94m",
    "pass":    "\033[92m",
    "reset":   "\033[0m",
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _report_lines(result: AuditResult) -> list[str]:
    lines = ["\n", f"\ta11y audit: {result.url}", "\n"]

    if not result.issues and not result.passes:
        lines.append("\tno issues detected (static analysis only)\n")
        return lines

    for level in ("error", "warning", "info"):
        issues = [i for i in result.issues if i.level == level]
        if not issues:
            continue
        for issue in issues:
            c = COLORS[level]
            r = COLORS["reset"]
            tag = _level_tag(issue.criterion)
            tag_str = f" [wcag {tag}]" if tag else ""
            lines.append(f"\t{c}[{issue.criterion}]{tag_str} {issue.title}{r}")
            lines.append(f"\t\t{issue.detail}")
            if issue.element:
                lines.append(f"\t\t\033[2m{issue.element}\033[0m")
            lines.append("")

    if result.passes:
        c = COLORS["pass"]
        r = COLORS["reset"]
        lines.append(f"\t{c}passed{r}")
        for p in result.passes:
            tag = _level_tag(p.criterion)
            tag_str = f" [wcag {tag}]" if tag else ""
            lines.append(f"\t{c}[{p.criterion}]{tag_str} {p.title}{r}")
        lines.append("")

    errors, warnings, infos = result.summary()
    lines.append("\n")
    lines.append(
        f"\t{COLORS['error']}errors: {errors}{COLORS['reset']}  "
        f"{COLORS['warning']}warnings: {warnings}{COLORS['reset']}  "
        f"{COLORS['info']}info: {infos}{COLORS['reset']}  "
        f"{COLORS['pass']}passed: {len(result.passes)}{COLORS['reset']}\n"
    )
    return lines


def _externals_lines(result: AuditResult) -> list[str]:
    lines = ["\n", f"\texternal resources: {result.url}", "\n"]
    labels = {
        "stylesheets": "stylesheets",
        "scripts":     "scripts",
        "iframes":     "iframes",
        "fonts":       "fonts",
        "media":       "media (video; audio)",
    }
    any_found = False
    for key, label in labels.items():
        items = [i for i in result.externals.get(key, []) if i]
        if not items:
            continue
        any_found = True
        lines.append(f"\t[{label}] ({len(items)})")
        for item in items:
            lines.append(f"\t\t{item}")
        lines.append("")
    if not any_found:
        lines.append("\tno external resources detected")
        lines.append("")
    return lines


def print_report(result: AuditResult):
    for line in _report_lines(result):
        print(line)


def write_report(result: AuditResult, path: str):
    report    = [strip_ansi(l) for l in _report_lines(result)]
    externals = _externals_lines(result)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report + externals))


# matrix
def print_matrix(results: list[Optional[AuditResult]], urls: list[str]):
    labels  = [_url_label(u) for u in urls]
    sc_col  = 7
    name_col = 32
    lvl_col  = 4
    cell_w   = max(6, *(len(l) + 2 for l in labels))

    CE = COLORS["error"]
    CW = COLORS["warning"]
    CP = COLORS["pass"]
    CD = "\033[2m"
    CR = COLORS["reset"]

    def cell_str(s: str) -> str:
        if s == "E": return f"{CE}{'error':^{cell_w}}{CR}"
        if s == "W": return f"{CW}{'warn':^{cell_w}}{CR}"
        if s == "P": return f"{CP}{'pass':^{cell_w}}{CR}"
        if s == "I": return f"{CD}{'n/a':^{cell_w}}{CR}"
        return f"{'—':^{cell_w}}"

    header_cells = "".join(f"{l:^{cell_w}}" for l in labels)
    sep = "─" * (sc_col + 1 + name_col + 1 + lvl_col + 1 + cell_w * len(labels))

    print(f"\n\twcag a/aa matrix: {len(urls)} page(s)\n")
    print(f"\t{'SC':<{sc_col}} {'Criterion':<{name_col}} {'Lvl':<{lvl_col}} {header_cells}")
    print(f"\t{sep}")

    for sc, level, name, static in _WCAG_CRITERIA:
        cells = "".join(
            cell_str(_criterion_status(r, sc) if static else "I")
            for r in results
        )
        name_trim = name[:name_col - 1] if len(name) >= name_col else name
        print(f"\t{sc:<{sc_col}} {name_trim:<{name_col}} {level:<{lvl_col}} {cells}")

    print()


def write_matrix_csv(results: list[Optional[AuditResult]], urls: list[str], path: str):
    import csv
    labels    = [_url_label(u) for u in urls]
    label_map = {"E": "error", "W": "warning", "P": "pass", "I": "n/a", "—": "fetch_failed"}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sc", "criterion", "level", "statically_testable"] + labels)
        for sc, level, name, static in _WCAG_CRITERIA:
            row = [sc, name, level, "yes" if static else "no"]
            for result in results:
                s = _criterion_status(result, sc) if static else "I"
                row.append(label_map.get(s, s))
            writer.writerow(row)


def _url_to_slug(url: str, max_len: int = 60) -> str:
    """derive a filesystem-safe slug from a full url"""
    stripped = re.sub(r"^https?://", "", url).rstrip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stripped)
    return slug[:max_len]


if __name__ == "__main__":
    args = sys.argv[1:]
    show_matrix = "--matrix" in args
    urls_raw = [a for a in args if not a.startswith("--")]

    if not urls_raw:
        sys.exit("usage: python ally.py [--matrix] <url> [url2 ...]")

    from pathlib import Path
    output_dir = Path("access_output")
    output_dir.mkdir(exist_ok=True)

    urls = [
        u if u.startswith(("http://", "https://")) else "https://" + u
        for u in urls_raw
    ]

    total_errors = 0
    results: list[Optional[AuditResult]] = []

    for i, url in enumerate(urls, 1):
        if len(urls) > 1:
            print(f"\n")
            print(f"site {i} of {len(urls)}: {url}")
            print(f"\n")

        try:
            result = audit(url)
            results.append(result)
            print_report(result)

            slug = _url_to_slug(url) or "audit"
            txt_path = output_dir / f"ally_{slug}_report.txt"
            write_report(result, str(txt_path))
            print(f"\treport written to: {txt_path}\n")

            errors, _, _ = result.summary()
            total_errors += errors

        except requests.RequestException as e:
            print(f"  fetch failed: {e}\n")
            results.append(None)
            total_errors += 1

    if len(urls) > 1:
        print(f"\n")
        print(f"\t{len(urls)} sites audited: total errors: {total_errors}")
        print(f"\n")

    if show_matrix:
        print_matrix(results, urls)

        hostnames = list({urlparse(u).hostname for u in urls if urlparse(u).hostname})
        base = hostnames[0] if len(hostnames) == 1 else "matrix"
        csv_path = output_dir / f"ally_{base}_matrix.csv"
        write_matrix_csv(results, urls, str(csv_path))
        print(f"\tmatrix written to: {csv_path}\n")

    sys.exit(1 if total_errors else 0)
