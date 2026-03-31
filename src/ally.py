#!/usr/bin/env python3
"""
author: Daniel Quigley
contact: dquigleydev@gmail.com

wcag 2.1 aa accessibility audit
NOTE: only for static sites

    python ally.py https://example.com
    python ally.py https://example.com https://example.com/about https://other.com

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
    level: str          # arranged data as "error" | "warning" | "info"
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
    return s[:max_len] + ("…" if len(s) > max_len else "")


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
    # tabindex="0" or any positive value (or empty string, treated as 0 by browsers)
    if tabindex is not None:
        return True
    return False


# individual checks
def check_page_title(soup, result):
    title = soup.find("title")
    if not title or not title.get_text(strip=True):
        result.add("error", "2.4.2", "page title",
                   "document has no meaningful <title> element")


def check_language(soup, result):
    html = soup.find("html")
    if not html or not html.get("lang", "").strip():
        result.add("error", "3.1.1", "language of page",
                   "<html> element is missing a lang attribute")


def check_images(soup, result):
    for img in soup.find_all("img"):
        if img.get("role") == "presentation":
            continue
        alt = img.get("alt")
        if alt is None:
            result.add("error", "1.1.1", "non-text content",
                       "image missing alt attribute entirely",
                       snippet(img))
        elif alt.strip().lower() in {"image", "photo", "picture", "graphic", "icon"}:
            result.add("warning", "1.1.1", "non-text content",
                       f'alt text "{alt}" is likely non-descriptive',
                       snippet(img))


def check_form_labels(soup, result):
    inputs = soup.find_all(["input", "select", "textarea"])
    for inp in inputs:
        itype = inp.get("type", "text").lower()
        if itype in {"hidden", "submit", "reset", "button", "image"}:
            continue
        inp_id = inp.get("id")
        has_label = (
            has_aria_label(inp)
            or inp.get("title")
            or (inp_id and soup.find("label", attrs={"for": inp_id}))
            or inp.find_parent("label")
        )
        if not has_label:
            result.add("error", "1.3.1, 4.1.2", "form label missing",
                       f'<{inp.name} type="{itype}"> has no associated label',
                       snippet(inp))


def check_buttons(soup, result):
    for btn in soup.find_all("button"):
        if not visible_text(btn) and not has_aria_label(btn) and not btn.get("title"):
            result.add("error", "4.1.2", "button name",
                       "button has no accessible name (no text, aria-label, title)",
                       snippet(btn))


def check_links(soup, result):
    for a in soup.find_all("a", href=True):
        text = visible_text(a)
        aria = a.get("aria-label", "").strip()
        title = a.get("title", "").strip()
        name = aria or title or text
        if not name:
            result.add("error", "2.4.4", "link purpose",
                       "link has no accessible name",
                       snippet(a))
        elif name.lower() in {"click here", "here", "read more", "more", "link", "learn more"}:
            result.add("warning", "2.4.4", "link purpose",
                       f'link text "{name}" is non-descriptive out of context',
                       snippet(a))


def check_headings(soup, result):
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        result.add("warning", "1.3.1", "info and relationships",
                   "no heading elements found; is page structure communicated semantically?")
        return

    h1s = soup.find_all("h1")
    if len(h1s) == 0:
        result.add("error", "1.3.1", "heading structure",
                   "no <h1> element found")
    elif len(h1s) > 1:
        result.add("warning", "1.3.1", "heading structure",
                   f"multiple <h1> elements ({len(h1s)}); only one usually appropriate")

    levels = [int(h.name[1]) for h in headings]
    for i in range(1, len(levels)):
        if levels[i] - levels[i - 1] > 1:
            result.add("warning", "1.3.1", "heading hierarchy",
                       f"heading level jumps from h{levels[i-1]} to h{levels[i]}; skipped level",
                       snippet(headings[i]))


def check_iframes(soup, result):
    for frame in soup.find_all("iframe"):
        if not frame.get("title") and not has_aria_label(frame):
            result.add("error", "4.1.2", "iframe title",
                       "iframe lacks a title attribute",
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
        result.add("warning", "2.4.1", "bypass blocks",
                   "no skip-navigation link detected near top of page")


def check_tables(soup, result):
    for table in soup.find_all("table"):
        headers = table.find_all("th")
        caption = table.find("caption")
        aria_label = has_aria_label(table)
        if not headers:
            result.add("warning", "1.3.1", "table headers",
                       "table has no <th> elements; data relationships may not be programmatically determinable",
                       snippet(table)[:80])
        if not caption and not aria_label:
            result.add("warning", "1.3.1", "table caption",
                       "table has no <caption> or aria-label",
                       snippet(table)[:80])
        for th in headers:
            if not th.get("scope") and not th.get("id"):
                result.add("warning", "1.3.1", "table header scope",
                           "<th> missing scope attribute",
                           snippet(th))
                break  # one warning per table is enough, I think


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
    dupes = [id_ for id_, count in Counter(ids).items() if count > 1]
    for id_ in dupes:
        result.add("error", "4.1.1", "duplicate id",
                   f'id="{id_}" appears multiple times; aria references and label associations targeting this id are unreliable')


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
    for el in soup.find_all(tabindex=True):
        try:
            val = int(el["tabindex"])
        except ValueError:
            continue
        if val > 0:
            result.add("warning", "2.4.3", "positive tabindex",
                       f"tabindex={val} overrides natural focus order; use 0 or -1 only",
                       snippet(el))


def check_aria_hidden_focusable(soup, result):
    for el in soup.find_all(attrs={"aria-hidden": "true"}):
        # check the element itself
        self_focusable = _is_in_tab_order(el)
        # check any focusable descendant (respecting their own tabindex)
        descendant_focusable = bool(el.find(
            lambda t: isinstance(t, Tag) and _is_in_tab_order(t)
        ))
        if self_focusable or descendant_focusable:
            result.add("error", "4.1.2", "aria-hidden on focusable element",
                       'aria-hidden="true" is set on a focusable element or container; '
                       "it remains in the tab order, but invisible to screen readers",
                       snippet(el))


def check_video_captions(soup, result):
    for video in soup.find_all("video"):
        tracks = video.find_all("track")
        has_captions = any(
            t.get("kind", "").lower() in {"captions", "subtitles"}
            for t in tracks
        )
        if not has_captions:
            result.add("error", "1.2.2", "video captions",
                       '<video> has no <track kind="captions"> or <track kind="subtitles">',
                       snippet(video))


def check_video_description(soup, result):
    """1.2.5 prerecorded video needs an audio description track"""
    for video in soup.find_all("video"):
        tracks = video.find_all("track")
        has_description = any(
            t.get("kind", "").lower() == "descriptions"
            for t in tracks
        )
        if not has_description:
            result.add("warning", "1.2.5", "audio description",
                       '<video> has no <track kind="descriptions">; '
                       "prerecorded video with meaningful visual content requires audio description",
                       snippet(video))


def check_svg_accessibility(soup, result):
    for svg in soup.find_all("svg"):
        role = svg.get("role", "")
        if role in {"presentation", "none"}:
            continue
        if svg.get("aria-hidden") == "true":
            continue
        has_title = bool(svg.find("title"))
        has_name  = has_aria_label(svg) or svg.get("title")
        if not has_title and not has_name:
            result.add("warning", "1.1.1", "svg accessible name",
                       "inline <svg> has no <title>, aria-label, or aria-labelledby; "
                       'add a <title> if meaningful, or role="presentation" if decorative',
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
        candidates = " ".join(filter(None, [
            inp.get("id", ""), inp.get("name", ""),
            inp.get("placeholder", ""), inp.get("aria-label", "")
        ])).lower()
        for keyword, purpose in _AUTOCOMPLETE_HINTS.items():
            if keyword in candidates:
                result.add("warning", "1.3.5", "autocomplete purpose",
                           f'input appears to collect "{keyword}" data but declares no autocomplete; '
                           f'consider autocomplete="{purpose}"',
                           snippet(inp))
                break


def check_empty_labels(soup, result):
    for label in soup.find_all("label"):
        text = "".join(
            t for t in label.strings
            if t.strip() and t.parent.name not in {"input", "select", "textarea"}
        ).strip()
        if not text and not label.get("aria-label"):
            result.add("warning", "1.3.1, 4.1.2", "empty label",
                       "<label> has no visible text; associated control will have no accessible name",
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
    """2.5.3 when visible text and aria-label coexist on an interactive element,
       the visible text must appear as a substring of the aria-label (case-insensitive);
       speech-input users activate controls by speaking the visible label"""
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
        # skip elements with no visible text (icon-only) or long containers
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
    """4.1.3 containers that look like notification regions need role=status/alert or aria-live
       so that dynamically injected messages are announced without a focus change;
       this is heuristic — false positives are possible"""
    live_roles = {"alert", "status", "log", "marquee", "timer"}
    for el in soup.find_all(True):
        cls = " ".join(el.get("class", []))
        el_id = el.get("id", "")
        if not _STATUS_CLASS_RE.search(cls + " " + el_id):
            continue
        role = el.get("role", "")
        aria_live = el.get("aria-live", "")
        if role not in live_roles and not aria_live:
            result.add("warning", "4.1.3", "status message",
                       f'element with class/id matching notification pattern ("{(cls or el_id).strip()}") '
                       "has no role=status/alert or aria-live; "
                       "dynamically injected messages won't be announced to screen readers",
                       snippet(el))


def check_language_parts(soup, result):
    """3.1.2 inline lang overrides on elements must also be valid BCP 47"""
    html_el = soup.find("html")
    for el in soup.find_all(attrs={"lang": True}):
        if el == html_el:
            continue  # already covered by check_language_validity
        lang = el.get("lang", "").strip()
        if not re.match(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$", lang):
            result.add("warning", "3.1.2", "language of parts",
                       f'lang="{lang}" on inline element does not look like a valid BCP 47 tag',
                       snippet(el))


def check_audit_limitations(soup, result):
    """
    detect page characteristics that degrade reliability;
    note: reports as 'info', so caller knows which findings to distrust
    """
    html_text = str(soup)

    # spa/js framework fingerprints
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

    # low content-to-script ratio (just a generic spa signal)
    body = soup.find("body")
    body_text_len = len(body.get_text(strip=True)) if body else 0
    script_tags = len(soup.find_all("script", src=True))
    if body_text_len < 200 and script_tags >= 3 and not frameworks:
        result.add("info", "—", "likely js-rendered content",
                   f"body text is sparse ({body_text_len} chars) with {script_tags} external scripts; "
                   "dom may be populated after page load; audit coverage is probably incomplete!")

    # external stylesheets to contrast is unverifiable
    stylesheets = soup.find_all("link", rel=lambda r: r and "stylesheet" in r)
    result.externals["stylesheets"] = [
        s.get("href", "(no href)") for s in stylesheets
    ]
    if stylesheets:
        result.add("info", "1.4.3", "contrast check incomplete",
                   f"{len(stylesheets)} external stylesheet(s) found; color contrast cannot be "
                   "evaluated without a rendering engine; use axe-core or browser devtools")

    # external scripts
    result.externals["scripts"] = [
        s.get("src") for s in soup.find_all("script", src=True)
    ]

    # iframes to subtrees are unaudited
    iframes = soup.find_all("iframe")
    result.externals["iframes"] = [
        f.get("src", "(no src)") for f in iframes
    ]
    if iframes:
        srcs = result.externals["iframes"][:3]
        result.add("info", "—", "iframe content not audited",
                   f"{len(iframes)} iframe(s) found; subtrees are not fetched or checked "
                   f"sources: {', '.join(srcs)}")

    # fonts
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

    # media
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

    # noscript, js is load-bearing
    if soup.find("noscript"):
        result.add("info", "—", "js dependency confirmed",
                   "<noscript> element present, confirming the page behaves differently without js; "
                   "interactive states (focus, expanded menus, modals) are not audited here")

    # focus and keyboard behaviour
    result.add("info", "2.1.1, 2.4.3, 2.1.2",
               "keyboard and focus order not tested",
               "focus order, keyboard operability, and focus traps require actual interaction simulation "
               "(e.g., playwright axe-core); not covered here!")


def check_autoplay(soup, result):
    for media in soup.find_all(["video", "audio"]):
        if media.get("autoplay") is not None and media.get("muted") is None:
            result.add("error", "1.4.2, 1.4.3", "audio control",
                       "media autoplays without muted attribute; must provide pause/stop control",
                       snippet(media))


def check_color_contrast_hints(soup, result):
    for el in soup.find_all(style=True):
        style = el.get("style", "")
        color = parse_inline_color(style, "color")
        bg    = parse_inline_color(style, "background-color")
        if color and bg:
            result.add("info", "1.4.3", "contrast (inline styles only)",
                       f"element has inline color:{color} + background:{bg} "
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
    """1.1.1 <input type="image"> is an interactive image and needs alt text"""
    for inp in soup.find_all("input", attrs={"type": re.compile(r"^image$", re.I)}):
        alt = inp.get("alt", "").strip()
        if not alt and not has_aria_label(inp) and not inp.get("title", "").strip():
            result.add("error", "1.1.1", "image button alt text",
                       '<input type="image"> has no alt, aria-label, or title; '
                       "acts as both a button and image, needs accessible name",
                       snippet(inp))


def check_audio(soup, result):
    """1.2.1 prerecorded audio-only content needs a text transcript or track"""
    for audio in soup.find_all("audio"):
        if audio.get("autoplay") is not None and audio.get("muted") is not None:
            continue  # muted autoplay is decorative
        tracks = audio.find_all("track")
        has_transcript = any(
            t.get("kind", "").lower() in {"captions", "descriptions", "subtitles"}
            for t in tracks
        )
        # also accept an adjacent link that plausibly points to transcript
        parent_text = visible_text(audio.parent).lower() if audio.parent else ""
        has_transcript_link = "transcript" in parent_text

        if not has_transcript and not has_transcript_link:
            result.add("warning", "1.2.1", "audio transcript",
                       "<audio> element has no <track> and no adjacent transcript link; "
                       "prerecorded audio-only content requires a text alternative",
                       snippet(audio))


def check_image_maps(soup, result):
    """1.1.1 <area> elements in image maps need alt text"""
    for area in soup.find_all("area"):
        if area.get("href") is None and area.get("role") in {"presentation", "none"}:
            continue
        alt = area.get("alt", "").strip()
        if not alt and not has_aria_label(area):
            result.add("error", "1.1.1", "image map area alt",
                       "<area> in image map has no alt text",
                       snippet(area))


def check_role_img(soup, result):
    """1.1.1 elements with role='img' need an accessible name"""
    for el in soup.find_all(attrs={"role": True}):
        if "img" not in el["role"].split():
            continue
        if el.get("aria-hidden") == "true":
            continue
        if not has_aria_label(el) and not el.get("title", "").strip():
            result.add("error", "1.1.1", "role=img without name",
                       f'<{el.name} role="img"> has no aria-label or aria-labelledby',
                       snippet(el))


def check_object_fallback(soup, result):
    """1.1.1 <object> elements need text alternative or title"""
    for obj in soup.find_all("object"):
        has_title = bool(obj.get("title", "").strip())
        has_aria  = has_aria_label(obj)
        has_fallback = bool(visible_text(obj).strip())
        if not has_title and not has_aria and not has_fallback:
            result.add("warning", "1.1.1", "object fallback",
                       "<object> has no title, aria-label, or fallback text content; "
                       "embedded content must have text alternative",
                       snippet(obj))


def check_fieldsets(soup, result):
    """1.3.1 <fieldset> grouping related controls should have a <legend>"""
    for fs in soup.find_all("fieldset"):
        legend = fs.find("legend")
        if not legend or not visible_text(legend).strip():
            result.add("warning", "1.3.1", "fieldset without legend",
                       "<fieldset> has no <legend> or empty legend; "
                       "grouped controls (radio buttons, checkboxes) need group label",
                       snippet(fs))


def check_details_summary(soup, result):
    """4.1.2 <details> disclosure widget needs a non-empty <summary>"""
    for details in soup.find_all("details"):
        summary = details.find("summary")
        if not summary:
            result.add("warning", "4.1.2", "details without summary",
                       "<details> has no <summary> element; "
                       "browsers insert default 'details' label which is non-descriptive",
                       snippet(details))
        elif not visible_text(summary).strip() and not has_aria_label(summary):
            result.add("warning", "4.1.2", "details empty summary",
                       "<summary> has no visible text; disclosure toggle has no accessible name",
                       snippet(summary))


def check_progress_meter(soup, result):
    """4.1.2 <progress> and <meter> need accessible names"""
    for el in soup.find_all(["progress", "meter"]):
        if not has_aria_label(el) and not el.get("title", "").strip():
            # also accept an associated label via id
            el_id = el.get("id")
            has_label = el_id and soup.find("label", attrs={"for": el_id})
            if not has_label:
                result.add("warning", "4.1.2", f"{el.name} accessible name",
                           f"<{el.name}> has no accessible name (no aria-label, title, or "
                           "associated <label>); value is meaningless without context",
                           snippet(el))


def check_language_validity(soup, result):
    """3.1.1 verify lang attribute is plausible BCP 47 tag"""
    html = soup.find("html")
    if not html:
        return
    lang = html.get("lang", "").strip()
    if not lang:
        return  # redundant? absence already caught by check_language
    # BCP 47 primary subtags are 2--3 alpha chars; region subtags are 2 alpha or 3 digit
    # note: this is heuristic
    if not re.match(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$", lang):
        result.add("warning", "3.1.1", "language tag format",
                   f'lang="{lang}" does not look like a valid BCP 47 tag; '
                   "common mistakes: \"english\" instead of \"en\", "
                   "\"en-UK\" instead of \"en-GB\".",
                   snippet(html))


def check_duplicate_adjacent_links(soup, result):
    """2.4.4 adjacent links pointing to same url create redundant tab stops"""
    links = soup.find_all("a", href=True)
    for i in range(len(links) - 1):
        a, b = links[i], links[i + 1]
        # must be siblings or very close in tree!
        if a.parent != b.parent:
            continue
        href_a = a.get("href", "").strip()
        href_b = b.get("href", "").strip()
        if href_a and href_a == href_b:
            name_a = has_aria_label(a) or visible_text(a)
            name_b = has_aria_label(b) or visible_text(b)
            # only flag if at least one has visible text (not both icon-only)
            if name_a or name_b:
                result.add("warning", "2.4.4", "adjacent duplicate links",
                           f'two adjacent links both point to "{href_a}"; '
                           "consider wrapping them in a single link to avoid a redundant tab stop",
                           snippet(a))


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

    # (function, primary criterion, short title for pass report)
    # pass is recorded when a check adds no errors or warnings
    _CHECKS = [
        (check_page_title,             "2.4.2",         "page title"),
        (check_language,               "3.1.1",         "language of page"),
        (check_language_validity,      "3.1.1",         "language tag format"),
        (check_language_parts,         "3.1.2",         "language of parts"),
        (check_images,                 "1.1.1",         "non-text content (images)"),
        (check_form_labels,            "1.3.1, 4.1.2", "form labels"),
        (check_buttons,                "4.1.2",         "button names"),
        (check_links,                  "2.4.4",         "link purpose"),
        (check_label_in_name,          "2.5.3",         "label in name"),
        (check_headings,               "1.3.1",         "heading structure"),
        (check_iframes,                "4.1.2",         "iframe titles"),
        (check_skip_link,              "2.4.1",         "bypass blocks"),
        (check_tables,                 "1.3.1",         "table structure"),
        (check_aria_roles,             "4.1.2",         "aria roles"),
        (check_autoplay,               "1.4.2",         "autoplay"),
        (check_color_contrast_hints,   "1.4.3",         "contrast (inline only)"),
        (check_target_size,            "2.5.8",         "target size"),
        (check_viewport,               "1.4.4",         "viewport scalability"),
        (check_duplicate_ids,          "4.1.1",         "duplicate ids"),
        (check_dangling_aria_refs,     "4.1.2",         "aria references"),
        (check_positive_tabindex,      "2.4.3",         "focus order (tabindex)"),
        (check_aria_hidden_focusable,  "4.1.2",         "aria-hidden on focusable"),
        (check_video_captions,         "1.2.2",         "video captions"),
        (check_video_description,      "1.2.5",         "audio description"),
        (check_svg_accessibility,      "1.1.1",         "svg accessible names"),
        (check_meta_refresh,           "2.2.1",         "meta refresh"),
        (check_autocomplete,           "1.3.5",         "autocomplete purpose"),
        (check_empty_labels,           "1.3.1, 4.1.2", "empty labels"),
        (check_moving_content,         "2.2.2",         "moving content"),
        (check_image_buttons,          "1.1.1",         "image button alt text"),
        (check_audio,                  "1.2.1",         "audio transcripts"),
        (check_image_maps,             "1.1.1",         "image map areas"),
        (check_role_img,               "1.1.1",         "role=img names"),
        (check_object_fallback,        "1.1.1",         "object fallback"),
        (check_fieldsets,              "1.3.1",         "fieldset legends"),
        (check_details_summary,        "4.1.2",         "details/summary"),
        (check_progress_meter,         "4.1.2",         "progress/meter names"),
        (check_status_messages,        "4.1.3",         "status messages"),
        (check_duplicate_adjacent_links, "2.4.4",       "adjacent duplicate links"),
    ]

    for fn, criterion, title in _CHECKS:
        # use error+warning count only; info items are not violations
        before = sum(1 for i in result.issues if i.level in {"error", "warning"})
        fn(soup, result)
        after = sum(1 for i in result.issues if i.level in {"error", "warning"})
        if after == before:
            result.passed(criterion, title)

    return result


# report
_WCAG_LEVEL = {
    "1.1.1": "A",  "1.2.1": "A",  "1.2.2": "A",  "1.2.5": "AA", "1.3.1": "A",  "1.3.5": "AA",
    "1.4.2": "A",  "1.4.3": "AA", "1.4.4": "AA",
    "2.1.1": "A",  "2.1.2": "A",  "2.2.1": "A",  "2.2.2": "A",
    "2.4.1": "A",  "2.4.2": "A",  "2.4.3": "A",  "2.4.4": "A",
    "2.5.3": "AA", "2.5.8": "AA", "3.1.1": "A",  "3.1.2": "AA", "4.1.1": "A",  "4.1.2": "A",
    "4.1.3": "AA",
}

def _level_tag(criterion: str) -> str:
    """return highest wcag level found in (possibly compound) criterion string"""
    codes = re.findall(r"\d+\.\d+\.\d+", criterion)
    levels = [_WCAG_LEVEL.get(c, "") for c in codes]
    if "AA" in levels:
        return "aa"
    if "A" in levels:
        return "a"
    return ""

COLORS = {"error": "\033[91m", "warning": "\033[93m", "info": "\033[94m", "pass": "\033[92m", "reset": "\033[0m"}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _report_lines(result: AuditResult) -> list[str]:
    """build report as list of ANSI-colored lines"""
    lines = []
    lines.append(f"\n")
    lines.append(f"  a11y audit: {result.url}")
    lines.append(f"\n")

    if not result.issues and not result.passes:
        lines.append("  no issues detected (static analysis only)\n")
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
            lines.append(f"  {c}[{issue.criterion}]{tag_str} {issue.title}{r}")
            lines.append(f"    {issue.detail}")
            if issue.element:
                lines.append(f"    \033[2m{issue.element}\033[0m")
            lines.append("")

    if result.passes:
        c = COLORS["pass"]
        r = COLORS["reset"]
        lines.append(f"  {c}passed{r}")
        for p in result.passes:
            tag = _level_tag(p.criterion)
            tag_str = f" [wcag {tag}]" if tag else ""
            lines.append(f"  {c}[{p.criterion}]{tag_str} {p.title}{r}")
        lines.append("")

    errors, warnings, infos = result.summary()
    lines.append(f"\n")
    lines.append(
        f"  {COLORS['error']}errors: {errors}{COLORS['reset']}  "
        f"{COLORS['warning']}warnings: {warnings}{COLORS['reset']}  "
        f"{COLORS['info']}info: {infos}{COLORS['reset']}  "
        f"{COLORS['pass']}passed: {len(result.passes)}{COLORS['reset']}\n"
    )
    return lines


def _externals_lines(result: AuditResult) -> list[str]:
    """build external resources section as plain lines (no ANSI needed)"""
    lines = [
        f"\n",
        f"  external resources: {result.url}",
        f"\n",
    ]
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
        lines.append(f"  [{label}] ({len(items)})")
        for item in items:
            lines.append(f"    {item}")
        lines.append("")

    if not any_found:
        lines.append("  no external resources detected")
        lines.append("")

    return lines


def print_report(result: AuditResult):
    for line in _report_lines(result):
        print(line)


def write_report(result: AuditResult, path: str):
    """write audit report, external resources inventory to single txt file"""
    report    = [strip_ansi(l) for l in _report_lines(result)]
    externals = _externals_lines(result)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report + externals))


def _url_to_slug(url: str, max_len: int = 60) -> str:
    """derive a filesystem-safe slug from a full url"""
    stripped = re.sub(r"^https?://", "", url).rstrip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stripped)
    return slug[:max_len]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python ally.py <url> [url2 ...]")

    from pathlib import Path
    output_dir = Path("access_output")
    output_dir.mkdir(exist_ok=True)

    urls = [
        u if u.startswith(("http://", "https://")) else "https://" + u
        for u in sys.argv[1:]
    ]

    total_errors = 0

    for i, url in enumerate(urls, 1):
        if len(urls) > 1:
            print(f"\n")
            print(f"  site {i} of {len(urls)}: {url}")
            print(f"\n")

        try:
            result = audit(url)
            print_report(result)

            slug = _url_to_slug(url) or "audit"
            txt_path = output_dir / f"ally_{slug}_report.txt"
            write_report(result, str(txt_path))
            print(f"  report written to: {txt_path}\n")

            errors, _, _ = result.summary()
            total_errors += errors

        except requests.RequestException as e:
            print(f"  fetch failed: {e}\n")
            total_errors += 1

    if len(urls) > 1:
        print(f"\n")
        print(f"  {len(urls)} sites audited; total errors: {total_errors}")
        print(f"\n")

    sys.exit(1 if total_errors else 0)
