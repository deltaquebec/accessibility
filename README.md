# Accessibility suite

**Scripts for auditing web pages, PDF documents, and color palettes against accessibility standards**

The accessibility suite is a collection of command-line Python scripts for catching accessibility failures. These are refactored scripts that I have been working on and using for many years now, and I wanted to make them available. The idea is to operate outside of a browser-based application, and operate on fetched HTML and parsed PDF internals through the command line.

All outputs land in `./access_output/` with script-prefixed filenames. For redundancy, terminal output mirrors plain-text file output for archiving and diffing.

## Scripts

| Script | Purpose |
|--------|---------|
| **ally.py** | WCAG 2.1 AA audit for static web pages |
| **pdftree.py** | PDF structure tree inspector and visualizer |
| **textaccess.py** | PDF text accessibility checker |
| **colors.py** | perceptually distinct color palette generator |
| **harmony.py** | color harmony palette generator (LCH and HSL) |

While not technically an accessibility too, **harmony.py** is included here for completeness, since I often go back and forth with it and **colors.py** when deciding color schemes.

## Quick start

```cli
# audit webpage
python ally.py https://example.com

# inspect PDF's structure tree
python pdftree.py document.pdf

# check whether PDF's content is actually usable
python textaccess.py document.pdf

# generate 8 perceptually distinct colors, safe for all CVD types
python colors.py 8 --cvd all

# check generated colors against background for WCAG contrast
python colors.py 8 --cvd all --contrast #1a1a1a

# generate harmony palettes from an anchor color
python harmony.py "#3498db"
```


## Scripts

### ally.py web accessibility

NOTE: that is two letter l; technically, the reference is a11y, with eleven letters between a and y in accessibility. Here, we use the letter l, not the number 1.

Fetches a URL and runs static checks against WCAG 2.1 Level A and AA criteria. Before checking, checks the page for conditions that degrade audit reliability: JS frameworks; sparse DOM; external stylesheets; iframes. Findings are grouped by severity with WCAG level tags. 

**Checks include:**
- **1.1.1** `<img>`, `<input type="image">`, `<area>`, `role="img"`, `<object>` without accessible names; `<svg>` without title or aria-label
- **1.2.1 / 1.2.2** `<audio>` without transcript link or track; `<video>` without captions track
- **1.3.1** form labels, empty labels, heading hierarchy, table headers and captions, ARIA landmarks, `<fieldset>` without `<legend>`
- **1.3.5** autocomplete on personal-data fields
- **1.4.3 / 1.4.4** inline contrast hints; viewport scalability
- **2.2.1 / 2.2.2** meta refresh; `<marquee>` and `<blink>`
- **2.4.1 / 2.4.2 / 2.4.4** skip links; page title; vague or adjacent duplicate link text
- **3.1.1** `<html lang>` presence and BCP 47 format validity
- **4.1.1 / 4.1.2** duplicate IDs; button names; invalid ARIA roles; dangling ARIA references; `aria-hidden` on focusable elements; `<details>`/`<summary>`; `<progress>`/`<meter>` without names

**Outputs:** `access_output/ally_{hostname}_report.txt` full audit findings followed by an inventory of external stylesheets, scripts, iframes, fonts, and media.

```cli
python ally.py https://example.com
python ally.py example.com          # https:// prepended automatically
```


### pdftree.py PDF structure tree

Extracts and visualizes the logical structure tree of a tagged PDF. Checks *what structure exists*; pair with `textaccess.py` for usability. Resolves `/RoleMap` entries, shows non-standard tags mapped to their standard equivalents inline. Flags heading-level skips and unmapped tags in the summary block before printing the tree.

**Outputs:**
- to terminal: unicode box-drawing tree with color-coded semantic tags and a summary block
- `access_output/pdftree_{stem}_report.txt` plain-text mirror of terminal output
- `--json` full tree as JSON including resolved tags and actual text

```cli
python pdftree.py document.pdf
python pdftree.py document.pdf --json tree.json
```

### textaccess.py PDF text accessibility

Answers *whether a PDF's content is actually usable* by assistive technology. A document can be tagged and still fail: scanned images with no extractable text; figures without alt text; fonts with broken ToUnicode tables that produce garbled output in screen readers. Also reports comprehensive document metadata.

**Checks include:**
- tagging and language `/Marked`, `/Lang`;
- metadata title, author, subject, keywords, creator/producer tools, dates, PDF version, encryption, XMP/PDF-UA conformance declaration;
- font encoding per-page scan for `/ToUnicode` tables; reports ratio of suspect characters (U+FFFD, private-use-area codepoints) in extracted text;
- extractable text, word count, chars/page, legibility;
- structure inventory single tree walk auditing figures, tables, lists, headings, links, forms, and formulas;
- untagged content pages with no BDC/BMC operators (heuristic-based);
- structure types non-standard tags without a RoleMap entry.

**Output:** `access_output/textaccess_{stem}_report.txt`

```cli
python textaccess.py document.pdf
```

### colors.py perceptually distinct colors

Generates a palette of $n$ colors by maximizing minimum pairwise CIEDE2000 \Delta E in CIELAB space. Supports color vision deficiency simulation (protanopia, deuteranopia, tritanopia), so palettes remain distinguishable across vision types. Anchor colors fix specific values while the algorithm fills remaining slots.

**Outputs:**
- `colors_{prefix}.png` swatch grid
- `colors_{prefix}.txt` hex, RGB, Lab values; \Delta E by vision type; CSS variables
- `colors_{prefix}_colorspace.png` a*b* chromaticity diagram and lightness distribution
- `colors_{prefix}_contrast.png` per-color contrast badges against a reference color (with `--contrast`)

```cli
python colors.py 8 --cvd all
python colors.py 8 --cvd all --contrast #1a1a1a -o my_palette
python colors.py 6 --anchor "#003366" --anchor "#cc0000" --cvd all
```

### harmony.py color harmony palette

Again, not technically an accessibility-focused script; I use it in conjunction with colors.py. Generates complementary, monochromatic, analogous, triadic, and tetradic palettes from a single anchor color. Runs in both LCH (perceptually uniform) and HSL (traditional color wheel) simultaneously by default, placing them side-by-side, so the difference between perceptual and geometric harmony is directly visible.

Harmony types include: complementary (2 colors); monochromatic (5); analogous (5); triadic (3); tetradic (4).

**Outputs:**
- `harmony_{prefix}.png` swatch grid, side-by-side in both mode
- `harmony_{prefix}.txt` hex/RGB/hue values; LCH vs HSL comparison; CSS variables
- `harmony_{prefix}_colorspace.png` polar hue-chroma and Cartesian a*b* plots

```cli
python harmony.py "#3498db"          # both LCH and HSL (default)
python harmony.py "#3498db" --lch    # LCH only
python harmony.py ff6b6b -o warm
```


## Note: browser checking really requires tactile testing

These require a headless browser or manual testing:

- computed color contrast: `ally.py` catches inline-style pairs only; use axe-core or browser DevTools for the rest;
- pages built by React, Vue, Next.js etc., are fetched as near-empty shells; `ally.py` flags this, but cannot audit what JavaScript renders;
- keyboard operability and focus order require live browser interaction;
- the link-purpose check does not traverse `<img alt>` inside a link;
- `ally.py` checks for a `<track>` or adjacent transcript link; off-page transcripts are not detected;
- `textaccess.py` checks that structure exists and is populated, not that reading order matches visual/logical order;
- none of WCAG AAA checked;
- `ally.py` audits one URL per invocation; future work may traverse deeper than the present queried page

## Future Work

- Playwright-based runner for JS-rendered pages and focus-order checks
- computed contrast checking via headless browser CSS evaluation
- multi-URL crawling mode for ally.py
- reading-order validation for PDFs
- WCAG 2.2 criteria (2.4.11, 2.4.12, 2.5.3, 3.2.6)

## Author

**Daniel Quigley**
[dquigleydev@gmail.com](mailto:dquigleydev@gmail.com)
[GitHub](https://github.com/deltaquebec)


## Acknowledgments

Conceptual grounding from the WCAG specification, PDF/UA standard (ISO 14289), and CIEDE2000 color difference formula (Sharma, Wu, Dalal 2005). CVD simulation matrices from Machado et al. (2009). 

---

**Python**: 3.9+ | **Status**: Active development
