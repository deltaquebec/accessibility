#!/usr/bin/env python3
"""
author: Daniel Quigley
contact: dquigleydev@gmail.com

generate color harmony palettes from an anchor color in
CIELAB/LCH color space for perceptual uniformity

produces complementary, monochromatic, analogous, triadic, tetradic schemes.
    python harmony.py "#3498db"
    python harmony.py "#3498db" --lch
    python harmony.py ff6b6b -o warm_palette
"""

import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# D65 illuminant
D65 = np.array([0.95047, 1.0, 1.08883])

# types
class HarmonyType(Enum):
    COMPLEMENTARY = "complementary"
    MONOCHROMATIC = "monochromatic"
    ANALOGOUS     = "analogous"
    TRIADIC       = "triadic"
    TETRADIC      = "tetradic"


@dataclass
class ColorHarmony:
    """color harmony scheme"""
    name: str
    colors_hex: List[str]
    colors_rgb: List[Tuple[int, int, int]]
    colors_lab: List[np.ndarray]
    colors_lch: List[np.ndarray]
    description: str


# conversions
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.abs(c) ** (1/2.4)) - 0.055)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear(rgb)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    return linear @ M.T


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    linear = xyz @ M_inv.T
    return linear_to_srgb(linear)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz_n = xyz / D65
    delta = 6/29
    f = np.where(xyz_n > delta**3, xyz_n ** (1/3), xyz_n / (3 * delta**2) + 4/29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    delta = 6/29
    xyz = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(xyz > delta, xyz**3, 3 * delta**2 * (xyz - 4/29))
    return xyz * D65


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return xyz_to_rgb(lab_to_xyz(lab))


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """convert Lab to LCH (cylindrical coordinates)"""
    L, a, b = lab[0], lab[1], lab[2]
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360
    return np.array([L, C, H])


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    """convert LCH to Lab"""
    L, C, H = lch[0], lch[1], lch[2]
    a = C * np.cos(np.radians(H))
    b = C * np.sin(np.radians(H))
    return np.array([L, a, b])


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """convert RGB (0-1) to HSL"""
    r, g, b = rgb[0], rgb[1], rgb[2]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    L = (max_c + min_c) / 2

    if max_c == min_c:
        H = S = 0
    else:
        d = max_c - min_c
        S = d / (2 - max_c - min_c) if L > 0.5 else d / (max_c + min_c)

        if max_c == r:
            H = (g - b) / d + (6 if g < b else 0)
        elif max_c == g:
            H = (b - r) / d + 2
        else:
            H = (r - g) / d + 4
        H *= 60

    return np.array([H, S * 100, L * 100])


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """convert HSL to RGB (0-1)"""
    H, S, L = hsl[0], hsl[1] / 100, hsl[2] / 100

    if S == 0:
        return np.array([L, L, L])

    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    q = L * (1 + S) if L < 0.5 else L + S - L * S
    p = 2 * L - q
    h = H / 360

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    return np.array([r, g, b])


def hsl_to_hex(hsl: np.ndarray) -> str:
    """convert HSL to hex color code"""
    rgb = hsl_to_rgb(hsl)
    rgb_255 = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return '#{:02x}{:02x}{:02x}'.format(*rgb_255)


def hsl_to_rgb_tuple(hsl: np.ndarray) -> Tuple[int, int, int]:
    """convert HSL to RGB tuple (0-255)"""
    rgb = hsl_to_rgb(hsl)
    return tuple(int(x) for x in (np.clip(rgb, 0, 1) * 255).round())


def is_in_srgb_gamut(lab: np.ndarray, tolerance: float = 1e-4) -> bool:
    """check if Lab color is within sRGB gamut"""
    rgb = lab_to_rgb(lab)
    return np.all((rgb >= -tolerance) & (rgb <= 1 + tolerance))


def gamut_clip_lch(lch: np.ndarray) -> np.ndarray:
    """clip LCH to sRGB gamut by binary-search chroma reduction; hue and lightness preserved"""
    L, C, H = lch[0], lch[1], lch[2]
    L = np.clip(L, 0, 100)

    if C <= 0:
        return np.array([L, 0, H])

    lab = lch_to_lab(np.array([L, C, H]))
    if is_in_srgb_gamut(lab):
        return np.array([L, C, H])

    low, high = 0, C
    for _ in range(20):
        mid = (low + high) / 2
        test_lab = lch_to_lab(np.array([L, mid, H]))
        if is_in_srgb_gamut(test_lab):
            low = mid
        else:
            high = mid

    return np.array([L, low, H])


def lab_to_hex(lab: np.ndarray) -> str:
    """convert Lab to hex color code"""
    rgb = lab_to_rgb(lab)
    rgb_255 = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return '#{:02x}{:02x}{:02x}'.format(*rgb_255)


def lab_to_rgb_tuple(lab: np.ndarray) -> Tuple[int, int, int]:
    """convert Lab to RGB tuple (0-255)"""
    rgb = lab_to_rgb(lab)
    return tuple(int(x) for x in (np.clip(rgb, 0, 1) * 255).round())


# color parsing
def parse_color(color_str: str) -> np.ndarray:
    """parse color string -> Lab; accepts #RRGGBB, RRGGBB, rgb(R,G,B), R,G,B"""
    color_str = color_str.strip()

    hex_match = color_str.lstrip('#')
    if len(hex_match) == 6 and all(c in '0123456789abcdefABCDEF' for c in hex_match):
        r = int(hex_match[0:2], 16) / 255.0
        g = int(hex_match[2:4], 16) / 255.0
        b = int(hex_match[4:6], 16) / 255.0
        return rgb_to_lab(np.array([r, g, b]))

    rgb_str = color_str.lower().replace('rgb', '').replace('(', '').replace(')', '').strip()
    parts = [p.strip() for p in rgb_str.split(',')]
    if len(parts) == 3:
        try:
            r, g, b = [int(p) for p in parts]
            if all(0 <= v <= 255 for v in [r, g, b]):
                return rgb_to_lab(np.array([r/255.0, g/255.0, b/255.0]))
        except ValueError:
            pass

    raise ValueError(f"cannot parse color: '{color_str}'; use #RRGGBB or R,G,B format")


# harmony
def generate_complementary(anchor_lch: np.ndarray, art_mode: bool = False,
                           anchor_hsl: np.ndarray = None) -> ColorHarmony:
    """generate complementary color (180 deg opposite)"""
    if art_mode and anchor_hsl is not None:
        H, S, L = anchor_hsl
        colors_hsl = [
            anchor_hsl,
            np.array([(H + 180) % 360, S, L])
        ]
        colors_hex = [hsl_to_hex(hsl) for hsl in colors_hsl]
        colors_rgb = [hsl_to_rgb_tuple(hsl) for hsl in colors_hsl]
        colors_lab = [rgb_to_lab(hsl_to_rgb(hsl)) for hsl in colors_hsl]
        colors_lch = [lab_to_lch(lab) for lab in colors_lab]
    else:
        L, C, H = anchor_lch
        colors_lch = [
            anchor_lch,
            gamut_clip_lch(np.array([L, C, (H + 180) % 360]))
        ]
        colors_lab = [lch_to_lab(lch) for lch in colors_lch]
        colors_hex = [lab_to_hex(lab) for lab in colors_lab]
        colors_rgb = [lab_to_rgb_tuple(lab) for lab in colors_lab]
    
    return ColorHarmony(
        name="complementary",
        colors_hex=colors_hex,
        colors_rgb=colors_rgb,
        colors_lab=colors_lab,
        colors_lch=colors_lch,
        description="two colors opposite on the color wheel (180 deg apart); high contrast"
    )


def generate_monochromatic(anchor_lch: np.ndarray, n_colors: int = 5,
                           art_mode: bool = False, anchor_hsl: np.ndarray = None) -> ColorHarmony:
    """generate monochromatic palette (same hue, varying L and C/S)"""
    if art_mode and anchor_hsl is not None:
        H, S, L = anchor_hsl
        # vary lightness from light to dark, saturation follows curve
        L_values = np.linspace(85, 20, n_colors)
        S_scale = np.array([0.5, 0.75, 1.0, 0.85, 0.6])
        if n_colors != 5:
            S_scale = np.interp(np.linspace(0, 4, n_colors), np.arange(5), S_scale)
        
        colors_hsl = []
        for i, L_new in enumerate(L_values):
            S_new = min(S * S_scale[i], 100)
            colors_hsl.append(np.array([H, S_new, L_new]))
        
        colors_hex = [hsl_to_hex(hsl) for hsl in colors_hsl]
        colors_rgb = [hsl_to_rgb_tuple(hsl) for hsl in colors_hsl]
        colors_lab = [rgb_to_lab(hsl_to_rgb(hsl)) for hsl in colors_hsl]
        colors_lch = [lab_to_lch(lab) for lab in colors_lab]
    else:
        L, C, H = anchor_lch
        L_values = np.linspace(85, 25, n_colors)
        C_scale = np.array([0.4, 0.7, 1.0, 0.8, 0.5])
        if n_colors != 5:
            C_scale = np.interp(np.linspace(0, 4, n_colors), np.arange(5), C_scale)
        
        colors_lch = []
        for i, L_new in enumerate(L_values):
            C_new = C * C_scale[i]
            lch = gamut_clip_lch(np.array([L_new, C_new, H]))
            colors_lch.append(lch)
        
        colors_lab = [lch_to_lab(lch) for lch in colors_lch]
        colors_hex = [lab_to_hex(lab) for lab in colors_lab]
        colors_rgb = [lab_to_rgb_tuple(lab) for lab in colors_lab]
    
    return ColorHarmony(
        name="monochromatic",
        colors_hex=colors_hex,
        colors_rgb=colors_rgb,
        colors_lab=colors_lab,
        colors_lch=colors_lch,
        description="single hue with varying lightness and saturation; cohesive, subtle"
    )


def generate_analogous(anchor_lch: np.ndarray, angle: float = 30,
                       art_mode: bool = False, anchor_hsl: np.ndarray = None) -> ColorHarmony:
    """generate analogous colors (adjacent hues, +/-30 deg by default)"""
    offsets = [-2*angle, -angle, 0, angle, 2*angle]
    
    if art_mode and anchor_hsl is not None:
        H, S, L = anchor_hsl
        colors_hsl = []
        for offset in offsets:
            # slight lightness variation for visual interest
            L_adj = L + offset * 0.15
            L_adj = np.clip(L_adj, 10, 95)
            colors_hsl.append(np.array([(H + offset) % 360, S, L_adj]))
        
        colors_hex = [hsl_to_hex(hsl) for hsl in colors_hsl]
        colors_rgb = [hsl_to_rgb_tuple(hsl) for hsl in colors_hsl]
        colors_lab = [rgb_to_lab(hsl_to_rgb(hsl)) for hsl in colors_hsl]
        colors_lch = [lab_to_lch(lab) for lab in colors_lab]
    else:
        L, C, H = anchor_lch
        colors_lch = []
        for offset in offsets:
            L_adj = L + offset * 0.15
            L_adj = np.clip(L_adj, 20, 90)
            lch = gamut_clip_lch(np.array([L_adj, C, (H + offset) % 360]))
            colors_lch.append(lch)
        
        colors_lab = [lch_to_lab(lch) for lch in colors_lch]
        colors_hex = [lab_to_hex(lab) for lab in colors_lab]
        colors_rgb = [lab_to_rgb_tuple(lab) for lab in colors_lab]
    
    return ColorHarmony(
        name="analogous",
        colors_hex=colors_hex,
        colors_rgb=colors_rgb,
        colors_lab=colors_lab,
        colors_lch=colors_lch,
        description=f"adjacent hues (+/-{angle} deg); harmonious, natural"
    )


def generate_triadic(anchor_lch: np.ndarray, art_mode: bool = False,
                     anchor_hsl: np.ndarray = None) -> ColorHarmony:
    """generate triadic colors (120 deg apart)"""
    if art_mode and anchor_hsl is not None:
        H, S, L = anchor_hsl
        colors_hsl = [
            anchor_hsl,
            np.array([(H + 120) % 360, S, L]),
            np.array([(H + 240) % 360, S, L])
        ]
        colors_hex = [hsl_to_hex(hsl) for hsl in colors_hsl]
        colors_rgb = [hsl_to_rgb_tuple(hsl) for hsl in colors_hsl]
        colors_lab = [rgb_to_lab(hsl_to_rgb(hsl)) for hsl in colors_hsl]
        colors_lch = [lab_to_lch(lab) for lab in colors_lab]
    else:
        L, C, H = anchor_lch
        colors_lch = [
            anchor_lch,
            gamut_clip_lch(np.array([L, C, (H + 120) % 360])),
            gamut_clip_lch(np.array([L, C, (H + 240) % 360]))
        ]
        colors_lab = [lch_to_lab(lch) for lch in colors_lch]
        colors_hex = [lab_to_hex(lab) for lab in colors_lab]
        colors_rgb = [lab_to_rgb_tuple(lab) for lab in colors_lab]
    
    return ColorHarmony(
        name="triadic",
        colors_hex=colors_hex,
        colors_rgb=colors_rgb,
        colors_lab=colors_lab,
        colors_lch=colors_lch,
        description="three colors evenly spaced (120 deg apart); vibrant, balanced"
    )


def generate_tetradic(anchor_lch: np.ndarray, rectangular: bool = False,
                      art_mode: bool = False, anchor_hsl: np.ndarray = None) -> ColorHarmony:
    """
    generate tetradic colors
      square: 90 deg apart (H, H+90, H+180, H+270)
      rectangular: two complementary pairs (H, H+60, H+180, H+240)
    """
    if rectangular:
        offsets = [0, 60, 180, 240]
        desc = "two complementary pairs (rectangular); rich color palette"
    else:
        offsets = [0, 90, 180, 270]
        desc = "four colors evenly spaced (90 deg apart, square); bold, dynamic"
    
    if art_mode and anchor_hsl is not None:
        H, S, L = anchor_hsl
        colors_hsl = [np.array([(H + off) % 360, S, L]) for off in offsets]
        colors_hex = [hsl_to_hex(hsl) for hsl in colors_hsl]
        colors_rgb = [hsl_to_rgb_tuple(hsl) for hsl in colors_hsl]
        colors_lab = [rgb_to_lab(hsl_to_rgb(hsl)) for hsl in colors_hsl]
        colors_lch = [lab_to_lch(lab) for lab in colors_lab]
    else:
        L, C, H = anchor_lch
        colors_lch = [gamut_clip_lch(np.array([L, C, (H + off) % 360])) for off in offsets]
        colors_lab = [lch_to_lab(lch) for lch in colors_lch]
        colors_hex = [lab_to_hex(lab) for lab in colors_lab]
        colors_rgb = [lab_to_rgb_tuple(lab) for lab in colors_lab]
    
    return ColorHarmony(
        name="tetradic" + (" (rectangular)" if rectangular else " (square)"),
        colors_hex=colors_hex,
        colors_rgb=colors_rgb,
        colors_lab=colors_lab,
        colors_lch=colors_lch,
        description=desc
    )


# output
def generate_palette_image(harmonies: Dict[str, ColorHarmony], anchor_hex: str,
                           output_path: str, mode_label: str = None) -> None:
    """generate PNG showing all harmony palettes for single mode"""
    if Image is None:
        print("warning: pillow not installed; skipping PNG generation", file=sys.stderr)
        return
    
    swatch_w, swatch_h = 80, 60
    padding = 15
    label_h = 25
    section_gap = 30
    
    # calculate dimensions
    max_colors = max(len(h.colors_hex) for h in harmonies.values())
    row_width = max_colors * swatch_w + (max_colors + 1) * padding
    
    n_sections = len(harmonies)
    total_height = (padding + label_h + swatch_h + padding) * n_sections + section_gap * (n_sections - 1) + 80
    
    img = Image.new('RGB', (row_width + 40, total_height), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    # font loading
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font
        font_title = font
    
    title = f"color harmonies for {anchor_hex.upper()}"
    if mode_label:
        title += f" ({mode_label})"
    draw.text((20, 15), title, fill=(40, 40, 40), font=font_title)
    
    y_offset = 55
    
    for name, harmony in harmonies.items():
        # section label
        draw.text((20, y_offset), harmony.name, fill=(60, 60, 60), font=font)
        y_offset += label_h
        
        # swatches
        for i, (hex_col, rgb_col) in enumerate(zip(harmony.colors_hex, harmony.colors_rgb)):
            x = 20 + i * (swatch_w + padding)
            
            # highlight anchor with border
            is_anchor = (hex_col.lower() == anchor_hex.lower())
            border_color = (40, 40, 40) if is_anchor else (200, 200, 200)
            border_width = 3 if is_anchor else 1
            
            draw.rectangle([x, y_offset, x + swatch_w, y_offset + swatch_h],
                          fill=rgb_col, outline=border_color, width=border_width)
            
            # hex label below
            text_bbox = draw.textbbox((0, 0), hex_col.upper(), font=font_small)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text((x + (swatch_w - text_w) // 2, y_offset + swatch_h + 3),
                     hex_col.upper(), fill=(80, 80, 80), font=font_small)
        
        y_offset += swatch_h + 35 + section_gap
    
    img.save(output_path)


def generate_comparison_image(harmonies_lch: Dict[str, ColorHarmony],
                               harmonies_hsl: Dict[str, ColorHarmony],
                               anchor_hex: str, output_path: str) -> None:
    """side-by-side LCH vs HSL comparison PNG"""
    if Image is None:
        print("warning: pillow not installed; skipping PNG generation", file=sys.stderr)
        return
    
    swatch_w, swatch_h = 70, 50
    padding = 10
    label_h = 22
    section_gap = 20
    column_gap = 40
    
    # calculate dimensions
    max_colors = max(
        max(len(h.colors_hex) for h in harmonies_lch.values()),
        max(len(h.colors_hex) for h in harmonies_hsl.values())
    )
    col_width = max_colors * swatch_w + (max_colors + 1) * padding
    total_width = 2 * col_width + column_gap + 60
    
    n_sections = len(harmonies_lch)
    row_height = label_h + swatch_h + 25
    total_height = row_height * n_sections + section_gap * (n_sections - 1) + 100
    
    img = Image.new('RGB', (total_width, total_height), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    # font loading
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font
        font_title = font
        font_header = font
    
    draw.text((20, 12), f"color harmonies for {anchor_hex.upper()}", fill=(40, 40, 40), font=font_title)

    lch_x = 30
    hsl_x = lch_x + col_width + column_gap
    header_y = 45

    draw.text((lch_x, header_y), "lch (perceptual)", fill=(80, 80, 80), font=font_header)
    draw.text((hsl_x, header_y), "hsl (traditional)", fill=(80, 80, 80), font=font_header)
    
    # draw vertical divider
    div_x = lch_x + col_width + column_gap // 2
    draw.line([(div_x, header_y), (div_x, total_height - 20)], fill=(200, 200, 200), width=1)
    
    y_offset = 75
    
    harmony_names = list(harmonies_lch.keys())
    
    for name in harmony_names:
        h_lch = harmonies_lch[name]
        h_hsl = harmonies_hsl[name]
        
        # section label (centered above both columns)
        draw.text((lch_x, y_offset), h_lch.name, fill=(60, 60, 60), font=font)
        y_offset += label_h
        
        # LCH swatches
        for i, (hex_col, rgb_col) in enumerate(zip(h_lch.colors_hex, h_lch.colors_rgb)):
            x = lch_x + i * (swatch_w + padding)
            is_anchor = (hex_col.lower() == anchor_hex.lower())
            border_color = (40, 40, 40) if is_anchor else (200, 200, 200)
            border_width = 2 if is_anchor else 1
            
            draw.rectangle([x, y_offset, x + swatch_w, y_offset + swatch_h],
                          fill=rgb_col, outline=border_color, width=border_width)
            
            text_bbox = draw.textbbox((0, 0), hex_col.upper(), font=font_small)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text((x + (swatch_w - text_w) // 2, y_offset + swatch_h + 2),
                     hex_col.upper(), fill=(100, 100, 100), font=font_small)
        
        # HSL swatches
        for i, (hex_col, rgb_col) in enumerate(zip(h_hsl.colors_hex, h_hsl.colors_rgb)):
            x = hsl_x + i * (swatch_w + padding)
            is_anchor = (hex_col.lower() == anchor_hex.lower())
            border_color = (40, 40, 40) if is_anchor else (200, 200, 200)
            border_width = 2 if is_anchor else 1
            
            draw.rectangle([x, y_offset, x + swatch_w, y_offset + swatch_h],
                          fill=rgb_col, outline=border_color, width=border_width)
            
            text_bbox = draw.textbbox((0, 0), hex_col.upper(), font=font_small)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text((x + (swatch_w - text_w) // 2, y_offset + swatch_h + 2),
                     hex_col.upper(), fill=(100, 100, 100), font=font_small)
        
        y_offset += swatch_h + 25 + section_gap
    
    img.save(output_path)


def generate_colorspace_plot(harmonies: Dict[str, ColorHarmony], anchor_hex: str,
                              output_path: str, mode_label: str = None) -> None:
    """Lab color space visualization of harmonies"""
    if not HAS_MATPLOTLIB:
        print("warning: matplotlib not installed; skipping color space plot", file=sys.stderr)
        return
    
    fig = plt.figure(figsize=(14, 6))
    
    # left: a*b* chromaticity with all harmonies
    ax1 = fig.add_subplot(121, projection='polar')
    
    # draw hue wheel background
    theta = np.linspace(0, 2*np.pi, 360)
    for i in range(len(theta)-1):
        lch = gamut_clip_lch(np.array([65, 50, i]))
        lab = lch_to_lab(lch)
        ax1.fill_between([theta[i], theta[i+1]], 0, 85, color=lab_to_hex(lab), alpha=0.15)
    
    # plot each harmony with different markers
    markers = {'complementary': 'o', 'monochromatic': 's', 'analogous': '^',
               'triadic': 'D', 'tetradic': 'p'}
    
    for name, harmony in harmonies.items():
        for i, (lch, hex_col) in enumerate(zip(harmony.colors_lch, harmony.colors_hex)):
            H_rad = np.radians(lch[2])
            C = lch[1]
            
            is_anchor = (hex_col.lower() == anchor_hex.lower())
            edge_color = 'black' if is_anchor else 'white'
            size = 200 if is_anchor else 120
            
            ax1.scatter(H_rad, C, c=hex_col, s=size, marker=markers.get(name, 'o'),
                       edgecolors=edge_color, linewidths=1.5, zorder=3, label=name if i == 0 else "")
    
    ax1.set_ylim(0, 100)
    title = 'hue-chroma (polar)'
    if mode_label:
        title += f' - {mode_label}'
    ax1.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax1.set_rticks([25, 50, 75])
    ax1.set_rlabel_position(45)
    ax1.grid(True, alpha=0.3)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right', 
               bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    # right: a*b* Cartesian with connecting lines
    ax2 = fig.add_subplot(122)
    
    # gamut background
    gamut_points = []
    for r in np.linspace(0, 1, 25):
        for g in np.linspace(0, 1, 25):
            for b in np.linspace(0, 1, 25):
                if r in [0, 1] or g in [0, 1] or b in [0, 1]:
                    gamut_points.append([r, g, b])
    gamut_rgb = np.array(gamut_points)
    gamut_lab = np.array([rgb_to_lab(rgb) for rgb in gamut_rgb])
    ax2.scatter(gamut_lab[:, 1], gamut_lab[:, 2], c='#e8e8e8', s=1, alpha=0.5, zorder=1)
    
    # harmonies with connecting lines
    line_styles = {'complementary': '-', 'monochromatic': ':', 'analogous': '--',
                   'triadic': '-', 'tetradic': '-'}
    line_colors = {'complementary': '#666', 'monochromatic': '#999', 'analogous': '#888',
                   'triadic': '#555', 'tetradic': '#444'}
    
    for name, harmony in harmonies.items():
        a_vals = [lab[1] for lab in harmony.colors_lab]
        b_vals = [lab[2] for lab in harmony.colors_lab]
        
        # connect points (close polygon for triadic/tetradic)
        if name in ['triadic', 'tetradic', 'complementary']:
            a_vals_closed = a_vals + [a_vals[0]]
            b_vals_closed = b_vals + [b_vals[0]]
            ax2.plot(a_vals_closed, b_vals_closed, line_styles[name], 
                    color=line_colors[name], linewidth=1.5, alpha=0.6, zorder=2)
        
        # points
        for i, (lab, hex_col) in enumerate(zip(harmony.colors_lab, harmony.colors_hex)):
            is_anchor = (hex_col.lower() == anchor_hex.lower())
            edge_color = 'black' if is_anchor else 'white'
            size = 180 if is_anchor else 100
            
            ax2.scatter(lab[1], lab[2], c=hex_col, s=size, marker=markers.get(name, 'o'),
                       edgecolors=edge_color, linewidths=1.5, zorder=4)
    
    ax2.axhline(y=0, color='#ccc', linestyle='-', linewidth=0.5, zorder=0)
    ax2.axvline(x=0, color='#ccc', linestyle='-', linewidth=0.5, zorder=0)
    ax2.set_xlabel('a* (green <-> red)', fontsize=11)
    ax2.set_ylabel('b* (blue <-> yellow)', fontsize=11)
    ax2.set_title('chromaticity (a*b* plane)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-100, 100)
    ax2.set_ylim(-100, 100)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_comparison_plot(harmonies_lch: Dict[str, ColorHarmony],
                              harmonies_hsl: Dict[str, ColorHarmony],
                              anchor_hex: str, output_path: str) -> None:
    """side-by-side Lab color space comparison of LCH and HSL"""
    if not HAS_MATPLOTLIB:
        print("warning: matplotlib not installed; skipping color space plot", file=sys.stderr)
        return
    
    fig = plt.figure(figsize=(14, 6))
    
    markers = {'complementary': 'o', 'monochromatic': 's', 'analogous': '^',
               'triadic': 'D', 'tetradic': 'p'}
    
    for idx, (harmonies, label) in enumerate([(harmonies_lch, 'lch (perceptual)'),
                                               (harmonies_hsl, 'hsl (traditional)')]):
        ax = fig.add_subplot(1, 2, idx + 1)
        
        # gamut background
        gamut_points = []
        for r in np.linspace(0, 1, 20):
            for g in np.linspace(0, 1, 20):
                for b in np.linspace(0, 1, 20):
                    if r in [0, 1] or g in [0, 1] or b in [0, 1]:
                        gamut_points.append([r, g, b])
        gamut_rgb = np.array(gamut_points)
        gamut_lab = np.array([rgb_to_lab(rgb) for rgb in gamut_rgb])
        ax.scatter(gamut_lab[:, 1], gamut_lab[:, 2], c='#e8e8e8', s=1, alpha=0.5, zorder=1)
        
        line_styles = {'complementary': '-', 'monochromatic': ':', 'analogous': '--',
                       'triadic': '-', 'tetradic': '-'}
        line_colors = {'complementary': '#666', 'monochromatic': '#999', 'analogous': '#888',
                       'triadic': '#555', 'tetradic': '#444'}
        
        for name, harmony in harmonies.items():
            a_vals = [lab[1] for lab in harmony.colors_lab]
            b_vals = [lab[2] for lab in harmony.colors_lab]
            
            if name in ['triadic', 'tetradic', 'complementary']:
                a_vals_closed = a_vals + [a_vals[0]]
                b_vals_closed = b_vals + [b_vals[0]]
                ax.plot(a_vals_closed, b_vals_closed, line_styles[name], 
                       color=line_colors[name], linewidth=1.5, alpha=0.6, zorder=2)
            
            for i, (lab, hex_col) in enumerate(zip(harmony.colors_lab, harmony.colors_hex)):
                is_anchor = (hex_col.lower() == anchor_hex.lower())
                edge_color = 'black' if is_anchor else 'white'
                size = 180 if is_anchor else 100
                
                ax.scatter(lab[1], lab[2], c=hex_col, s=size, marker=markers.get(name, 'o'),
                          edgecolors=edge_color, linewidths=1.5, zorder=4,
                          label=name if i == 0 and idx == 0 else "")
        
        ax.axhline(y=0, color='#ccc', linestyle='-', linewidth=0.5, zorder=0)
        ax.axvline(x=0, color='#ccc', linestyle='-', linewidth=0.5, zorder=0)
        ax.set_xlabel('a* (green <-> red)', fontsize=10)
        ax.set_ylabel('b* (blue <-> yellow)', fontsize=10)
        ax.set_title(label.lower(), fontsize=12, fontweight='bold')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    handles, labels = fig.axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', 
               bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_results_txt(harmonies: Dict[str, ColorHarmony], anchor_color: str,
                          anchor_hex: str, anchor_lch: np.ndarray, output_path: str,
                          mode: str = 'lch', anchor_hsl: np.ndarray = None) -> None:
    """harmony results txt; single mode"""
    lines = []
    lines.append("\n")
    lines.append("color harmony palettes")
    lines.append(f"anchor color: {anchor_color} to {anchor_hex.upper()}")
    if mode == 'hsl' and anchor_hsl is not None:
        lines.append(f"hsl values: H={anchor_hsl[0]:.1f} deg, S={anchor_hsl[1]:.1f}%, L={anchor_hsl[2]:.1f}%")
        lines.append("mode: traditional (HSL)")
    else:
        lines.append(f"lch values: L={anchor_lch[0]:.1f}, C={anchor_lch[1]:.1f}, H={anchor_lch[2]:.1f} deg")
        lines.append("mode: perceptual (CIELAB/LCH)")
    lines.append("")
    
    for name, harmony in harmonies.items():
        lines.append(f"{harmony.name}")
        lines.append(f"\t{harmony.description}")
        lines.append("")
        
        for i, (h, rgb, lab, lch) in enumerate(zip(
            harmony.colors_hex, harmony.colors_rgb, harmony.colors_lab, harmony.colors_lch
        )):
            marker = " [anchor]" if h.lower() == anchor_hex.lower() else ""
            lines.append(f"\t{i+1}. {h.upper()}  RGB({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})  "
                        f"H={lch[2]:5.1f} deg{marker}")
        
        lines.append(f"\n\thex list: {harmony.colors_hex}")
        lines.append("")
    
    lines.append("\n")
    lines.append("css variables")
    lines.append("\n")
    for name, harmony in harmonies.items():
        lines.append(f"\n/* {harmony.name} */")
        for i, h in enumerate(harmony.colors_hex):
            lines.append(f"--{name}-{i+1}: {h};")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_comparison_txt(harmonies_lch: Dict[str, ColorHarmony],
                             harmonies_hsl: Dict[str, ColorHarmony],
                             anchor_color: str, anchor_hex: str,
                             anchor_lch: np.ndarray, anchor_hsl: np.ndarray,
                             output_path: str) -> None:
    """comparison txt; LCH vs HSL harmonies"""
    lines = []

    lines.append("color harmony palettes comparison")
    lines.append("\n")
    lines.append(f"anchor color: {anchor_color} to {anchor_hex.upper()}")
    lines.append(f"lch: L={anchor_lch[0]:.1f}, C={anchor_lch[1]:.1f}, H={anchor_lch[2]:.1f} deg")
    lines.append(f"hsl: H={anchor_hsl[0]:.1f} deg, S={anchor_hsl[1]:.1f}%, L={anchor_hsl[2]:.1f}%")
    lines.append("")

    harmony_names = list(harmonies_lch.keys())

    for name in harmony_names:
        h_lch = harmonies_lch[name]
        h_hsl = harmonies_hsl[name]

        lines.append("\n")
        lines.append(f"{h_lch.name}")
        lines.append(f"\t{h_lch.description}")
        lines.append("")

        lines.append(f"\t{'lch (perceptual)':<40} {'hsl (traditional)':<40}")

        max_colors = max(len(h_lch.colors_hex), len(h_hsl.colors_hex))
        for i in range(max_colors):
            lch_part = ""
            hsl_part = ""

            if i < len(h_lch.colors_hex):
                h = h_lch.colors_hex[i]
                rgb = h_lch.colors_rgb[i]
                marker = "*" if h.lower() == anchor_hex.lower() else " "
                lch_part = f"{marker}{h.upper()} RGB({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d})"

            if i < len(h_hsl.colors_hex):
                h = h_hsl.colors_hex[i]
                rgb = h_hsl.colors_rgb[i]
                marker = "*" if h.lower() == anchor_hex.lower() else " "
                hsl_part = f"{marker}{h.upper()} RGB({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d})"

            lines.append(f"\t{i+1}. {lch_part:<37}   {i+1}. {hsl_part:<37}")

        lines.append("")
        lines.append(f"\tlch hex: {h_lch.colors_hex}")
        lines.append(f"\thsl hex: {h_hsl.colors_hex}")
        lines.append("")

    lines.append("\n")
    lines.append("css variables")
    lines.append("\n")

    lines.append("\n/* lch (perceptual) */")
    for name, harmony in harmonies_lch.items():
        for i, h in enumerate(harmony.colors_hex):
            lines.append(f"--lch-{name}-{i+1}: {h};")

    lines.append("\n/* hsl (traditional) */")
    for name, harmony in harmonies_hsl.items():
        for i, h in enumerate(harmony.colors_hex):
            lines.append(f"--hsl-{name}-{i+1}: {h};")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_harmonies(anchor_color: str, output_prefix: str = "harmonies",
                       verbose: bool = True, no_png: bool = False,
                       no_txt: bool = False, no_plot: bool = False,
                       mode: str = 'both') -> Dict[str, Dict[str, ColorHarmony]]:
    """generate color harmonies from an anchor color"""
    anchor_lab = parse_color(anchor_color)
    anchor_lch = lab_to_lch(anchor_lab)
    anchor_hex = lab_to_hex(anchor_lab)
    anchor_rgb = lab_to_rgb(anchor_lab)
    anchor_hsl = rgb_to_hsl(anchor_rgb)
    
    if verbose:
        print()
        print(f"\tanchor: {anchor_color} to {anchor_hex}")
        print(f"\tlch: L={anchor_lch[0]:.1f}, C={anchor_lch[1]:.1f}, H={anchor_lch[2]:.1f} deg")
        print(f"\thsl: H={anchor_hsl[0]:.1f} deg, S={anchor_hsl[1]:.1f}%, L={anchor_hsl[2]:.1f}%")
        print(f"\tmode: {mode}\n")
    
    result = {}
    
    # generate LCH harmonies
    if mode in ('lch', 'both'):
        harmonies_lch = {
            'complementary': generate_complementary(anchor_lch, art_mode=False),
            'monochromatic': generate_monochromatic(anchor_lch, art_mode=False),
            'analogous': generate_analogous(anchor_lch, art_mode=False),
            'triadic': generate_triadic(anchor_lch, art_mode=False),
            'tetradic': generate_tetradic(anchor_lch, art_mode=False),
        }
        result['lch'] = harmonies_lch
        
        if verbose and mode == 'lch':
            print("\tlch (perceptual):")
            for name, harmony in harmonies_lch.items():
                print(f"\t{harmony.name}: {harmony.colors_hex}")
    
    # generate HSL harmonies
    if mode in ('hsl', 'both'):
        harmonies_hsl = {
            'complementary': generate_complementary(anchor_lch, art_mode=True, anchor_hsl=anchor_hsl),
            'monochromatic': generate_monochromatic(anchor_lch, art_mode=True, anchor_hsl=anchor_hsl),
            'analogous': generate_analogous(anchor_lch, art_mode=True, anchor_hsl=anchor_hsl),
            'triadic': generate_triadic(anchor_lch, art_mode=True, anchor_hsl=anchor_hsl),
            'tetradic': generate_tetradic(anchor_lch, art_mode=True, anchor_hsl=anchor_hsl),
        }
        result['hsl'] = harmonies_hsl
        
        if verbose and mode == 'hsl':
            print("\thsl (traditional):")
            for name, harmony in harmonies_hsl.items():
                print(f"\t{harmony.name}: {harmony.colors_hex}")
    
    # print both side by side
    if verbose and mode == 'both':
        print(f"\t{'lch (perceptual)':<45} {'hsl (traditional)':<45}")
        print()
        for name in harmonies_lch.keys():
            lch_hex = harmonies_lch[name].colors_hex
            hsl_hex = harmonies_hsl[name].colors_hex
            label = harmonies_lch[name].name
            print(f"{label}:")
            print(f"\t{str(lch_hex):<43} {str(hsl_hex):<43}")
            print()
    
    # generate output files
    if mode == 'both':
        if not no_png:
            png_path = f"{output_prefix}.png"
            generate_comparison_image(harmonies_lch, harmonies_hsl, anchor_hex, png_path)
            if verbose:
                print(f"\tcomparison image saved: {png_path}")
        
        if not no_txt:
            txt_path = f"{output_prefix}.txt"
            generate_comparison_txt(harmonies_lch, harmonies_hsl, anchor_color, 
                                    anchor_hex, anchor_lch, anchor_hsl, txt_path)
            if verbose:
                print(f"\tresults saved: {txt_path}")
        
        if not no_plot:
            plot_path = f"{output_prefix}_colorspace.png"
            generate_comparison_plot(harmonies_lch, harmonies_hsl, anchor_hex, plot_path)
            if verbose:
                print(f"\tcolor space plot saved: {plot_path}")
    
    else:
        harmonies = result.get('lch') or result.get('hsl')
        mode_label = 'lch' if mode == 'lch' else 'hsl'
        
        if not no_png:
            png_path = f"{output_prefix}.png"
            generate_palette_image(harmonies, anchor_hex, png_path, mode_label=mode_label)
            if verbose:
                print(f"\tpalette image saved: {png_path}")
        
        if not no_txt:
            txt_path = f"{output_prefix}.txt"
            generate_results_txt(harmonies, anchor_color, anchor_hex, anchor_lch, txt_path,
                                mode=mode, anchor_hsl=anchor_hsl)
            if verbose:
                print(f"\tresults saved: {txt_path}")
        
        if not no_plot:
            plot_path = f"{output_prefix}_colorspace.png"
            generate_colorspace_plot(harmonies, anchor_hex, plot_path, mode_label=mode_label)
            if verbose:
                print(f"\tcolor space plot saved: {plot_path}")
    
    return result


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='generate color harmony palettes from an anchor color',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
harmony types generated:
	complementary: 180 deg opposite (2 colors)
	monochromatic: same hue, varying L/C (5 colors)
	analogous: adjacent hues +/-30 deg (5 colors)
	triadic: 120 deg apart (3 colors)
	tetradic: 90 deg apart, square (4 colors)

color modes:
	(default)  both LCH and HSL, side-by-side comparison
	--lch: perceptually uniform (CIELAB/LCH only)
	--hsl: traditional artist's color wheel (HSL only)

color formats accepted:
	#RRGGBB, RRGGBB, R,G,B, rgb(R,G,B)

examples:
	%(prog)s "#3498db"              # both modes (default)
	%(prog)s "#3498db" --lch        # LCH only
	%(prog)s "#3498db" --hsl        # HSL only
	%(prog)s ff6b6b -o warm_palette
	%(prog)s "#2c3e50" --no-plot
        """)

    parser.add_argument('anchor', type=str, help='anchor color (#RRGGBB or R,G,B)')
    parser.add_argument('-o', '--output', type=str, default='harmonies',
                        help='output filename prefix (default: harmonies)')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--lch', action='store_true',
                            help='use only LCH (perceptual) color space')
    mode_group.add_argument('--hsl', action='store_true',
                            help='use only HSL (traditional) color space')

    parser.add_argument('--quiet', '-q', action='store_true', help='suppress output')
    parser.add_argument('--no-png',  action='store_true', help='skip PNG generation')
    parser.add_argument('--no-txt',  action='store_true', help='skip TXT generation')
    parser.add_argument('--no-plot', action='store_true', help='skip color space plot')

    args = parser.parse_args()

    output_dir = Path("access_output")
    output_dir.mkdir(exist_ok=True)
    out = str(output_dir / f"harmony_{args.output}")

    if args.lch:
        mode = 'lch'
    elif args.hsl:
        mode = 'hsl'
    else:
        mode = 'both'

    try:
        generate_harmonies(
            args.anchor,
            output_prefix=out,
            verbose=not args.quiet,
            no_png=args.no_png,
            no_txt=args.no_txt,
            no_plot=args.no_plot,
            mode=mode
        )
    except ValueError as e:
        sys.exit(f"error: {e}")
