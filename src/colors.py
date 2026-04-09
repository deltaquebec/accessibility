#!/usr/bin/env python3
"""
author: Daniel Quigley
contact: dquigleydev@gmail.com

generate n perceptually distinct colors using cielab color space:
    maximize minimum pairwise delta-e (ciede2000) within srgb gamut

    python colors.py 8
    python colors.py 8 --cvd all -o my_palette
    python colors.py 6 --anchor #FF0000 --anchor 255,200,0
    python colors.py 8 --contrast #ffffff
"""

import sys
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# sRGB, XYZ, Lab conversions (D65 illuminant)
D65 = np.array([0.95047, 1.0, 1.08883])


# color vision deficiency
class CVD(Enum):
    """color vision deficiency types"""
    NONE        = "none"
    PROTANOPIA  = "protan"    # red-blind
    DEUTERANOPIA = "deutan"   # green-blind (most common)
    TRITANOPIA  = "tritan"    # blue-blind


# see Machado et al. (2009): simulation matrices for severe (1.0) CVD, transform linear sRGB to simulated linear sRG.
CVD_MATRICES = {
    CVD.NONE: np.eye(3),
    CVD.PROTANOPIA: np.array([
        [ 0.152286,  1.052583, -0.204868],
        [ 0.114503,  0.786281,  0.099216],
        [-0.003882, -0.048116,  1.051998],
    ]),
    CVD.DEUTERANOPIA: np.array([
        [ 0.367322,  0.860646, -0.227968],
        [ 0.280085,  0.672501,  0.047414],
        [-0.011820,  0.042940,  0.968881],
    ]),
    CVD.TRITANOPIA: np.array([
        [ 1.255528, -0.076749, -0.178779],
        [-0.078411,  0.930809,  0.147602],
        [ 0.004733,  0.691367,  0.303900],
    ]),
}


# conversions
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB gamma expansion"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """sRGB gamma compression"""
    with np.errstate(invalid='ignore'):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.abs(c) ** (1/2.4)) - 0.055)


def simulate_cvd(rgb: np.ndarray, cvd_type: CVD) -> np.ndarray:
    """simulate how color appears to someone with CVD"""
    if cvd_type == CVD.NONE:
        return rgb
    linear = srgb_to_linear(rgb)
    simulated_linear = linear @ CVD_MATRICES[cvd_type].T
    return linear_to_srgb(simulated_linear)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """sRGB to XYZ (D65)"""
    linear = srgb_to_linear(rgb)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return linear @ M.T


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    """XYZ (D65) to sRGB"""
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    linear = xyz @ M_inv.T
    return linear_to_srgb(linear)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """XYZ to CIELAB"""
    xyz_n = xyz / D65
    delta = 6/29
    f = np.where(xyz_n > delta**3, xyz_n ** (1/3), xyz_n / (3 * delta**2) + 4/29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """CIELAB to XYZ"""
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


def is_in_srgb_gamut(lab: np.ndarray, tolerance: float = 1e-4) -> np.ndarray:
    """check if Lab color is within sRGB gamut"""
    rgb = lab_to_rgb(lab)
    return np.all((rgb >= -tolerance) & (rgb <= 1 + tolerance), axis=-1)


# delta-e; Sharma, Wu, Dalal (2005)
def ciede2000(lab1: np.ndarray, lab2: np.ndarray,
              kL: float = 1, kC: float = 1, kH: float = 1) -> float:
    """CIEDE2000 color difference formula"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2

    G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))

    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = 0.0
    if C1_prime * C2_prime != 0:
        dh = h2_prime - h1_prime
        if dh > 180:
            dh_prime = dh - 360
        elif dh < -180:
            dh_prime = dh + 360
        else:
            dh_prime = dh

    dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))

    L_avg = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2

    h_avg_prime = 0.0
    if C1_prime * C2_prime != 0:
        if abs(h1_prime - h2_prime) <= 180:
            h_avg_prime = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            h_avg_prime = (h1_prime + h2_prime + 360) / 2
        else:
            h_avg_prime = (h1_prime + h2_prime - 360) / 2

    T = (1
         - 0.17 * np.cos(np.radians(h_avg_prime - 30))
         + 0.24 * np.cos(np.radians(2 * h_avg_prime))
         + 0.32 * np.cos(np.radians(3 * h_avg_prime + 6))
         - 0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))

    dTheta = 30 * np.exp(-((h_avg_prime - 275) / 25)**2)

    R_C = 2 * np.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))

    S_L = 1 + (0.015 * (L_avg - 50)**2) / np.sqrt(20 + (L_avg - 50)**2)
    S_C = 1 + 0.045 * C_avg_prime
    S_H = 1 + 0.015 * C_avg_prime * T

    R_T = -np.sin(np.radians(2 * dTheta)) * R_C

    return np.sqrt(
        (dL_prime / (kL * S_L))**2 +
        (dC_prime / (kC * S_C))**2 +
        (dH_prime / (kH * S_H))**2 +
        R_T * (dC_prime / (kC * S_C)) * (dH_prime / (kH * S_H))
    )


def min_delta_e(colors_lab: np.ndarray, cvd_types: List[CVD] = None) -> float:
    """minimum pairwise delta-e among colors across all specified vision types"""
    if cvd_types is None:
        cvd_types = [CVD.NONE]

    n = len(colors_lab)
    if n < 2:
        return float('inf')

    min_de = float('inf')

    for cvd in cvd_types:
        if cvd == CVD.NONE:
            labs = colors_lab
        else:
            rgbs = np.array([lab_to_rgb(lab) for lab in colors_lab])
            rgbs_sim = np.array([simulate_cvd(rgb, cvd) for rgb in rgbs])
            rgbs_sim = np.clip(rgbs_sim, 0, 1)
            labs = np.array([rgb_to_lab(rgb) for rgb in rgbs_sim])

        for i in range(n):
            for j in range(i + 1, n):
                de = ciede2000(labs[i], labs[j])
                if de < min_de:
                    min_de = de

    return min_de


# candidate generation and selection
def generate_candidate_grid(L_range=(25, 90), a_range=(-80, 80), b_range=(-80, 80),
                            step=8) -> np.ndarray:
    """generate grid of Lab colors within sRGB gamut"""
    L_vals = np.arange(L_range[0], L_range[1] + 1, step)
    a_vals = np.arange(a_range[0], a_range[1] + 1, step)
    b_vals = np.arange(b_range[0], b_range[1] + 1, step)

    grid = np.array(np.meshgrid(L_vals, a_vals, b_vals)).T.reshape(-1, 3)
    mask = is_in_srgb_gamut(grid)
    return grid[mask]


def get_min_distance_to_selected(candidate_lab: np.ndarray,
                                  selected_labs: List[np.ndarray],
                                  cvd_types: List[CVD]) -> float:
    """minimum delta-e from candidate to any selected color, across all CVD types"""
    min_dist = float('inf')
    cand_rgb = lab_to_rgb(candidate_lab)

    for cvd in cvd_types:
        if cvd == CVD.NONE:
            cand_lab_sim = candidate_lab
        else:
            cand_rgb_sim = np.clip(simulate_cvd(cand_rgb, cvd), 0, 1)
            cand_lab_sim = rgb_to_lab(cand_rgb_sim)

        for sel_lab in selected_labs:
            sel_rgb = lab_to_rgb(sel_lab)
            if cvd == CVD.NONE:
                sel_lab_sim = sel_lab
            else:
                sel_rgb_sim = np.clip(simulate_cvd(sel_rgb, cvd), 0, 1)
                sel_lab_sim = rgb_to_lab(sel_rgb_sim)

            dist = ciede2000(cand_lab_sim, sel_lab_sim)
            if dist < min_dist:
                min_dist = dist

    return min_dist


def greedy_select(candidates: np.ndarray, n: int, cvd_types: List[CVD],
                  anchor_labs: List[np.ndarray] = None,
                  verbose: bool = True) -> np.ndarray:
    """
    greedy farthest-point selection
        candidates:  Lab color candidates
        n:           total colors to return (including anchors)
        cvd_types:   CVD types to optimize for
        anchor_labs: pre-selected mandatory colors
        verbose:     show progress bar
    """
    if n <= 0:
        return np.array([])

    selected_labs = []
    selected_indices = []

    if anchor_labs is not None and len(anchor_labs) > 0:
        for anchor in anchor_labs:
            selected_labs.append(anchor)
        colors_to_select = n - len(anchor_labs)
    else:
        mid_gray = np.array([50, 0, 0])
        dists = np.linalg.norm(candidates - mid_gray, axis=1)
        first_idx = int(np.argmin(dists))
        selected_indices.append(first_idx)
        selected_labs.append(candidates[first_idx])
        colors_to_select = n - 1

    if colors_to_select <= 0:
        return np.array(selected_labs[:n])

    remaining = set(range(len(candidates))) - set(selected_indices)

    iterator = range(colors_to_select)
    if verbose and tqdm is not None:
        iterator = tqdm(iterator, desc="selecting colors", unit="color",
                        initial=len(selected_labs), total=n)

    for _ in iterator:
        if not remaining:
            break

        best_idx = None
        best_min_dist = -1

        for idx in remaining:
            min_dist = get_min_distance_to_selected(candidates[idx], selected_labs, cvd_types)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx

        selected_indices.append(best_idx)
        selected_labs.append(candidates[best_idx])
        remaining.remove(best_idx)

        if verbose and tqdm is not None:
            iterator.set_postfix({"min_dE": f"{best_min_dist:.1f}"})

    return np.array(selected_labs)


# color parsing, serialization
def lab_to_hex(lab: np.ndarray) -> str:
    """convert Lab to hex color code"""
    rgb = lab_to_rgb(lab)
    rgb_255 = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return '#{:02x}{:02x}{:02x}'.format(*rgb_255)


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


# output image
def generate_palette_image(hex_colors: List[str], rgb_colors: List[Tuple[int, int, int]],
                           output_path: str, swatch_width: int = 120, swatch_height: int = 80,
                           show_labels: bool = True) -> None:
    """generate PNG palette image"""
    if Image is None:
        print("warning: pillow not installed; skipping png generation", file=sys.stderr)
        return

    n = len(hex_colors)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    padding = 10
    label_height = 40 if show_labels else 0

    img_width  = cols * swatch_width + (cols + 1) * padding
    img_height = rows * (swatch_height + label_height) + (rows + 1) * padding

    img = Image.new('RGB', (img_width, img_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # font loading
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (hex_col, rgb_col) in enumerate(zip(hex_colors, rgb_colors)):
        row = i // cols
        col = i % cols

        x = padding + col * (swatch_width + padding)
        y = padding + row * (swatch_height + label_height + padding)

        draw.rectangle([x, y, x + swatch_width, y + swatch_height],
                       fill=rgb_col, outline=(200, 200, 200))

        if show_labels:
            label = hex_col.upper()
            text_bbox  = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (swatch_width - text_width) // 2
            text_y = y + swatch_height + 5
            draw.text((text_x, text_y), label, fill=(60, 60, 60), font=font)

            idx_label = f"#{i+1}"
            idx_bbox  = draw.textbbox((0, 0), idx_label, font=font)
            idx_width = idx_bbox[2] - idx_bbox[0]
            draw.text((x + (swatch_width - idx_width) // 2, text_y + 16),
                      idx_label, fill=(120, 120, 120), font=font)

    img.save(output_path)


# txt
def generate_results_txt(result: dict, cvd_types: List[CVD], output_path: str) -> None:
    """write palette results to plain-text file"""
    lines = []
    n = len(result['hex'])
    anchor_count = result.get('anchor_count', 0)

    lines.append("distinct color palette")
    lines.append(f"colors generated: {n}")
    if anchor_count > 0:
        lines.append(f"anchor colors: {anchor_count}")
    lines.append(f"minimum dE (CIEDE2000): {result['min_delta_e']}")
    lines.append(f"optimized for: {', '.join(c.value for c in cvd_types)}")
    lines.append("")

    lines.append("colors")
    for i, (h, rgb, lab) in enumerate(zip(result['hex'], result['rgb'], result['lab'])):
        marker = " [anchor]" if i < anchor_count else ""
        lines.append(f"{i+1:3d}. {h}  RGB({rgb[0]:3d}, {rgb[1]:3d}, {rgb[2]:3d})  "
                     f"Lab({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}){marker}")
    lines.append("")

    lines.append("dE by vision type")
    optimized = {c.value for c in cvd_types}
    for cvd_name, de in result['per_cvd_delta_e'].items():
        marker = " optimized" if cvd_name in optimized else ""
        lines.append(f"\t{cvd_name:12s}: {de:6.2f}{marker}")
    lines.append("")

    lines.append("hex list")
    lines.append(str(result['hex']))
    lines.append("")

    lines.append("css variables")
    for i, h in enumerate(result['hex']):
        lines.append(f"\t--color-{i+1}: {h};")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# colorspace plot
def generate_colorspace_plot(result: dict, output_path: str) -> None:
    """Lab color space visualization; left: a*b* chromaticity, right: L* lightness"""
    if not HAS_MATPLOTLIB:
        print("warning: matplotlib not installed; skipping color space plot", file=sys.stderr)
        return

    labs       = np.array(result['lab'])
    hex_colors = result['hex']
    anchor_count = result.get('anchor_count', 0)
    n = len(labs)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)

    # gamut boundary: sample sRGB cube surface
    gamut_points = []
    steps = 30
    for val in np.linspace(0, 1, steps):
        for r in [0, 1]:
            for g in np.linspace(0, 1, steps):
                gamut_points.append([r, g, val])
                gamut_points.append([r, val, g])
        for g in [0, 1]:
            for r in np.linspace(0, 1, steps):
                gamut_points.append([r, g, val])
                gamut_points.append([val, g, r])
        for b in [0, 1]:
            for r in np.linspace(0, 1, steps):
                gamut_points.append([r, val, b])
                gamut_points.append([val, r, b])

    gamut_rgb = np.array(gamut_points)
    gamut_lab = np.array([rgb_to_lab(rgb) for rgb in gamut_rgb])
    ax1.scatter(gamut_lab[:, 1], gamut_lab[:, 2], c='#e0e0e0', s=1, alpha=0.5, zorder=1)

    for i, (lab, hex_col) in enumerate(zip(labs, hex_colors)):
        is_anchor  = i < anchor_count
        marker     = 's' if is_anchor else 'o'
        edge_color = 'black' if is_anchor else 'white'
        edge_width = 2 if is_anchor else 1.5
        size       = 200 if is_anchor else 150

        ax1.scatter(lab[1], lab[2], c=hex_col, s=size, marker=marker,
                    edgecolors=edge_color, linewidths=edge_width, zorder=3)
        ax1.annotate(f'{i+1}', (lab[1], lab[2]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='#333333', zorder=4)

    ax1.set_xlabel('a* (green-red)', fontsize=11)
    ax1.set_ylabel('b* (blue-yellow)', fontsize=11)
    ax1.set_title('chromaticity (a*b* plane)', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.5, zorder=0)
    ax1.axvline(x=0, color='#cccccc', linestyle='-', linewidth=0.5, zorder=0)
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(-90, 90)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = fig.add_subplot(122)
    indices  = np.arange(1, n + 1)
    L_values = labs[:, 0]

    y_labels = []
    for i, (idx, L, hex_col) in enumerate(zip(indices, L_values, hex_colors)):
        is_anchor  = i < anchor_count
        edge_color = 'black' if is_anchor else 'white'
        edge_width = 2 if is_anchor else 1.5

        ax2.barh(idx, L, color=hex_col, edgecolor=edge_color,
                 linewidth=edge_width, height=0.7)
        ax2.text(L + 2, idx, f'{L:.0f}', va='center', fontsize=9)
        y_labels.append(f"#{i+1} {hex_col.upper()}")

    ax2.set_xlabel('L* (lightness)', fontsize=11)
    ax2.set_ylabel('color', fontsize=11)
    ax2.set_title('lightness distribution', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.set_ylim(0.3, n + 0.7)
    ax2.set_yticks(indices)
    ax2.set_yticklabels(y_labels, fontfamily='monospace', fontsize=9)
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    if anchor_count > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                       markeredgecolor='black', markersize=10, label='anchor'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='white', markersize=10, label='generated'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def generate_distinct_colors(n: int,
                             L_range: Tuple[float, float] = (30, 85),
                             grid_step: int = 6,
                             cvd_types: List[CVD] = None,
                             anchor_colors: List[str] = None,
                             verbose: bool = True) -> dict:
    """
    generate n perceptually distinct colors
        n:             number of colors to generate
        L_range:       lightness range (0-100); narrower = more uniform brightness
        grid_step:     sampling density in Lab space; smaller = finer but slower
        cvd_types:     CVD types to optimize for; None = normal vision only
        anchor_colors: color strings (#hex or R,G,B) to include
        verbose:       print progress
    """
    if n <= 0:
        return {'hex': [], 'rgb': [], 'lab': [], 'min_delta_e': None}

    if cvd_types is None:
        cvd_types = [CVD.NONE]

    anchor_labs = []
    if anchor_colors:
        for color_str in anchor_colors:
            lab = parse_color(color_str)  # raises ValueError on bad input
            anchor_labs.append(lab)
            if verbose:
                print(f"anchor color: {color_str} -> {lab_to_hex(lab)}")

        if len(anchor_labs) >= n:
            raise ValueError(
                f"number of anchor colors ({len(anchor_labs)}) must be less than n ({n})"
            )

    if verbose:
        print(f"optimizing for: {', '.join(c.value for c in cvd_types)}")

    candidates = generate_candidate_grid(L_range=L_range, step=grid_step)

    colors_needed = n - len(anchor_labs)
    if len(candidates) < colors_needed:
        raise ValueError(
            f"cannot generate {colors_needed} additional colors with current settings; "
            f"only {len(candidates)} candidates available; try smaller grid_step"
        )

    if verbose:
        print(f"searching {len(candidates)} candidates...")

    selected_lab = greedy_select(candidates, n, cvd_types,
                                 anchor_labs=anchor_labs if anchor_labs else None,
                                 verbose=verbose)

    hex_colors = [lab_to_hex(lab) for lab in selected_lab]
    rgb_colors = [
        tuple(int(x) for x in np.clip(lab_to_rgb(lab) * 255, 0, 255).round())
        for lab in selected_lab
    ]

    per_cvd_de = {
        cvd.value: round(min_delta_e(selected_lab, [cvd]), 2)
        for cvd in [CVD.NONE, CVD.PROTANOPIA, CVD.DEUTERANOPIA, CVD.TRITANOPIA]
    }
    overall_min = min(per_cvd_de[c.value] for c in cvd_types)

    return {
        'hex':           hex_colors,
        'rgb':           rgb_colors,
        'lab':           selected_lab.tolist(),
        'min_delta_e':   round(overall_min, 2),
        'per_cvd_delta_e': per_cvd_de,
        'anchor_count':  len(anchor_labs),
    }


# contrast checking relative to reference
_WCAG_THRESHOLDS = {
    "normal text aa":  4.5,
    "normal text aaa": 7.0,
    "large text aa":   3.0,
    "ui component aa": 3.0,
}


def relative_luminance(rgb_0_1: Tuple[float, float, float]) -> float:
    """wcag 2.x relative luminance (IEC 61966-2-1)"""
    def linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (linearize(c) for c in rgb_0_1)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(rgb1_255: Tuple[int, int, int],
                   rgb2_255: Tuple[int, int, int]) -> float:
    l1 = relative_luminance(tuple(c / 255.0 for c in rgb1_255))
    l2 = relative_luminance(tuple(c / 255.0 for c in rgb2_255))
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def parse_color_to_rgb255(color_str: str) -> Tuple[int, int, int]:
    """parse color string -> (r, g, b) in 0-255; accepts all formats from parse_color plus css named colors"""
    import re
    s = color_str.strip().lower()

    _CSS = {
        "white": (255, 255, 255), "black": (0, 0, 0),
        "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
        "yellow": (255, 255, 0), "gray": (128, 128, 128),
        "grey": (128, 128, 128), "silver": (192, 192, 192),
        "navy": (0, 0, 128), "teal": (0, 128, 128),
        "purple": (128, 0, 128), "orange": (255, 165, 0),
    }
    if s in _CSS:
        return _CSS[s]

    # delegate to existing parse_color, which handles hex, rgb(), r,g,b
    lab = parse_color(color_str)
    rgb_0_1 = lab_to_rgb(lab)
    rgb_255 = tuple(int(np.clip(v * 255, 0, 255).round()) for v in rgb_0_1)
    return rgb_255


def generate_contrast_image(palette_hex: List[str], palette_rgb: List[Tuple],
                            against_rgb: Tuple[int, int, int],
                            against_hex: str, output_path: str,
                            anchor_count: int = 0) -> None:
    """contrast PNG; palette vs reference with contrast ratio and wcag pass/fail per row"""
    if Image is None:
        print("warning: pillow not installed; skipping contrast visualization", file=sys.stderr)
        return

    n           = len(palette_hex)
    row_h       = 72
    swatch_w    = 110
    gap         = 6
    text_x0     = swatch_w * 2 + gap + 20
    header_h    = 56
    padding     = 14
    img_w       = 620
    img_h       = header_h + n * row_h + padding * 2

    bg          = (245, 245, 245)
    txt_dark    = (30, 30, 30)
    txt_muted   = (110, 110, 110)
    pass_bg     = (39, 174, 96)
    fail_bg     = (192, 57, 43)
    badge_txt   = (255, 255, 255)

    img  = Image.new('RGB', (img_w, img_h), color=bg)
    draw = ImageDraw.Draw(img)

    try:
        font_mono  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        font_bold  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    except (OSError, IOError):
        font_mono = font_bold = font_small = ImageFont.load_default()

    # header: reference color
    draw.rectangle([0, 0, img_w, header_h], fill=against_rgb)
    # choose legible label color based on luminance
    lum = relative_luminance(tuple(c / 255.0 for c in against_rgb))
    label_col = (15, 15, 15) if lum > 0.18 else (240, 240, 240)
    header_text = f"reference color: {against_hex}  RGB{against_rgb}"
    draw.text((padding, header_h // 2 - 8), header_text, fill=label_col, font=font_bold)

    # columns
    col_y = header_h + 6
    draw.text((padding, col_y), "palette color", fill=txt_muted, font=font_small)
    draw.text((swatch_w + gap + padding, col_y), "reference", fill=txt_muted, font=font_small)
    draw.text((text_x0, col_y),
              f"{'ratio':>7}   {'norm aa':>7}  {'norm aaa':>8}  {'large aa':>8}  {'ui aa':>6}",
              fill=txt_muted, font=font_small)

    # per-color rows
    criteria = [
        ("normal text aa",  4.5),
        ("normal text aaa", 7.0),
        ("large text aa",   3.0),
        ("ui component aa", 3.0),
    ]

    def badge(draw, x, y, label, passed):
        col = pass_bg if passed else fail_bg
        tw  = draw.textlength(label, font=font_small)
        bw, bh = int(tw) + 8, 14
        draw.rectangle([x, y, x + bw, y + bh], fill=col)
        draw.text((x + 4, y + 1), label, fill=badge_txt, font=font_small)
        return bw + 6

    for i, (h, rgb) in enumerate(zip(palette_hex, palette_rgb)):
        row_y    = header_h + 20 + i * row_h
        swatch_y = row_y
        swatch_h = row_h - 14

        # palette swatch
        draw.rectangle([padding, swatch_y, padding + swatch_w, swatch_y + swatch_h],
                       fill=rgb, outline=(180, 180, 180))

        # reference swatch (same for every row)
        ref_x = padding + swatch_w + gap
        draw.rectangle([ref_x, swatch_y, ref_x + swatch_w, swatch_y + swatch_h],
                       fill=against_rgb, outline=(180, 180, 180))

        # hex, anchor marker
        anchor_mark = " [anchor]" if i < anchor_count else ""
        draw.text((padding, swatch_y + swatch_h + 2),
                  f"{h}{anchor_mark}", fill=txt_muted, font=font_small)

        # ratio
        ratio = contrast_ratio(rgb, against_rgb)
        ratio_str = f"{ratio:.2f}:1"
        draw.text((text_x0, swatch_y + 4), ratio_str, fill=txt_dark, font=font_bold)

        # badges
        bx = text_x0 + 72
        for _, threshold in criteria:
            passed = ratio >= threshold
            label  = "pass" if passed else "fail"
            bx += badge(draw, bx, swatch_y + 4, label, passed) + 2

    img.save(output_path)


def print_contrast_report(palette_hex: List[str], palette_rgb: List[Tuple],
                          against_str: str, anchor_count: int = 0,
                          output_prefix: str = "palette") -> None:
    try:
        against_rgb = parse_color_to_rgb255(against_str)
    except ValueError as e:
        sys.exit(f"error parsing --contrast color: {e}")

    against_hex = "#{:02x}{:02x}{:02x}".format(*against_rgb)

    print(f"\n")
    print(f"\tcontrast against {against_hex}  RGB{against_rgb}")
    print(f"\n")
    print(f"\t{'#':<4}  {'color':<10}  {'ratio':>7}  "
          f"{'norm aa':>8}  {'norm aaa':>9}  {'large aa':>9}  {'ui aa':>6}")

    any_fail = False
    for i, (h, rgb) in enumerate(zip(palette_hex, palette_rgb)):
        ratio = contrast_ratio(rgb, against_rgb)
        marker = " *" if i < anchor_count else ""

        def pf(threshold):
            return "pass" if ratio >= threshold else "fail"

        n_aa  = pf(_WCAG_THRESHOLDS["normal text aa"])
        n_aaa = pf(_WCAG_THRESHOLDS["normal text aaa"])
        l_aa  = pf(_WCAG_THRESHOLDS["large text aa"])
        u_aa  = pf(_WCAG_THRESHOLDS["ui component aa"])

        if "fail" in (n_aa, l_aa, u_aa):
            any_fail = True

        print(f"\t{i+1:<4}  {h:<10}  {ratio:>6.2f}:1  "
              f"{n_aa:>8}  {n_aaa:>9}  {l_aa:>9}  {u_aa:>6}{marker}")

    print()
    if any_fail:
        print(f"\tsome colors fail wcag aa thresholds against {against_hex}")
    else:
        print(f"\tall colors pass wcag aa thresholds against {against_hex}")
    print()

    img_path = f"{output_prefix}_contrast.png"
    generate_contrast_image(palette_hex, palette_rgb, against_rgb, against_hex,
                            img_path, anchor_count)
    if Image is not None:
        print(f"\tcontrast visualization saved: {img_path}\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='generate perceptually distinct colors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
cvd types:
	none:   normal color vision (default)
	protan: protanopia (red-blind)
	deutan: deuteranopia (green-blind, most common)
	tritan: tritanopia (blue-blind)
	all:    optimize for all types simultaneously

anchor colors:
	specify colors that must be included in the palette; algorithm selects remaining colors to maximize distance from anchors and from each other.
	formats: #RRGGBB, RRGGBB, or R,G,B

examples:
	%(prog)s 6
	%(prog)s 8 --cvd deutan
	%(prog)s 10 --cvd protan deutan
	%(prog)s 12 --cvd all --L-min 35 --L-max 80
	%(prog)s 8 --cvd all -o my_palette
	%(prog)s 6 --anchor #FF0000
	%(prog)s 8 --anchor #1a1a1a --anchor 255,200,0
	%(prog)s 10 --anchor "#003366" --anchor "rgb(200,50,50)" --cvd all
	%(prog)s 8 --contrast #ffffff
	%(prog)s 8 --cvd all --contrast white
        """)

    parser.add_argument('n', type=int,
                        help='number of colors (including anchors)')
    parser.add_argument('--cvd', nargs='+', default=['none'],
                        choices=['none', 'protan', 'deutan', 'tritan', 'all'],
                        help='color vision deficiency types to optimize for')
    parser.add_argument('--anchor', action='append', metavar='COLOR',
                        help='anchor color(s) to include (#RRGGBB or R,G,B). can be repeated.')
    parser.add_argument('--L-min', type=float, default=30, help='min lightness (0-100)')
    parser.add_argument('--L-max', type=float, default=85, help='max lightness (0-100)')
    parser.add_argument('--step', type=int, default=6,
                        help='grid sampling step (smaller = slower but better)')
    parser.add_argument('--quiet', '-q', action='store_true', help='suppress progress output')
    parser.add_argument('-o', '--output', type=str, default='palette',
                        help='output filename prefix (default: palette)')
    parser.add_argument('--no-png',  action='store_true', help='skip png palette swatch generation')
    parser.add_argument('--no-txt',  action='store_true', help='skip txt generation')
    parser.add_argument('--no-plot', action='store_true', help='skip color space plot generation')
    parser.add_argument('--contrast', metavar='COLOR',
                        help='check each generated color against this color for wcag contrast ratios'
                             ' (#RRGGBB, R,G,B, or css name)')

    args = parser.parse_args()

    output_dir = Path("access_output")
    output_dir.mkdir(exist_ok=True)
    out = str(output_dir / f"colors_{args.output}")

    cvd_map = {
        'none':  CVD.NONE, 'protan': CVD.PROTANOPIA,
        'deutan': CVD.DEUTERANOPIA, 'tritan': CVD.TRITANOPIA,
    }
    if 'all' in args.cvd:
        cvd_types = list(cvd_map.values())
    else:
        cvd_types = [cvd_map[c] for c in args.cvd]

    try:
        result = generate_distinct_colors(
            args.n,
            L_range=(args.L_min, args.L_max),
            grid_step=args.step,
            cvd_types=cvd_types,
            anchor_colors=args.anchor,
            verbose=not args.quiet,
        )
    except ValueError as e:
        sys.exit(f"error: {e}")

    anchor_count = result.get('anchor_count', 0)
    print(f"\n")
    print(f"\tgenerated {args.n} colors  (min dE = {result['min_delta_e']})")
    print(f"\n")
    for i, (h, rgb) in enumerate(zip(result['hex'], result['rgb'])):
        marker = "  [anchor]" if i < anchor_count else ""
        print(f"\t{i+1:2d}.  {h}  RGB{rgb}{marker}")

    print(f"\n\tdE by vision type:")
    optimized = {c.value for c in cvd_types}
    for cvd_name, de in result['per_cvd_delta_e'].items():
        marker = "  *" if cvd_name in optimized else ""
        print(f"\t\t{cvd_name:10s}: {de:5.1f}{marker}")

    print(f"\n\thex list: {result['hex']}\n")

    if not args.no_png:
        png_path = f"{out}.png"
        generate_palette_image(result['hex'], result['rgb'], png_path)
        print(f"\tpalette image saved: {png_path}")

    if not args.no_txt:
        txt_path = f"{out}.txt"
        generate_results_txt(result, cvd_types, txt_path)
        print(f"\tresults saved: {txt_path}")

    if not args.no_plot:
        plot_path = f"{out}_colorspace.png"
        generate_colorspace_plot(result, plot_path)
        print(f"\tcolor space plot saved: {plot_path}")

    if args.contrast:
        print_contrast_report(result['hex'], result['rgb'],
                              args.contrast, anchor_count,
                              output_prefix=out)

    print()
