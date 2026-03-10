from __future__ import annotations
from PIL import Image, ImageFilter, ImageOps
import colorsys

import numpy as np

adjust = {
    'shadow':    [0, 0, 0, 1.0, 1.0, 1.0, [{}]],
    'middle':    [0, 0, 0, 1.0, 1.0, 1.0, [{}]],
    'highlight': [0, 0, 0, 1.0, 1.0, 1.0, [{}]],
}


# ------------------------------------------------------------
# Util
# ------------------------------------------------------------
def _clip255(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 255.0).astype(np.float32)


def _extract_rgb_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if img.mode == 'RGBA':
        return img.convert('RGB'), img.split()[-1]
    return (img if img.mode == 'RGB' else img.convert('RGB')), None


def _restore_alpha(rgb_img: Image.Image, alpha: Image.Image | None) -> Image.Image:
    if alpha is None:
        return rgb_img
    return Image.merge('RGBA', (*rgb_img.split(), alpha))


def _to_gray_luma(arr: np.ndarray) -> np.ndarray:
    gray = (
        arr[..., 0] * 0.299 +
        arr[..., 1] * 0.587 +
        arr[..., 2] * 0.114
    )
    return gray[..., None].astype(np.float32)


def _adjust_saturation(arr: np.ndarray, saturation: float) -> np.ndarray:
    gray = _to_gray_luma(arr)
    out = gray + (arr - gray) * float(saturation)
    return _clip255(out)


def _adjust_contrast(arr: np.ndarray, contrast: float) -> np.ndarray:
    mean = arr.mean(axis=(0, 1), keepdims=True)
    out = (arr - mean) * float(contrast) + mean
    return _clip255(out)


def _blend_amount(base: np.ndarray, fx: np.ndarray, amount: float) -> np.ndarray:
    out = base + (fx - base) * float(amount)
    return _clip255(out)


def _blur_rgb(arr: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray(_clip255(arr).astype(np.uint8), 'RGB')
    pil = pil.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(pil, dtype=np.float32)


def _screen_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 255.0 - ((255.0 - a) * (255.0 - b) / 255.0)


def _overlay_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / 255.0
    b_n = b / 255.0
    out = np.where(
        a_n < 0.5,
        2.0 * a_n * b_n,
        1.0 - 2.0 * (1.0 - a_n) * (1.0 - b_n)
    )
    return _clip255(out * 255.0)


def _normalize_filter_name(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def _iter_filter_items(filters_spec) -> list[tuple[str, float]]:
    if not filters_spec:
        return []

    items: list[tuple[str, float]] = []

    if isinstance(filters_spec, dict):
        for k, v in filters_spec.items():
            items.append((str(k), float(v)))
        return items

    if isinstance(filters_spec, (list, tuple)):
        for obj in filters_spec:
            if not isinstance(obj, dict):
                raise ValueError("Filters must be dict or list[dict]")
            for k, v in obj.items():
                items.append((str(k), float(v)))
        return items

    raise ValueError("Filters must be dict or list[dict]")


def _arr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(_clip255(arr).astype(np.uint8), 'RGB')


def _pil_to_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert('RGB'), dtype=np.float32)


def _pil_filter_rgb(arr: np.ndarray, pil_filter) -> np.ndarray:
    return _pil_to_arr(_arr_to_pil(arr).filter(pil_filter))


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2).astype(np.float32)


def _simple_edges_gray(arr: np.ndarray, strength: float = 2.2) -> np.ndarray:
    gray = _to_gray_luma(arr)[..., 0]
    gx = np.abs(gray - np.roll(gray, 1, axis=1))
    gy = np.abs(gray - np.roll(gray, 1, axis=0))
    edge = np.clip((gx + gy) * strength, 0.0, 255.0)
    return edge.astype(np.float32)


def _lineart_base(arr: np.ndarray, strength: float = 2.4) -> np.ndarray:
    edge = _simple_edges_gray(arr, strength=strength)
    line = 255.0 - edge
    return _gray_to_rgb(np.clip(line, 0.0, 255.0))


# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------
def _filter_sepia(arr: np.ndarray) -> np.ndarray:
    m = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ], dtype=np.float32)
    return _clip255(arr @ m.T)


def _filter_cyber(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 2.0)
    edges = np.abs(arr - blur)
    out = arr * np.array([0.90, 1.05, 1.18], dtype=np.float32)
    out += np.array([0.0, 8.0, 14.0], dtype=np.float32)
    out += edges * np.array([0.35, 0.95, 1.20], dtype=np.float32)
    out = _adjust_saturation(out, 1.25)
    out = _adjust_contrast(out, 1.06)
    return _clip255(out)


def _filter_glitch(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    shift = max(1, int(round(min(h, w) * 0.008)))
    r = np.roll(arr[..., 0], -shift, axis=1)
    g = arr[..., 1]
    b = np.roll(arr[..., 2], shift, axis=1)
    out = np.stack([r, g, b], axis=2)
    out[::2, :, :] *= 0.97
    return _clip255(out)


def _filter_bloom(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 6.0)
    bloom = _screen_blend(arr, blur)
    bloom = _blend_amount(arr, bloom, 0.75)
    return _clip255(bloom)


def _filter_noir(arr: np.ndarray) -> np.ndarray:
    gray = _to_gray_luma(arr)
    out = np.repeat(gray, 3, axis=2)
    out = _adjust_contrast(out, 1.25)
    return _clip255(out)


def _filter_vintage(arr: np.ndarray) -> np.ndarray:
    out = _filter_sepia(arr)
    out = _adjust_saturation(out, 0.88)
    out = out * np.array([1.02, 1.00, 0.96], dtype=np.float32)
    out += np.array([6.0, 3.0, 0.0], dtype=np.float32)
    out = _adjust_contrast(out, 0.94)
    return _clip255(out)


def _filter_warm(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([1.08, 1.03, 0.94], dtype=np.float32)
    out += np.array([8.0, 3.0, -2.0], dtype=np.float32)
    out = _adjust_saturation(out, 1.05)
    return _clip255(out)


def _filter_cool(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([0.94, 1.02, 1.08], dtype=np.float32)
    out += np.array([-2.0, 2.0, 8.0], dtype=np.float32)
    out = _adjust_saturation(out, 1.03)
    return _clip255(out)


def _filter_matte(arr: np.ndarray) -> np.ndarray:
    out = arr * 0.92 + 16.0
    out = _adjust_contrast(out, 0.88)
    out = _adjust_saturation(out, 0.92)
    return _clip255(out)


def _filter_vivid(arr: np.ndarray) -> np.ndarray:
    out = _adjust_saturation(arr, 1.35)
    out = _adjust_contrast(out, 1.10)
    return _clip255(out)


def _filter_crossprocess(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[..., 0] = np.clip(out[..., 0] * 1.08 + 4.0, 0, 255)
    out[..., 1] = np.clip(out[..., 1] * 1.03 + 8.0, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * 0.92 + 10.0, 0, 255)
    out = _adjust_contrast(out, 1.08)
    out = _adjust_saturation(out, 1.08)
    return _clip255(out)


def _filter_fade(arr: np.ndarray) -> np.ndarray:
    out = arr * 0.90 + 18.0
    out = _adjust_saturation(out, 0.82)
    out = _adjust_contrast(out, 0.90)
    return _clip255(out)


def _filter_dramatic(arr: np.ndarray) -> np.ndarray:
    out = _adjust_contrast(arr, 1.30)
    out = out * np.array([1.02, 1.00, 0.98], dtype=np.float32)
    return _clip255(out)


def _filter_softglow(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 3.5)
    out = _screen_blend(arr, blur * 0.85)
    out = _adjust_contrast(out, 0.96)
    return _clip255(out)


def _filter_pastel(arr: np.ndarray) -> np.ndarray:
    out = arr * 0.93 + 14.0
    out = _adjust_saturation(out, 0.78)
    out = _adjust_contrast(out, 0.92)
    return _clip255(out)


def _filter_tealorange(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    lum = out.mean(axis=2, keepdims=True)
    shadow_mask = np.clip((128.0 - lum) / 128.0, 0.0, 1.0)
    highlight_mask = np.clip((lum - 128.0) / 128.0, 0.0, 1.0)

    teal = out * np.array([0.92, 1.03, 1.12], dtype=np.float32)
    orange = out * np.array([1.10, 1.03, 0.92], dtype=np.float32)

    out = out * (1.0 - shadow_mask - highlight_mask) + teal * shadow_mask + orange * highlight_mask
    out = _adjust_saturation(out, 1.10)
    return _clip255(out)


def _filter_amber(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([1.10, 1.04, 0.88], dtype=np.float32)
    out += np.array([10.0, 4.0, -4.0], dtype=np.float32)
    return _clip255(out)


def _filter_moonlight(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([0.88, 0.97, 1.10], dtype=np.float32)
    out += np.array([-6.0, 0.0, 8.0], dtype=np.float32)
    out = _adjust_contrast(out, 1.04)
    return _clip255(out)


def _filter_forest(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([0.95, 1.08, 0.93], dtype=np.float32)
    out += np.array([-2.0, 6.0, -2.0], dtype=np.float32)
    return _clip255(out)


def _filter_lavender(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([1.03, 0.98, 1.08], dtype=np.float32)
    out += np.array([4.0, -1.0, 8.0], dtype=np.float32)
    out = _adjust_saturation(out, 0.96)
    return _clip255(out)


def _filter_duotone(arr: np.ndarray) -> np.ndarray:
    gray = _to_gray_luma(arr) / 255.0
    c1 = np.array([20.0, 30.0, 70.0], dtype=np.float32)
    c2 = np.array([255.0, 210.0, 120.0], dtype=np.float32)
    out = c1 * (1.0 - gray) + c2 * gray
    return _clip255(out)


def _filter_iron(arr: np.ndarray) -> np.ndarray:
    gray = _to_gray_luma(arr)
    out = np.repeat(gray, 3, axis=2)
    out *= np.array([0.95, 1.00, 1.05], dtype=np.float32)
    out = _adjust_contrast(out, 1.18)
    return _clip255(out)


def _filter_sunset(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([1.12, 0.98, 0.90], dtype=np.float32)
    out += np.array([12.0, 2.0, -4.0], dtype=np.float32)
    out = _adjust_saturation(out, 1.12)
    return _clip255(out)


def _filter_frost(arr: np.ndarray) -> np.ndarray:
    out = arr * np.array([0.92, 1.00, 1.12], dtype=np.float32)
    out += np.array([-4.0, 2.0, 10.0], dtype=np.float32)
    out = _adjust_saturation(out, 0.95)
    return _clip255(out)


def _filter_sharpenlite(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 1.2)
    out = arr + (arr - blur) * 0.8
    return _clip255(out)


def _filter_dream(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 4.0)
    out = _screen_blend(arr, blur * 0.9)
    out = _adjust_saturation(out, 0.92)
    out = _adjust_contrast(out, 0.93)
    return _clip255(out)


def _filter_noise(arr: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0.0, 18.0, arr.shape).astype(np.float32)
    out = arr + noise
    return _clip255(out)


def _filter_filmgrain(arr: np.ndarray) -> np.ndarray:
    grain = np.random.normal(0.0, 12.0, arr.shape[:2]).astype(np.float32)
    grain = _gray_to_rgb(grain)
    out = arr + grain
    out = _adjust_contrast(out, 1.04)
    out = _adjust_saturation(out, 0.97)
    return _clip255(out)


def _filter_sharpen(arr: np.ndarray) -> np.ndarray:
    return _pil_filter_rgb(arr, ImageFilter.UnsharpMask(radius=1.6, percent=160, threshold=2))


def _filter_strongsharpen(arr: np.ndarray) -> np.ndarray:
    return _pil_filter_rgb(arr, ImageFilter.UnsharpMask(radius=2.2, percent=260, threshold=2))


def _filter_clarity(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 1.8)
    out = arr + (arr - blur) * 0.9
    out = _adjust_contrast(out, 1.10)
    return _clip255(out)


def _filter_highpass(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 3.0)
    out = arr - blur + 128.0
    out = _adjust_contrast(out, 1.15)
    return _clip255(out)


def _filter_findedges(arr: np.ndarray) -> np.ndarray:
    edges = _pil_filter_rgb(arr, ImageFilter.FIND_EDGES)
    edges = _adjust_contrast(edges, 1.3)
    return _clip255(edges)


def _filter_lineart(arr: np.ndarray) -> np.ndarray:
    return _lineart_base(arr, strength=2.6)


def _filter_ink(arr: np.ndarray) -> np.ndarray:
    line = _lineart_base(arr, strength=3.0)[..., 0]
    ink = np.where(line < 215.0, 0.0, 255.0).astype(np.float32)
    return _gray_to_rgb(ink)


def _filter_sketch(arr: np.ndarray) -> np.ndarray:
    gray = _to_gray_luma(arr)[..., 0]
    inv = 255.0 - gray
    inv_rgb = _gray_to_rgb(inv)
    blur_inv = _blur_rgb(inv_rgb, 6.0)[..., 0]
    denom = np.maximum(255.0 - blur_inv, 1.0)
    sketch = np.clip(gray * 255.0 / denom, 0.0, 255.0)
    return _gray_to_rgb(sketch)


def _filter_emboss(arr: np.ndarray) -> np.ndarray:
    out = _pil_filter_rgb(arr, ImageFilter.EMBOSS)
    out = _adjust_contrast(out, 1.08)
    return _clip255(out)


def _filter_posterize(arr: np.ndarray) -> np.ndarray:
    pil = _arr_to_pil(arr)
    out = ImageOps.posterize(pil, 4)
    return _pil_to_arr(out)


def _filter_solarize(arr: np.ndarray) -> np.ndarray:
    pil = _arr_to_pil(arr)
    out = ImageOps.solarize(pil, threshold=128)
    return _pil_to_arr(out)


def _filter_thresholdmono(arr: np.ndarray) -> np.ndarray:
    gray = _to_gray_luma(arr)[..., 0]
    mono = np.where(gray >= 128.0, 255.0, 0.0).astype(np.float32)
    return _gray_to_rgb(mono)


def _filter_blueprint(arr: np.ndarray) -> np.ndarray:
    line = _lineart_base(arr, strength=2.8)[..., 0] / 255.0
    bg = np.zeros_like(arr, dtype=np.float32)
    bg[..., 0] = 15.0
    bg[..., 1] = 45.0
    bg[..., 2] = 120.0
    ink = np.zeros_like(arr, dtype=np.float32)
    ink[..., 0] = 170.0
    ink[..., 1] = 220.0
    ink[..., 2] = 255.0
    out = bg * (1.0 - line[..., None]) + ink * line[..., None]
    return _clip255(out)


def _filter_neonedges(arr: np.ndarray) -> np.ndarray:
    edge = _simple_edges_gray(arr, strength=3.0) / 255.0
    base = arr * 0.35
    neon = np.zeros_like(arr, dtype=np.float32)
    neon[..., 0] = edge * 120.0
    neon[..., 1] = edge * 255.0
    neon[..., 2] = edge * 255.0
    out = _screen_blend(base, neon)
    return _clip255(out)


def _filter_hdr(arr: np.ndarray) -> np.ndarray:
    blur = _blur_rgb(arr, 2.4)
    out = arr + (arr - blur) * 1.1
    out = _adjust_contrast(out, 1.16)
    out = _adjust_saturation(out, 1.12)
    return _clip255(out)


def _filter_dehaze(arr: np.ndarray) -> np.ndarray:
    lo = np.percentile(arr, 1.5, axis=(0, 1), keepdims=True)
    hi = np.percentile(arr, 98.5, axis=(0, 1), keepdims=True)
    out = (arr - lo) * (255.0 / np.maximum(hi - lo, 1.0))
    out = _adjust_contrast(out, 1.05)
    return _clip255(out)


def _filter_toon(arr: np.ndarray) -> np.ndarray:
    smooth = _blur_rgb(arr, 1.4)
    pil = _arr_to_pil(smooth)
    smooth = _pil_to_arr(ImageOps.posterize(pil, 4))
    line = _lineart_base(arr, strength=3.2)[..., 0]
    line_mask = (255.0 - line) / 255.0
    out = smooth * (1.0 - line_mask[..., None]) + 0.0 * line_mask[..., None]
    return _clip255(out)


def _filter_medianclean(arr: np.ndarray) -> np.ndarray:
    return _pil_filter_rgb(arr, ImageFilter.MedianFilter(size=3))
    

FILTERS: dict[str, tuple[str, callable]] = {
    'sepia':        ('Sepia', _filter_sepia),
    'cyber':        ('Cyber', _filter_cyber),
    'glitch':       ('Glitch', _filter_glitch),
    'bloom':        ('Bloom', _filter_bloom),
    'noir':         ('Noir', _filter_noir),
    'vintage':      ('Vintage', _filter_vintage),
    'warm':         ('Warm', _filter_warm),
    'cool':         ('Cool', _filter_cool),
    'matte':        ('Matte', _filter_matte),
    'vivid':        ('Vivid', _filter_vivid),
    'crossprocess': ('CrossProcess', _filter_crossprocess),
    'fade':         ('Fade', _filter_fade),
    'dramatic':     ('Dramatic', _filter_dramatic),
    'softglow':     ('SoftGlow', _filter_softglow),
    'pastel':       ('Pastel', _filter_pastel),
    'tealorange':   ('TealOrange', _filter_tealorange),
    'amber':        ('Amber', _filter_amber),
    'moonlight':    ('Moonlight', _filter_moonlight),
    'forest':       ('Forest', _filter_forest),
    'lavender':     ('Lavender', _filter_lavender),
    'duotone':      ('Duotone', _filter_duotone),
    'iron':         ('Iron', _filter_iron),
    'sunset':       ('Sunset', _filter_sunset),
    'frost':        ('Frost', _filter_frost),
    'sharpenlite':  ('SharpenLite', _filter_sharpenlite),
    'dream':        ('Dream', _filter_dream),
    'noise':         ('Noise', _filter_noise),
    'filmgrain':     ('FilmGrain', _filter_filmgrain),
    'sharpen':       ('Sharpen', _filter_sharpen),
    'strongsharpen': ('StrongSharpen', _filter_strongsharpen),
    'clarity':       ('Clarity', _filter_clarity),
    'highpass':      ('HighPass', _filter_highpass),
    'findedges':     ('FindEdges', _filter_findedges),
    'lineart':       ('LineArt', _filter_lineart),
    'ink':           ('Ink', _filter_ink),
    'sketch':        ('Sketch', _filter_sketch),
    'emboss':        ('Emboss', _filter_emboss),
    'posterize':     ('Posterize', _filter_posterize),
    'solarize':      ('Solarize', _filter_solarize),
    'thresholdmono': ('ThresholdMono', _filter_thresholdmono),
    'blueprint':     ('Blueprint', _filter_blueprint),
    'neonedges':     ('NeonEdges', _filter_neonedges),
    'hdr':           ('HDR', _filter_hdr),
    'dehaze':        ('Dehaze', _filter_dehaze),
    'toon':          ('Toon', _filter_toon),
    'medianclean':   ('MedianClean', _filter_medianclean),

    # alias
    'cyberpunk':    ('Cyber', _filter_cyber),
    'blackwhite':   ('Noir', _filter_noir),
    'soft':         ('SoftGlow', _filter_softglow),
    'tealorangecinematic': ('TealOrange', _filter_tealorange),
    'grain':         ('FilmGrain', _filter_filmgrain),
    'edge':          ('FindEdges', _filter_findedges),
    'edges':         ('FindEdges', _filter_findedges),
    'outline':       ('LineArt', _filter_lineart),
    'comic':         ('Toon', _filter_toon),
    'denoise':       ('MedianClean', _filter_medianclean),
}


def list_available_filters() -> list[str]:
    names = []
    seen = set()
    for display_name, _fn in FILTERS.values():
        if display_name not in seen:
            seen.add(display_name)
            names.append(display_name)
    return names


def _apply_filters_array(arr: np.ndarray, filters_spec) -> np.ndarray:
    items = _iter_filter_items(filters_spec)
    if not items:
        return arr

    out = arr.copy()

    for name, amount in items:
        if amount < 0.0:
            raise ValueError(f"filter amount for '{name}' must be >= 0.0")

        key = _normalize_filter_name(name)
        if key not in FILTERS:
            raise ValueError(
                f"Unknown filter '{name}'. Available filters: {', '.join(list_available_filters())}"
            )

        if amount == 0.0:
            continue

        _display_name, fn = FILTERS[key]
        fx = fn(out.copy())
        out = _blend_amount(out, fx, amount)

    return _clip255(out)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def color_balance(img: Image.Image, adj: dict) -> Image.Image:
    """
    adj[region] = [R, G, B, Brightness, Contrast, Saturation, Filters]

    Filters:
        [{}]
        [{'Cyber': 0.7}]
        [{'Sepia': 0.6}, {'Bloom': 0.3}]
        [{'Sepia': 0.6}, {'Bloom': 0.3}, {'Glitch': 0.2}]
    """
    norm_adj: dict[str, tuple[float, float, float, float, float, float, object]] = {}

    for k in ('shadow', 'middle', 'highlight'):
        v = adj.get(k)
        if not isinstance(v, (list, tuple)) or len(v) not in (6, 7):
            raise ValueError(
                f"adjustments['{k}'] must be length-6 or length-7:\n"
                f"[R, G, B, Brightness, Contrast, Saturation, Filters]"
            )

        if len(v) == 6:
            r, g, b, bright, contrast, sat = v
            filters_spec = [{}]
        else:
            r, g, b, bright, contrast, sat, filters_spec = v

        norm_adj[k] = (
            float(r), float(g), float(b),
            float(bright), float(contrast), float(sat),
            filters_spec
        )

    rgb, alpha = _extract_rgb_alpha(img)
    orig = np.asarray(rgb, dtype=np.float32)

    lum = orig.mean(axis=2)
    ws = np.clip((128.0 - lum) / 128.0, 0.0, 1.0)
    wh = np.clip((lum - 128.0) / 128.0, 0.0, 1.0)
    wm = 1.0 - ws - wh

    res = np.zeros_like(orig, dtype=np.float32)

    for w, region in zip((ws, wm, wh), ('shadow', 'middle', 'highlight')):
        r, g, b, bright, contrast, sat, filters_spec = norm_adj[region]

        af = np.array([r, g, b], dtype=np.float32) / 100.0

        delta = (
            (255.0 - orig) * np.maximum(af, 0.0) +
            orig * np.minimum(af, 0.0)
        )

        rv = orig + delta
        rv = rv * bright
        rv = _adjust_contrast(rv, contrast)
        rv = _adjust_saturation(rv, sat)

        rv = _apply_filters_array(rv, filters_spec)

        res += rv * w[..., None]

    out = Image.fromarray(_clip255(res).astype(np.uint8), 'RGB')
    out = _restore_alpha(out, alpha)

    del orig, res, rv, delta, lum, ws, wh, wm
    return out

def invert_brightness(image: Image.Image) -> Image.Image:
    img = image.convert('RGB')
    pixels = img.load()
    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
            l = 1.0 - l
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            pixels[x, y] = (int(r2 * 255), int(g2 * 255), int(b2 * 255))

    return img
