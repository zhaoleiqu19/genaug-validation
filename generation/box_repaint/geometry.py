"""Pure box/resize math shared by the box-repaint generation pipeline.

Deliberately free of PIL/torch imports: this module is exercised by
`python3 -m pytest` under the repo's system Python 3.6.8, which has
neither installed. PIL-dependent operations live in raster.py instead.
"""


def round16(x):
    """Round to the nearest multiple of 16, minimum 16 (FLUX requires
    both dimensions to be multiples of 16)."""
    return max(16, int(round(x / 16)) * 16)


def compute_resize(orig_w, orig_h, target_w=1024):
    """Return (new_w, new_h, scale_x, scale_y) so the resized width is
    close to target_w and both dimensions are multiples of 16."""
    scale = target_w / orig_w
    new_w = round16(orig_w * scale)
    new_h = round16(orig_h * scale)
    return new_w, new_h, new_w / orig_w, new_h / orig_h


def scale_bbox(bbox, scale_x, scale_y):
    """bbox = (x, y, w, h) in original pixels -> scaled pixels."""
    x, y, w, h = bbox
    return (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
