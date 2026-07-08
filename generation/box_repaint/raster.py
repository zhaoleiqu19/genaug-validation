"""PIL-dependent image operations for box repaint.

Not covered by `python3 -m pytest`: the repo's system Python 3.6.8 has no
Pillow installed. These functions only ever run inside the `flux2` conda
env via generate.py, and are checked by the manual smoke test in
docs/plans/2026-07-08-clipart1k-box-repaint-stage1.md Task 10, not by
unit tests.
"""
from PIL import Image, ImageDraw


def load_and_resize(path, new_w, new_h):
    img = Image.open(path).convert("RGB")
    return img.resize((new_w, new_h), Image.LANCZOS)


def build_box_mask(size, box):
    """size=(w,h); box=(x,y,w,h) in the same pixel space as size.
    Returns an 'L' mode mask: 255 = repaint (inside box), 0 = frozen."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    x, y, w, h = box
    d.rectangle([x, y, x + w - 1, y + h - 1], fill=255)
    return mask


def freeze_outside_box(generated_img, source_img, box):
    """Paste only the box region of generated_img onto a copy of
    source_img; every other pixel stays identical to source_img."""
    out = source_img.copy()
    x, y, w, h = box
    left, top = int(round(x)), int(round(y))
    right, bottom = int(round(x + w)), int(round(y + h))
    region = generated_img.crop((left, top, right, bottom))
    out.paste(region, (left, top))
    return out
