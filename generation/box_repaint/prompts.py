"""Per-category prompt template for clipart1k box repaint.

Names the class in the prompt (opposite of the parked background-repaint
route, which omitted it to avoid spawning extra instances) to anchor
identity under partial repaint at strength<1 — see
report/genaug-rung0-precheck.md, Route 3.
"""

CLIPART_TEMPLATE = "a {cat}, flat-color cartoon clipart illustration, bold black outlines"


def prompt_for_category(category_name):
    return CLIPART_TEMPLATE.format(cat=category_name)
