"""Explicit conversion between MinerU page coordinates and PDF coordinates."""
from __future__ import annotations

import json
import math
from typing import Any


def pdf_bbox(page: Any, value: object, coordinate_system: str = "pdf_points"):
    """Return an unrotated PDF rectangle; never guess units from magnitude.

    MinerU content_list uses 0..1000 coordinates on the displayed page. PDF
    annotations use unrotated points; pixmap clips use the rotated page view.
    """
    import fitz
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        numbers = [float(x) for x in value]
    except (ValueError, TypeError):
        return None
    if not all(math.isfinite(x) for x in numbers):
        return None
    if numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    rect = fitz.Rect(numbers)
    view = page.rect
    if coordinate_system == "normalized_1000":
        if min(numbers) < 0 or max(numbers) > 1000:
            return None
        rect = fitz.Rect(numbers[0] * view.width / 1000,
                         numbers[1] * view.height / 1000,
                         numbers[2] * view.width / 1000,
                         numbers[3] * view.height / 1000)
        rect = rect * page.derotation_matrix
    elif coordinate_system not in {"pdf_points", "pdf", ""}:
        return None
    bounds = view * page.derotation_matrix
    rect &= bounds
    return None if rect.is_empty or rect.is_infinite else rect
