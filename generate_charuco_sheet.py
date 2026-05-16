from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


OUT_PDF = Path("aruco_markers_1p5cm_1cm_0p5cm_0p25cm_.pdf")
DICT_NAME = "DICT_4X4_50"
MARKER_IDS = list(range(8))
PAGE_SIZES_MM = [15.0, 10.0, 5.0, 2.5]


def _get_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def _make_marker_image(marker_id: int, px: int = 1024) -> Image.Image:
    aruco_dict = _get_dictionary()
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, px)
    return Image.fromarray(marker)


def _draw_page(c: canvas.Canvas, marker_size_mm: float) -> None:
    page_w, page_h = A4
    margin = 12 * mm
    gap_x = 8 * mm
    gap_y = 8 * mm
    cols = 2
    rows = 4

    cell_w = (page_w - 2 * margin - gap_x) / cols
    cell_h = (page_h - 2 * margin - 12 * mm - gap_y * (rows - 1)) / rows

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, page_h - margin + 2 * mm, f"ArUco {DICT_NAME} - {marker_size_mm:g} mm markers")
    c.setFont("Helvetica", 9)
    c.drawString(margin, page_h - margin - 3 * mm, "Print at 100% scale. IDs 0-7 from the box config.")

    for idx, marker_id in enumerate(MARKER_IDS):
        col = idx % cols
        row = idx // cols

        x0 = margin + col * (cell_w + gap_x)
        y0 = page_h - margin - 12 * mm - (row + 1) * cell_h - row * gap_y

        c.setLineWidth(0.4)
        c.setDash(3, 2)
        c.rect(x0, y0, cell_w, cell_h)
        c.setDash()

        img = _make_marker_image(marker_id)
        img_reader = ImageReader(img)

        marker_pt = marker_size_mm * mm
        x_img = x0 + (cell_w - marker_pt) / 2
        y_img = y0 + 15 * mm
        c.drawImage(img_reader, x_img, y_img, width=marker_pt, height=marker_pt, mask="auto")

        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x0 + cell_w / 2, y0 + 5 * mm, f"ID {marker_id}")


def main() -> None:
    c = canvas.Canvas(str(OUT_PDF), pagesize=A4)

    for marker_size_mm in PAGE_SIZES_MM:
        _draw_page(c, marker_size_mm)
        c.showPage()

    c.save()
    print(f"Wrote {OUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
