from pathlib import Path

import pymupdf
from PIL import Image

root = Path(__file__).resolve().parents[3]
assets = Path(__file__).resolve().parent / "assets"

# High-resolution crop of the control-route diagram from page 2.
pdf = pymupdf.open(root / "MOTION_CONTROL_REPORT.pdf")
page = pdf[1]
clip = pymupdf.Rect(64, 76, 548, 530)
pix = page.get_pixmap(matrix=pymupdf.Matrix(3.2, 3.2), clip=clip, alpha=False)
pix.save(assets / "control_route.png")

# Tight crop of the MuJoCo-only render for the cover panel.
mujoco = Image.open(assets / "mujoco_suspended.png").convert("RGB")
mujoco.crop((205, 95, 1395, 910)).save(assets / "mujoco_suspended_crop.png", quality=95)

print("prepared", assets)
