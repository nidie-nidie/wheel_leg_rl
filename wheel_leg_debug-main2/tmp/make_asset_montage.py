from pathlib import Path
from PIL import Image, ImageDraw

files = sorted(Path("mujoco_control_extract/.playwright-cli").glob("*.png"))
tiles = []
for path in files:
    image = Image.open(path).convert("RGB")
    image.thumbnail((480, 270))
    tile = Image.new("RGB", (500, 310), "white")
    tile.paste(image, ((500 - image.width) // 2, 10))
    ImageDraw.Draw(tile).text((10, 285), path.name, fill="black")
    tiles.append(tile)

rows = (len(tiles) + 1) // 2
out = Image.new("RGB", (1000, rows * 310), (230, 230, 230))
for index, tile in enumerate(tiles):
    out.paste(tile, ((index % 2) * 500, (index // 2) * 310))
out.save("tmp/playwright_pages_montage.png")
