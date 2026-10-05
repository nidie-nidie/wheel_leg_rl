import io
from pathlib import Path
import zipfile

from PIL import Image

task = Path("tmp/slides/robot_motion_interview_v2")
source = max((p for p in task.glob("*.pptx") if not p.name.startswith("deck_v2_")), key=lambda p: p.stat().st_mtime)
with zipfile.ZipFile(source) as archive:
    for name in archive.namelist():
        if not name.startswith("ppt/media/") or name.endswith("/"):
            continue
        payload = archive.read(name)
        print(name, len(payload), payload[:16])
        image = Image.open(io.BytesIO(payload))
        image.verify()
        print("  ", image.format, image.size)
