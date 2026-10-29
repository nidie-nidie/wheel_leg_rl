"""Fetch the pinned portable official Arm GNU compiler into PACE/build only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

PACE = Path(__file__).resolve().parents[1]
DESTINATION = PACE / "build/toolchain"
URL = "https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-mingw-w64-x86_64-arm-none-eabi.zip"
EXPECTED_SHA256 = "864c0c8815857d68a1bbba2e5e2782255bb922845c71c97636004a3d74f60986"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archive = DESTINATION / "arm-gnu-toolchain-14.3.rel1.zip"
    if not archive.exists():
        print("Downloading pinned Arm GNU compiler into PACE/build (no system install)", flush=True)
        urllib.request.urlretrieve(URL, archive)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError("Compiler archive SHA256 does not match the pinned artifact")
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            destination = (DESTINATION / member.filename).resolve()
            if not destination.is_relative_to(DESTINATION.resolve()):
                raise RuntimeError("Archive member escapes the toolchain directory")
            if member.filename.startswith(("bin/", "lib/gcc/", "libexec/gcc/", "arm-none-eabi/include/", "arm-none-eabi/lib/", "share/doc/")) or member.filename == ".version":
                package.extract(member, DESTINATION)
    compiler = DESTINATION / "bin/arm-none-eabi-gcc.exe"
    if not compiler.exists():
        raise RuntimeError("The pinned archive lacks its Windows compiler")
    (DESTINATION / "download.json").write_text(json.dumps({"url": URL, "sha256": actual}, indent=2) + "\n", encoding="utf-8")
    print(f"Portable compiler verified: {compiler}", flush=True)


if __name__ == "__main__":
    main()
