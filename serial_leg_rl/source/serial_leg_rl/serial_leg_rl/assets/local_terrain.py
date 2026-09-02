from __future__ import annotations

from pathlib import Path

from serial_leg_rl.assets.sim_urdf import GENERATED_ASSET_DIR


LOCAL_FLAT_TERRAIN_USD_PATH = GENERATED_ASSET_DIR / "flat_terrain.usda"


def prepare_local_flat_terrain_usd(size: float = 200.0) -> Path:
    GENERATED_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    half_size = size * 0.5
    terrain_usd = f"""#usda 1.0
(
    defaultPrim = "terrain"
    metersPerUnit = 1
)

def Xform "terrain"
{{
    def Mesh "collision_mesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        point3f[] points = [
            (-{half_size}, -{half_size}, 0),
            ({half_size}, -{half_size}, 0),
            ({half_size}, {half_size}, 0),
            (-{half_size}, {half_size}, 0),
        ]
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
        uniform token subdivisionScheme = "none"
        token physics:approximation = "none"
    }}
}}
"""
    if not LOCAL_FLAT_TERRAIN_USD_PATH.exists() or LOCAL_FLAT_TERRAIN_USD_PATH.read_text(
        encoding="utf-8"
    ) != terrain_usd:
        LOCAL_FLAT_TERRAIN_USD_PATH.write_text(terrain_usd, encoding="utf-8")
    return LOCAL_FLAT_TERRAIN_USD_PATH
