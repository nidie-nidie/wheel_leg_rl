from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


SERIAL_LEG_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.005, 0.045),
            noise_step=0.005,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.08),
            step_width=0.35,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.08),
            step_width=0.35,
            platform_width=2.0,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_width_range=(0.2, 0.6),
            obstacle_height_range=(0.01, 0.08),
            num_obstacles=20,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
