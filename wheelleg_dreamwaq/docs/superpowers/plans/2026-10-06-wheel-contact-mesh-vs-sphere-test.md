# Wheel Contact Mesh vs Sphere Test Implementation Plan

**Goal:** Add an isolated MuJoCo model variant that changes only the two wheel-floor collision shapes from sphere proxies to the existing wheel meshes, then compare it with the frozen sphere baseline.

**Architecture:** Keep `wheel_leg_urdf4_v1.xml` and `model_manifest.json` unchanged. Add one explicitly named experimental XML beside the baseline, plus a structural regression test that proves the two models differ only in the model name and the two wheel collision geoms.

**Verification:** Compile both models, enforce the single-variable XML diff, and run the same reset plus 20 zero-action PD physics steps in both models.
