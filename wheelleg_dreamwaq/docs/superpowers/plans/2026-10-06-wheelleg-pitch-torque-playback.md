# WheelLeg Pitch-Torque Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hold-to-apply `+/-4.0 Nm` pitch disturbance keys to native PPO playback.

**Architecture:** `scripts/play.py` owns the keyboard state and resolves the existing `base_link` body. Before every environment step it writes a pure local-frame torque through Isaac Lab's permanent wrench composer, leaving environment and actuator code unchanged.

**Tech Stack:** Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2, PyTorch, RSL-RL 3.1.2.

---

### Task 1: Keyboard Pitch Torque

**Files:**
- Modify: `scripts/play.py`

- [x] Add and validate `--pitch-torque-nm` with default `4.0`.
- [x] Track `J` and `L` press/release state and clear it for stop, reset, quit, and cleanup.
- [x] Resolve `base_link` uniquely and create the reusable zero-force and pitch-torque tensors.
- [x] Write the selected local-frame torque before every `env.step()` through `permanent_wrench_composer`.
- [x] Print concise press/release status without changing command controls.

### Task 2: Verification

**Files:**
- Test: `scripts/play.py`
- Test: `tests/unit`

- [x] Compile `scripts/play.py`.
- [x] Run the unit test suite.
- [x] Run bounded headless playback with an existing checkpoint.
- [x] Run native keyboard playback and verify both torque directions and release-to-zero behavior.
- [x] Review the final changes and confirm no training/environment/contract files changed.
