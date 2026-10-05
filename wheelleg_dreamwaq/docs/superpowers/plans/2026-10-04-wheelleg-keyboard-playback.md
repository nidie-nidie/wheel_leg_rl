# WheelLeg Keyboard Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide native Isaac Sim keyboard control for WheelLeg PPO playback without loading the failing RTX sensor extension.

**Architecture:** A project-local Kit Experience controls startup dependencies. The existing playback script owns keyboard events and continues writing the existing three-element command tensor.

**Tech Stack:** Isaac Sim 5.1.0, Isaac Lab 2.3.2, PyTorch, RSL-RL 3.1.2, Python 3.11.

---

### Task 1: Playback Experience

**Files:**
- Create: `apps/wheelleg_play_no_rtx.kit`

- [x] Copy the current Isaac Lab GUI Experience into the project.
- [x] Remove only the `isaacsim.sensors.rtx` dependency and identify the file as WheelLeg playback.
- [x] Verify the resulting dependency list retains viewport, renderer, app window, and PhysX extensions.

### Task 2: Keyboard Controller

**Files:**
- Modify: `scripts/play.py`

- [x] Select the project-local Experience before constructing `AppLauncher`.
- [x] Add concise native keyboard event handling with command clamping.
- [x] Integrate reset, quit, command overwrite, and cleanup into the existing evaluation loop.
- [x] Preserve fixed-command and video behavior when `--keyboard` is absent.

### Task 3: Verification

**Files:**
- Test: `tests/unit`
- Runtime: `scripts/play.py`

- [x] Compile the modified Python script.
- [x] Run the existing unit test suite.
- [x] Launch bounded native GUI keyboard playback and confirm it avoids the RTX sensor startup failure.
- [x] Review the final diff and run an independent code-review agent.
