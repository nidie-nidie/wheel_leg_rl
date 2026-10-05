"""Flat-ground asymmetric PPO environment for the wheel-leg robot.

The package initializer intentionally avoids importing Isaac Lab so pure math tests can
run without starting Isaac Sim. Runtime entry points import ``env`` and ``env_cfg`` directly.
"""
