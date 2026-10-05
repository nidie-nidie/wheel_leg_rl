import importlib
import pathlib
import sys

print(f"python={sys.executable}")
print(f"version={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info[:2] != (3, 10):
    raise SystemExit("expected Python 3.10")
if sys.prefix == sys.base_prefix:
    raise SystemExit("the interpreter is not a virtual environment")

for name in ("isaacsim", "isaaclab", "isaaclab_rl", "rsl_rl"):
    module = importlib.import_module(name)
    print(f"import {name}=ok ({pathlib.Path(module.__file__).resolve()})")

import gymnasium as gym
import serial_leg_rl.tasks

spec = gym.spec("SerialLeg-Standing-Direct-v0")
print(f"task={spec.id}")
if spec.id != "SerialLeg-Standing-Direct-v0":
    raise SystemExit("serial-leg task registration mismatch")

usd = pathlib.Path("wheel_leg_urdf4_usd (1)/wheel_leg_urdf4/wheel_leg_urdf4.usd").resolve()
print(f"usd={usd}")
if not usd.is_file():
    raise SystemExit(f"missing USD asset: {usd}")

import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(f"gpu={torch.cuda.get_device_name(0)}")
