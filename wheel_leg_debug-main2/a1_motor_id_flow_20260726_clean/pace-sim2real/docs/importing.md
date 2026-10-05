# Importing PACE

This section explains the **recommended way to import the `pace_sim2real` package** in your own projects.

The import structure is intentionally designed to:

* keep the public API clean and future-proof,
* separate user-facing interfaces from internal implementation details,
* avoid accidental reliance on unstable deep module paths.

---

## Recommended public imports

For most use cases, you should import directly from the top-level package:

```python
from pace_sim2real import (
    PaceSim2realEnvCfg,
    PaceSim2realSceneCfg,
    PaceCfg,
    CMAESOptimizer,
)
```

These symbols form the **official public API** of PACE.

---

## Utilities and helper modules

Lower-level actuator models are intentionally namespaced under `utils` and should be imported explicitly:

```python
from pace_sim2real.utils import (
    PaceDCMotorCfg,
    PaceDCMotor,
)
```

---

## IsaacLab environment registration

To make PACE environments available in the Gym / IsaacLab registry, the task package must be imported **after the simulator is launched**.

```python
import pace_sim2real.tasks  # Registers PACE environments
```

A typical usage pattern looks like this:

```python
from isaaclab.app import AppLauncher
from pace_sim2real import PaceCfg

# Launch Isaac Sim / Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Register PACE environments
import pace_sim2real.tasks
```

After this, environments such as `Isaac-Pace-Anymal-D-v0` become available.

## Framework evolution & stability

PACE is an actively evolving research framework. While we aim for long-term stability, parts of the system may change as new capabilities, robots, and workflows are integrated.

### What you should expect

- The core concepts and philosophy of PACE will remain consistent.

- The public API is designed to be stable, but may receive deprecations or refinements as the framework matures.

- Internal implementation details and experimental features may change more frequently.

### We follow a best-effort approach to:

- clearly communicate breaking changes in release notes,

- provide migration hints when applicable,

- keep examples and documentation aligned with the latest recommended usage.

If you build long-term projects on top of PACE, we recommend pinning a specific version and consulting the changelog when upgrading.