# Basic Usage

This section provides a brief overview of the main PACE scripts and their most important command-line arguments. It is intended as a quick reference for running common workflows, without going into full implementation or theory details.

The typical workflow consists of:

1. Collecting excitation data from the robot or simulation
2. Running parameter identification using CMA-ES
3. Visualising and analysing the resulting trajectories and scores

---

## Overview of main scripts

| Script               | Purpose                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `data_collection.py` | Collects excitation data using a chirp signal for system identification |
| `fit.py`             | Runs the PACE parameter identification using CMA-ES                     |
| `plot_trajectory.py` | Visualises the resulting trajectories and optimization progress         |

---

## 1. Data collection

```bash
python scripts/pace/data_collection.py
```

This script generates excitation trajectories using a chirp signal and records the corresponding robot states for later parameter identification.

### Common arguments

| Argument          | Description                                   |
| ----------------- | --------------------------------------------- |
| `--num_envs`      | Number of parallel simulation environments    |
| `--task`          | Name of the task / robot configuration        |
| `--min_frequency` | Minimum frequency of the chirp signal (Hz)    |
| `--max_frequency` | Maximum frequency of the chirp signal (Hz)    |
| `--duration`      | Duration of the chirp signal in seconds       |
| `--device`        | Simulation device: `cpu`, `cuda`, or `cuda:N` |
| `--headless`      | Run without rendering                         |

### Example

```bash
python scripts/pace/data_collection.py \
  --task Isaac-Pace-Anymal-D-v0 \
  --num_envs 1 \
  --min_frequency 0.1 \
  --max_frequency 10 \
  --duration 20 \
  --device cuda
```

---

## 2. Parameter fitting

```bash
python scripts/pace/fit.py
```

This script performs system identification using CMA-ES to optimize physical parameters such as armature, damping, friction, bias, and delay.

### Common arguments

| Argument     | Description                            |
| ------------ | -------------------------------------- |
| `--num_envs` | Number of simulation environments      |
| `--task`     | Name of the task / robot               |
| `--device`   | Simulation device                      |
| `--headless` | Disable rendering for faster execution |

### Example

```bash
python scripts/pace/fit.py \
  --task Isaac-Pace-Anymal-D-v0 \
  --num_envs 4096 \
  --headless
```

---

## 3. Plotting trajectories and scores

```bash
python scripts/pace/plot_trajectory.py
```

This utility visualises the optimized trajectory and/or the evolution of the score over iterations. If no `folder_name` argument is given, the script will use the most recent evolution.

### Common arguments

| Argument            | Description                                     |
| ------------------- | ----------------------------------------------- |
| `--folder_name`     | Name of the optimization output folder          |
| `--mean_name`       | Name of the parameter file (e.g. `mean_020.pt`) |
| `--robot_name`      | Name of the robot                               |
| `--plot_trajectory` | Plot the optimized joint trajectories           |
| `--plot_score`      | Plot the score over iterations                  |

### Example

```bash
python scripts/pace/plot_trajectory.py \
  --folder_name 24_03_12_14-32-10 \
  --robot_name anymal_d_sim \
  --plot_trajectory 
```

---

## Notes

* All scripts share common Isaac Lab launcher arguments such as `--headless`, `--livestream`, and `--rendering_mode`.
* For full argument lists, run any script with:

```bash
python script_name.py --help
```

More advanced workflows and in-depth explanations are covered in the Examples and Guides sections.
