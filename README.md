# 2-player Ping Pong game with Kuka iiwa R820 arm + gantry

## Requirements

- Python 3.12
- macOS (Apple Silicon or Intel) or Ubuntu 20.04+

## Installation

### 1. Create and activate an environment

**macOS:**
```bash
python3.12 -m venv venv
source venv/bin/activate
```

**Ubuntu (conda):**

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed, then:
```bash
conda create -n sim2real python=3.12 -y
conda activate sim2real
```

### 2. Install MuJoCo

```bash
pip install mujoco
```

MuJoCo 3.x is installed as a Python package — no separate binary installation needed.

### 3. Install remaining dependencies

```bash
pip install stable-baselines3 torch gymnasium scipy numpy
```

**Ubuntu only** — install OpenGL/GLFW system libraries required by MuJoCo's renderer:

```bash
sudo apt install -y libgl1-mesa-dev libglfw3 libglew-dev
```

### 4. (macOS only) Set up mjpython

MuJoCo's passive viewer requires the `mjpython` launcher on macOS to ensure the window is created on the main thread. It ships with the `mujoco` package:

```bash
# Find mjpython
python -c "import mujoco, os; print(os.path.join(os.path.dirname(mujoco.__file__), 'mjpython'))"
```

Add the printed directory to your PATH permanently:

```bash
# Add to ~/.zshrc
export PATH="$(python -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")":$PATH
```

## Running

**macOS** — use `mjpython` instead of `python`, otherwise the viewer crashes with `NSWindow should only be instantiated on the main thread`:

```bash
mjpython comp.py
```

**Ubuntu** — plain `python` works:

```bash
python comp.py
```

## SB3 deserialization warnings

When loading the PPO model you may see:

```
UserWarning: Could not deserialize object clip_range / lr_schedule.
Exception: code() argument 13 must be str, not int
```

These are harmless. They occur because the model was saved on Python ≤3.10 and the `code()` constructor signature changed in Python 3.12. Policy inference is unaffected.

## File structure

```
lfr-project/
├── comp.py                  # Main competition script
├── mujoco_env_comp.py       # MuJoCo Gymnasium environment
├── model_arch.py            # SimpleModel neural network definition
├── models/
│   ├── model1.pth           # Learned Q-value model for player 1 (This is pretty old, may be wrong, please consider training new one)
│   ├── model2.pth           # Learned Q-value model for player 2 (This is pretty old, may be wrong, please consider training new one)
│   └── model_p.pth          # Nash-p joint strategy model (This is pretty old, may be wrong, please consider training new one)
└── logs/
    └── best_model_tracker_slow/best_model   # PPO policy checkpoint (I just trained this, so should be good to run comp.py wiht random strategy)
```
