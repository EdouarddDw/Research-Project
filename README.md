# Research Project
## synth.py usage

```python
    X, Y, ground_truth = synth.functions[3](num_samples=30000, seed=42, noise_std=0.1)
```
don't forget to set the noise_std (start off with 0.1 and go up from there

All required Python packages are listed in `requirements.txt`.

## Setup

### 1. Create a virtual environment

Mac / Linux:

```bash
python3 -m venv .venv
```

Windows:

```bash
py -m venv .venv
```

### 2. Activate the virtual environment

Mac / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\\Scripts\\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

