# QAS_modern

# QAS Environment Setup

## 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -u
```

Follow the prompts and say `yes` to initialization. If you accidentally say `no`, run:

```bash
eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda init
```

Then reload your shell:

```bash
source ~/.bashrc
```

You should now see `(base)` in your terminal prompt.

---

## 2. Accept Conda Terms of Service

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## 3. Create and Activate the Environment

```bash
conda create -n qas python=3.9 -y
conda activate qas
```

---

## 4. Install All Packages

```bash
pip install pennylane==0.38.0
pip install autoray==0.6.7
pip install qiskit qiskit-aer
pip install pennylane-qiskit
pip install openfermionpyscf pyscf
pip install pymoo
pip install aim
pip install pyyaml
```

> **Note:** `autoray==0.6.7` and `pennylane==0.38.0` are pinned to avoid a known incompatibility where newer versions of `autoray` break `pennylane`'s numpy module.

---

## 5. Every Time You Start a New Terminal

```bash
conda activate qas
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `conda: command not found` | Run `source ~/.bashrc` or re-run `conda init` |
| `AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'` | Run `pip install autoray==0.6.7` |
| `ModuleNotFoundError: No module named 'yaml'` | Run `pip install pyyaml` |
| `CondaToSNonInteractiveError: Terms of Service` | Run the `conda tos accept` commands in step 2 |
| `ERROR: File or directory already exists` | Add `-u` flag: `bash Miniconda3-latest-Linux-x86_64.sh -u` |