"""
list_all.py — prints every run with hash, expr_tag, mol_name, noise
"""
from aim import Repo, Run

repo = Repo(".")
try:
    run_iterator = repo.iter_runs(num_runs=-1)
except TypeError:
    run_iterator = repo.iter_runs()

TARGET_TAGS = {
    "2cnot3rot_1", "2cnot3rot_2", "2cnot3rot_3",
    "block_structure_1", "block_structure_2", "block_structure_3",
        "block_1", "block_2", "block_3",
    "2cnot3rot_dropout_RC_safe_controller_1",
    "2cnot3rot_dropout_RC_safe_controller_2",
    "2cnot3rot_dropout_RC_safe_controller_3",
}

for run_meta in run_iterator:
    try:
        tag = run_meta["expr_tag"]
    except Exception:
        continue
    if tag not in TARGET_TAGS:
        continue
    run = Run(run_meta.hash, repo=repo, read_only=True)
    print(f"  hash={run_meta.hash[:6]}  tag={tag!r:45s}  mol={run.get('mol_name')!r:8s}  noise={str(run.get('noise'))[:30]!r}")