"""
export_best_energy.py
---------------------
Place next to your .aim folder and run:  python3 export_best_energy.py
Output: best_energy_results.txt
"""

from aim import Repo, Run

# ═══════════════════════════════════════════════════════════════════════════════
#  EDIT THIS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

METRICS  = ["finetune_best_energy", "num_gates"]
MOL_NAME = "LiH"
NOISE    = True
FILTER_THESIS_TAG = True

# Each entry: (tags_list, group_label, check_noise)
GROUPS = [
    (
        ["classic_vqe_run_1", "classic_vqe_run_2", "classic_vqe_run_3"],
        "Classic VQE",
        False,
    ),
    (
        ["2cnot3rot_1", "2cnot3rot_2", "2cnot3rot_3"],
        "Genome structure",
        True,
    ),
    (
        ["1cnot3rot_block_1", "1cnot3rot_block_2", "1cnot3rot_block_3"],
        "Block structure",
        True,
    ),
    (
        ["2cnot3rot_dropout_safe_controller_1",
         "2cnot3rot_dropout_safe_controller_2",
         "2cnot3rot_dropout_safe_controller_3"],
        "Genome + dropout + controller",
        True,
    ),
]

OUTPUT_FILE = "best_energy_results.txt"

# ═══════════════════════════════════════════════════════════════════════════════

tag_to_group = {}
for idx, (tags, label, _) in enumerate(GROUPS):
    for t in tags:
        tag_to_group[t] = idx

all_tags = set(tag_to_group.keys())
# results[group_index] = list of (tag, hash, {metric_name: value})
results  = {i: [] for i in range(len(GROUPS))}

def get_last_metric_value(run, metric_name):
    """Return the last logged value of a metric, or None if not found."""
    for m in run.metrics():
        if m.name == metric_name:
            df = m.dataframe()
            if not df.empty:
                return df["value"].iloc[-1]
    return None

# ── fetch ─────────────────────────────────────────────────────────────────────
repo = Repo(".")
try:
    run_iterator = repo.iter_runs(num_runs=-1)
except TypeError:
    run_iterator = repo.iter_runs()

for run_meta in run_iterator:
    try:
        tag = run_meta["expr_tag"]
    except Exception:
        continue

    if tag not in all_tags:
        continue

    try:
        run = Run(run_meta.hash, repo=repo, read_only=True)

        if run.get("mol_name") != MOL_NAME:
            continue

        g = tag_to_group[tag]
        if GROUPS[g][2] and run.get("noise") is not NOISE:
            continue

        if FILTER_THESIS_TAG and "used_in_thesis" not in run.tags:
            print(f"  skip {tag} ({run_meta.hash[:6]}) — missing 'used_in_thesis'")
            continue

        values = {}
        for metric_name in METRICS:
            v = get_last_metric_value(run, metric_name)
            if v is None:
                print(f"  [warn] '{metric_name}' not found in {tag} ({run_meta.hash[:6]})")
            values[metric_name] = v

        results[g].append((tag, run_meta.hash[:6], values))
        summary = "  ".join(f"{k}={v:.6f}" if v is not None else f"{k}=N/A"
                            for k, v in values.items())
        print(f"  ✓  {tag} ({run_meta.hash[:6]})  {summary}")

    except Exception as e:
        print(f"  [err] {tag} ({run_meta.hash[:6]}): {e}")

# ── write output ──────────────────────────────────────────────────────────────
noise_label = "with noise" if NOISE else "no noise"
col_w = 50

with open(OUTPUT_FILE, "w") as f:
    f.write(f"Molecule: {MOL_NAME}   Noise: {noise_label}\n")
    f.write("=" * 70 + "\n\n")

    for i, (tags, label, _) in enumerate(GROUPS):
        f.write(f"{label}:\n")
        # header
        header = f"  {'run':{col_w}s}" + "".join(f"  {m:>20s}" for m in METRICS)
        f.write(header + "\n")
        f.write("  " + "-" * (col_w + 22 * len(METRICS)) + "\n")

        if not results[i]:
            f.write("  (no runs found)\n")
        for tag, hash6, values in sorted(results[i], key=lambda x: x[0]):
            row = f"  {tag:{col_w}s}"
            for m in METRICS:
                v = values.get(m)
                row += f"  {v:>20.8f}" if v is not None else f"  {'N/A':>20s}"
            f.write(row + "\n")
        f.write("\n")

print(f"\nSaved → {OUTPUT_FILE}")