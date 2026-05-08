"""
plot_runs.py
------------
Place next to your .aim folder and run:  python3 plot_runs.py
Output: energy_comparison.png

Each group: (expr_tags, legend_label, colour, check_noise)
  check_noise=True  → only include runs where noise == NOISE
  check_noise=False → skip the noise filter for this group
"""

from aim import Repo, Run
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ═══════════════════════════════════════════════════════════════════════════════
#  EDIT THIS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

METRIC   = "finetune_energy"
MOL_NAME = "LiH"
NOISE    = False       # set to True for noisy runs, False for no-noise runs
FILTER_THESIS_TAG = True

# Each entry: (tags_list, legend_label, colour, check_noise)
GROUPS = [
    (
        ["classic_vqe_run_1", "classic_vqe_run_2", "classic_vqe_run_3"],
        "Classic VQE",
        "#90EE90",   # pale green
        False,       # skip noise filter for this group
    ),
    (
        ["2cnot3rot_1", "2cnot3rot_2", "2cnot3rot_3"],
        "Genome structure",
        "#FFCC99",   # pale orange
        True,
    ),
    (
        ["block_structure_1", "block_structure_2", "block_structure_3"],
        "Block structure",
        "#ADD8E6",   # pale blue
        True,
    ),
    (
        ["2cnot3rot_dropout_RC_safe_controller_1",
         "2cnot3rot_dropout_RC_safe_controller_2",
         "2cnot3rot_dropout_RC_safe_controller_3"],
        "Genome + dropout + controller",
        "#DDA0DD",   # pale purple
        True,
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.9
LW    = 1.8

tag_to_group = {}
for idx, (tags, label, colour, _) in enumerate(GROUPS):
    for t in tags:
        tag_to_group[t] = idx

all_tags = set(tag_to_group.keys())
curves   = {i: {} for i in range(len(GROUPS))}

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
        check_noise = GROUPS[g][3]
        if check_noise and run.get("noise") is not NOISE:
            continue

        if FILTER_THESIS_TAG and "used_in_thesis" not in run.tags:
            print(f"  skip {tag} ({run_meta.hash[:6]}) — missing 'used_in_thesis'")
            continue

        df = None
        for m in run.metrics():
            if m.name == METRIC:
                candidate = m.dataframe()
                if not candidate.empty:
                    df = candidate
                    break

        if df is None:
            print(f"  [warn] '{METRIC}' not found in {tag} ({run_meta.hash[:6]})")
            continue

        steps, values = df["step"].tolist(), df["value"].tolist()
        curves[g][run_meta.hash] = (steps, values)
        print(f"  ✓  {tag} ({run_meta.hash[:6]})  {len(steps)} steps  → '{GROUPS[g][1]}'")

    except Exception as e:
        print(f"  [err] {tag} ({run_meta.hash[:6]}): {e}")

print()
for i, (tags, label, colour, _) in enumerate(GROUPS):
    print(f"  {label}: {len(curves[i])}/{len(tags)} runs loaded")

# ── plot ──────────────────────────────────────────────────────────────────────
noise_label = "with noise" if NOISE else "no noise"
fig, ax = plt.subplots(figsize=(10, 5))

legend_handles = []
for i, (tags, label, colour, _) in enumerate(GROUPS):
    if not curves[i]:
        continue
    for steps, values in curves[i].values():
        ax.plot(steps, values, color=colour, alpha=ALPHA, linewidth=LW)
    handle = mlines.Line2D([], [], color=colour, linewidth=2.5,
                           label=f"{label}  (n={len(curves[i])})")
    legend_handles.append(handle)

ax.legend(handles=legend_handles, fontsize=11, framealpha=0.85)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Energy (Ha)", fontsize=12)
ax.set_title(f"Fine-tune energy — {MOL_NAME}, {noise_label}", fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("energy_comparison.png", dpi=150)
print("\nSaved → energy_comparison.png")
plt.show()