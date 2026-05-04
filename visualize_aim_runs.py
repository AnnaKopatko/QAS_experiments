"""
plot_runs.py
------------
Place next to your .aim folder and run:  python3 plot_runs.py
Output: energy_comparison.png

Filters: mol_name == "LiH"  AND  noise == True
"""

from aim import Repo, Run
from aim.storage.context import Context
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── config ───────────────────────────────────────────────────────────────────
METRIC   = "finetune_energy"
CONTEXT  = Context({"stage": "finetune"})

MOL_NAME = "LiH"
NOISE    = True

TAGS_A   = {"3rots2cnots_random_1",
            "3rots2cnots_random_2",
            "3rots2cnots_random_3"}

TAGS_B   = {"3rots2cnots_popsize50_1",
            "3rots2cnots_popsize50_2",
            "3rots2cnots_popsize50_3"}

COLOR_A  = "#90EE90"   # pale green
COLOR_B  = "#FFCC99"   # pale orange
ALPHA    = 0.9
LW       = 1.8

# ── fetch ─────────────────────────────────────────────────────────────────────
repo = Repo(".")
curves_a, curves_b = {}, {}

for run_meta in repo.iter_runs():
    try:
        tag = run_meta["expr_tag"]
    except Exception:
        continue

    if tag not in (TAGS_A | TAGS_B):
        continue

    try:
        run = Run(run_meta.hash, repo=repo, read_only=True)

        # ── filters ──────────────────────────────────────────────────────────
        if run.get("mol_name") != MOL_NAME:
            print(f"  skip {tag} — mol_name={run.get('mol_name')!r}")
            continue
        if run.get("noise") != NOISE:
            print(f"  skip {tag} — noise={run.get('noise')!r}")
            continue

        metric = run.get_metric(METRIC, context=CONTEXT)
        if metric is None:
            print(f"  [warn] metric not found in {tag}")
            continue
        df = metric.dataframe()
        if df.empty:
            print(f"  [warn] empty dataframe for {tag}")
            continue

        steps, values = df["step"].tolist(), df["value"].tolist()
        (curves_a if tag in TAGS_A else curves_b)[tag] = (steps, values)
        print(f"  ✓  {tag}  ({len(steps)} steps)")

    except Exception as e:
        print(f"  [err] {tag}: {e}")

print(f"\nGroup A: {len(curves_a)}/3   Group B: {len(curves_b)}/3")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for steps, values in curves_a.values():
    ax.plot(steps, values, color=COLOR_A, alpha=ALPHA, linewidth=LW)

for steps, values in curves_b.values():
    ax.plot(steps, values, color=COLOR_B, alpha=ALPHA, linewidth=LW)

leg_a = mlines.Line2D([], [], color=COLOR_A, linewidth=2.5,
                      label=f"random init  (n={len(curves_a)})")
leg_b = mlines.Line2D([], [], color=COLOR_B, linewidth=2.5,
                      label=f"popsize-50 evolution  (n={len(curves_b)})")
ax.legend(handles=[leg_a, leg_b], fontsize=11, framealpha=0.85)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Energy (Ha)", fontsize=12)
ax.set_title(f"Fine-tune energy — {MOL_NAME}, noise={NOISE}\n"
             f"random init vs. popsize-50 evolution", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("energy_comparison.png", dpi=150)
print("Saved → energy_comparison.png")
plt.show()