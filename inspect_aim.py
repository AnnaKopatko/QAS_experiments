"""
inspect_aim.py  —  run once to see metric names + contexts
"""
from aim import Repo, Run

repo = Repo(".")

for run_meta in repo.iter_runs():
    try:
        tag = run_meta["expr_tag"]
    except Exception:
        continue

    if "3rots2cnots_random_1" not in str(tag):
        continue

    run = Run(run_meta.hash, repo=repo, read_only=True)
    print(f"\nrun hash : {run_meta.hash}")
    print(f"expr_tag : {tag}")
    print("metrics  :")
    for m in run.metrics():
        # context may be a custom object — print its repr and dir()
        ctx = m.context
        print(f"  name={m.name!r}")
        print(f"    context repr  : {repr(ctx)}")
        print(f"    context type  : {type(ctx)}")
        try:
            print(f"    context.to_dict(): {ctx.to_dict()}")
        except Exception:
            pass
        try:
            print(f"    context.__dict__: {ctx.__dict__}")
        except Exception:
            pass
        try:
            print(f"    vars(context)   : {vars(ctx)}")
        except Exception:
            pass
    break