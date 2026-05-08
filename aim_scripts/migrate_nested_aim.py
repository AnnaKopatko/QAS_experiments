from aim import Repo

PROJ_ROOT = '/home/anna/Desktop/other/NAS_VQE_new_version/QAS_modern'

# Repo('/path') looks for .aim inside that path,
# so this opens .aim/.aim (the nested repo)
nested = Repo(f'{PROJ_ROOT}/.aim')
outer  = Repo(PROJ_ROOT)

outer_hashes = set(outer.list_all_runs())
nested_hashes = set(nested.list_all_runs())

unique_to_nested = sorted(nested_hashes - outer_hashes)
print(f"Outer repo:  {len(outer_hashes)} runs")
print(f"Nested repo: {len(nested_hashes)} runs")
print(f"Runs only in nested (to migrate): {len(unique_to_nested)}")
for h in unique_to_nested:
    print(f"  {h}")

if not unique_to_nested:
    print("\nNothing to migrate.")
else:
    print("\nCopying...")
    ok, failed = nested.copy_runs(unique_to_nested, outer)
    if ok:
        print(f"Done — {len(unique_to_nested)} runs copied into the main repo.")
    else:
        print(f"Completed with errors. Failed hashes: {failed}")
    print(f"\nOuter repo now has {outer.total_runs_count()} runs.")
