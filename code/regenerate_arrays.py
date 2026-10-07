"""Regenerate the three array files that are not shipped (they are large and reproduce bitwise).

- results/one-step/canonical-calibration/canonical_arrays.npz and
  results/one-step/canonical-validation/canonical_arrays.npz come from
  one_step_study.py --calibrate (about 2 s) and --validate (about 7 s).  The runner
  refuses to write into a folder that already holds a summary, so the shipped
  summaries are set aside while it runs and put back afterwards; the regenerated
  summaries differ from the shipped ones only in wall-clock time (and, for the
  validation run, in the hash of the calibration summary) and are discarded.
- results/verified-instances/verified-instances-full/arrays.npz comes from
  verified_instances.py --run (about 2 s) into a temporary folder, from which
  arrays.npz is moved into place.

Every regenerated file is compared with the SHA-256 recorded in
results/one-step/one_step_figure_audit.json and
results/verified-instances/verified_instances_figure_audit.json; files that are
already present and match are left alone.  A mismatch (possible on another
platform or with other library versions) is reported and the file is kept: the
plotting scripts still run, but the hash-pinned checks of exploration_study.py
and of the relabel scripts will then fail.

    python code/regenerate_arrays.py
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ONE_STEP = ROOT/"results/one-step"
VERIFIED = ROOT/"results/verified-instances"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(script, *arguments):
    print("+", "python", "code/"+script, *arguments, flush=True)
    subprocess.run([sys.executable, str(ROOT/"code"/script), *arguments], cwd=ROOT, check=True)


def report(path, expected):
    matches = sha256(path) == expected
    print(("MATCH   " if matches else "DIFFERS ")+str(path.relative_to(ROOT))+" vs. recorded SHA-256 "+expected[:12]+"...")
    return matches


def one_step():
    recorded = json.loads((ONE_STEP/"one_step_figure_audit.json").read_text())["artifact_hashes"]
    folders = {phase: ONE_STEP/("canonical-"+phase) for phase in ("calibration", "validation")}
    arrays = {phase: folder/"canonical_arrays.npz" for phase, folder in folders.items()}
    if all(path.is_file() and sha256(path) == recorded[phase]["canonical_arrays.npz"] for phase, path in arrays.items()):
        print("one-step arrays already present and matching")
        return True
    shipped = {phase: folder/"canonical_summary.json" for phase, folder in folders.items()}
    aside = {phase: folder/"canonical_summary.shipped.json" for phase, folder in folders.items()}
    for phase in folders:
        assert shipped[phase].is_file(), "missing shipped summary: "+str(shipped[phase])
        shipped[phase].rename(aside[phase])
        arrays[phase].unlink(missing_ok=True)
    try:
        run("one_step_study.py", "--calibrate")
        run("one_step_study.py", "--validate")
    finally:
        for phase in folders:   # discard the regenerated summaries, restore the shipped ones
            shipped[phase].unlink(missing_ok=True)
            aside[phase].rename(shipped[phase])
    return all([report(arrays[phase], recorded[phase]["canonical_arrays.npz"]) for phase in folders])


def verified_instances():
    recorded = json.loads((VERIFIED/"verified_instances_figure_audit.json").read_text())["arrays_sha256"]
    target = VERIFIED/"verified-instances-full/arrays.npz"
    if target.is_file() and sha256(target) == recorded:
        print("verified-instances arrays already present and matching")
        return True
    with tempfile.TemporaryDirectory(prefix="regenerated-", dir=VERIFIED) as temporary:
        output = Path(temporary)/"verified-instances-full"
        run("verified_instances.py", "--run", "--output", str(output))
        shutil.move(output/"arrays.npz", target)
    return report(target, recorded)


if __name__ == "__main__":
    ok = one_step()
    ok = verified_instances() and ok
    if not ok:
        raise SystemExit("A regenerated array file differs from the recorded hash (see above).")
    print("All regenerated array files match the recorded hashes.")
