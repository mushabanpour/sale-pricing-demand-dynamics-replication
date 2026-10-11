"""Run manifests. Records what a run was, so a number can be traced back to it.

Motivated by audit rounds 02 and 04: no existing run recorded its arguments, so
two rounds were spent reconstructing which output directory produced Table 8 and
inferring the value of varsigma.

Usage
-----
    from run_manifest import RunManifest
    with RunManifest(out_dir, args=vars(args), inputs=[p1, p2], tag="stage2 S0") as rm:
        ...
        rm.add_start_record(start=0, objective=-1591.25, converged=True, iters=164)
        ...
        rm.add_outputs([out1, out2])

Writes outputs/runs/<run_id>/manifest.json and appends one line to
registry/runs.jsonl. Never touches an existing output directory.
"""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import _paths

PAPER_ROOT = _paths.WORK
RUNS_DIR = PAPER_ROOT / "outputs" / "runs"
RUNS_JSONL = PAPER_ROOT / "registry" / "runs.jsonl"

NUMERIC_PACKAGES = ["numpy", "pandas", "scipy", "duckdb", "pyarrow", "statsmodels"]


def sha256(path, chunk=1 << 22):
    """SHA256 of a file. Large parquet files are chunked."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError as e:
        return f"UNREADABLE: {e}"
    return h.hexdigest()


def git_state(cwd=None):
    """Returns the commit and cleanliness, or an explicit not-under-version-control marker."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd or PAPER_ROOT,
                             capture_output=True, text=True, timeout=10)
        if top.returncode != 0:
            return {"under_version_control": False,
                    "note": "NOT A GIT REPOSITORY. Provenance cannot be pinned to a commit."}
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd or PAPER_ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd or PAPER_ROOT,
                                capture_output=True, text=True, timeout=30).stdout
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd or PAPER_ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return {"under_version_control": True, "toplevel": top.stdout.strip(),
                "branch": branch, "commit": commit, "clean": status.strip() == "",
                "dirty_files": [l[3:] for l in status.splitlines()][:50]}
    except Exception as e:
        return {"under_version_control": False, "note": f"git probe failed: {e}"}


def package_versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in NUMERIC_PACKAGES:
        try:
            m = __import__(p)
            out[p] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[p] = None
    return out


class RunManifest:
    def __init__(self, out_dir, args=None, inputs=None, tag="", seed=None, n_starts=None):
        self.out_dir = Path(out_dir).resolve()
        self.args = {k: str(v) for k, v in (args or {}).items()}
        self.inputs = [str(p) for p in (inputs or [])]
        self.tag = tag
        self.seed = seed
        self.n_starts = n_starts
        self.starts = []
        self.outputs = []
        self.notes = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        h = hashlib.sha1(f"{stamp}{self.out_dir}{tag}".encode()).hexdigest()[:8]
        self.run_id = f"{stamp}-{h}"
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        self.started = datetime.now(timezone.utc).isoformat()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / self.run_id).mkdir(parents=True, exist_ok=True)
        return self

    def add_start_record(self, **kw):
        self.starts.append(kw)

    def add_note(self, s):
        self.notes.append(s)

    def add_outputs(self, paths):
        self.outputs.extend(str(p) for p in paths)

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "tag": self.tag,
            "git": git_state(),
            "args": self.args,
            "seed": self.seed,
            "n_starts": self.n_starts,
            "packages": package_versions(),
            "inputs": [{"path": p, "sha256": sha256(p),
                        "bytes": (os.path.getsize(p) if os.path.exists(p) else None)}
                       for p in self.inputs],
            "outputs": [{"path": p, "sha256": sha256(p),
                         "bytes": (os.path.getsize(p) if os.path.exists(p) else None)}
                        for p in self.outputs],
            "starts": self.starts,
            "notes": self.notes,
            "started_utc": self.started,
            "ended_utc": self.ended,
            "wall_clock_seconds": self.wall,
            "out_dir": str(self.out_dir),
            "manifest_dir": str(RUNS_DIR / self.run_id),
        }

    def __exit__(self, exc_type, exc, tb):
        self.ended = datetime.now(timezone.utc).isoformat()
        self.wall = round(time.time() - self._t0, 3)
        if exc_type is not None:
            self.notes.append(f"FAILED: {exc_type.__name__}: {exc}")
        d = self.to_dict()
        mp = RUNS_DIR / self.run_id / "manifest.json"
        mp.write_text(json.dumps(d, indent=2))
        RUNS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNS_JSONL, "a") as fh:
            fh.write(json.dumps({k: d[k] for k in
                     ("run_id", "tag", "started_utc", "wall_clock_seconds", "out_dir")}) + "\n")
        print(f"[run_manifest] wrote {mp}")
        return False
