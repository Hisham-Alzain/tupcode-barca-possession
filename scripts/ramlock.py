"""Run one memory-heavy command at a time across parallel agent worktrees.

    uv run python scripts/ramlock.py -- uv run python scripts/evaluate.py --variant X --bench real2 --no-public

Takes a machine-wide lock (atomic mkdir), checks free RAM >= 1.5 GB, runs the command in the current
directory, releases the lock. If the lock or the RAM is not available within WAIT seconds it exits
with code 75: nothing was run, call it again. A lock older than STALE seconds is treated as dead.
"""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time

LOCK = os.path.join(tempfile.gettempdir(), "barca_possession_ram.lock")
WAIT, STALE, MIN_GB = 420, 1200, 1.5


def free_gb():
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("a", ctypes.c_ulong), ("t", ctypes.c_ulonglong),
                    ("f", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong * 5)]
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.f / 1e9


def main():
    if "--" not in sys.argv:
        sys.exit("usage: ramlock.py -- <command ...>")
    cmd = sys.argv[sys.argv.index("--") + 1:]
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > STALE:
                    shutil.rmtree(LOCK, ignore_errors=True)
                    continue
            except OSError:
                continue
            if time.time() - t0 > WAIT:
                print("ramlock: busy (another agent is running a heavy job); nothing run, retry", file=sys.stderr)
                sys.exit(75)
            time.sleep(10)
    try:
        while free_gb() < MIN_GB:
            if time.time() - t0 > WAIT:
                print("ramlock: free RAM %.1f GB < %.1f; nothing run, retry later" % (free_gb(), MIN_GB), file=sys.stderr)
                sys.exit(75)
            time.sleep(10)
        with open(os.path.join(LOCK, "owner.txt"), "w") as fh:
            fh.write("%s\n%s\n" % (os.getcwd(), " ".join(cmd)))
        sys.exit(subprocess.call(cmd))
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)


if __name__ == "__main__":
    main()
