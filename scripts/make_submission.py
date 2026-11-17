"""Prepare the final submission: fill team/repo placeholders, copy + validate the CSV, render the PDF.

Usage:
    uv run --with markdown python scripts/make_submission.py --team "<Team Name>" \
        [--csv submissions/draft_2026-11-17f.csv] \
        [--repo https://github.com/Hisham-Alzain/barca-possession] [--dry-run]

--dry-run copies the text files into a temp folder and does everything there
(replacements, CSV copy + validation, PDF); no tracked file and nothing in
submissions/ is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM_PH = "<TEAM NAME>"
REPO_PH = "<PUBLIC REPO URL>"
# file -> placeholders it is expected to contain (as of 2026-11-18)
TARGETS = {
    "docs/TECHNICAL.md": (TEAM_PH, REPO_PH),
    "docs/TECHNICAL_SHORT.md": (TEAM_PH, REPO_PH),
    "README.md": (TEAM_PH, REPO_PH),
    "LICENSE": (TEAM_PH,),
}
VALIDATOR = ROOT / "vendor" / "mtah" / "validate_submission.py"
DEFAULT_CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"
CSS = """body{font-family:Segoe UI,Arial,sans-serif;font-size:10.5pt;line-height:1.42;color:#111;margin:0}
h1{font-size:19pt;margin:0 0 6pt}h2{font-size:13.5pt;margin:16pt 0 5pt;border-bottom:1px solid #999;padding-bottom:2pt}
h3{font-size:11.5pt;margin:12pt 0 4pt}table{border-collapse:collapse;width:100%;margin:7pt 0;font-size:9pt}
th,td{border:1px solid #888;padding:3pt 5pt;vertical-align:top;text-align:left}th{background:#eee}
code{font-family:Consolas,monospace;font-size:9pt;background:#f2f2f2;padding:0 2pt}
pre{background:#f2f2f2;padding:6pt;font-size:8.5pt;white-space:pre-wrap}tr,pre{page-break-inside:avoid}
@page{size:A4;margin:16mm 15mm}"""


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def safe_stem(team: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]", "", team.strip().replace(" ", "_"))
    if not stem:
        die(f"team name {team!r} gives an empty file stem")
    return stem


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_in_file(path: Path, expected: tuple[str, ...], values: dict[str, str]) -> bool:
    """Replace placeholders in-place (bytes, UTF-8, newlines untouched). Returns True if changed."""
    if not path.is_file():
        die(f"missing file: {path}")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    for ph in expected:
        if ph not in text and values[ph] not in text:
            die(f"{path.name}: placeholder {ph} not found and replacement value {values[ph]!r} absent")
    new = text
    for ph in expected:
        new = new.replace(ph, values[ph])
    leftover = [ph for ph in (TEAM_PH, REPO_PH) if ph in new]
    if leftover:
        die(f"{path.name}: unexpected placeholders remain: {leftover}")
    if new != text:
        path.write_bytes(new.encode("utf-8"))
        return True
    return False


def apply_replacements(base: Path, values: dict[str, str]) -> list[str]:
    return [rel for rel, exp in TARGETS.items() if replace_in_file(base / rel, exp, values)]


def validate_csv(path: Path) -> None:
    r = subprocess.run([sys.executable, str(VALIDATOR), str(path)], cwd=str(ROOT),
                       capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print(r.stderr.rstrip(), file=sys.stderr)
    if r.returncode != 0 or not r.stdout.lstrip().startswith("OK"):
        die(f"validator rejected {path} (exit {r.returncode})")


def render_pdf(md_path: Path, html_path: Path, pdf_path: Path, title: str) -> None:
    import markdown

    body = markdown.markdown(md_path.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>%s</title>"
            "<style>%s</style></head><body>%s</body></html>" % (title, CSS, body))
    html_path.write_text(html, encoding="utf-8")
    chrome = os.environ.get("CHROME", DEFAULT_CHROME)
    if not Path(chrome).is_file():
        die(f"Chrome not found at {chrome} (set CHROME env var)")
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        die(f"PDF not produced: {pdf_path}")
    size = pdf_path.stat().st_size
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(pdf_path)).pages)
        how = "pypdf"
    except ImportError:
        pages = len(re.findall(rb"/Type\s*/Page(?!s)", pdf_path.read_bytes()))
        how = "/Type /Page count"
    print(f"PDF: {pdf_path}  {size:,} bytes  {pages} pages ({how})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True)
    ap.add_argument("--csv", default="submissions/draft_2026-11-17f.csv")
    ap.add_argument("--repo", default="https://github.com/Hisham-Alzain/barca-possession")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    team = args.team.strip()
    stem = safe_stem(team)
    values = {TEAM_PH: team, REPO_PH: args.repo}
    src_csv = (ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    if not src_csv.is_file():
        die(f"CSV not found: {src_csv}")

    if args.dry_run:
        base = Path(tempfile.mkdtemp(prefix="make_submission_"))
        for rel in TARGETS:
            src = ROOT / rel
            dst = base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copyfile(src, dst)
            elif rel == "docs/TECHNICAL_SHORT.md":
                print(f"NOTE: {rel} absent in repo; using a temp stand-in (dry run only)")
                dst.write_bytes(b"# Technical summary\n\nTeam: **<TEAM NAME>**\n\n"
                                b"Repository: <PUBLIC REPO URL>\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
            else:
                die(f"missing file: {src}")
        out_dir = base / "submissions"
        out_dir.mkdir()
        print(f"DRY RUN in temp folder: {base}")
    else:
        base = ROOT
        out_dir = ROOT / "submissions"

    changed = apply_replacements(base, values)
    print(f"Replaced placeholders in: {changed or 'nothing (already filled)'}")
    if args.dry_run:
        again = apply_replacements(base, values)
        if again:
            die(f"replacement not idempotent: second pass changed {again}")
        print("Idempotency check: second pass changed nothing")

    dst_csv = out_dir / f"{stem}.csv"
    shutil.copyfile(src_csv, dst_csv)
    validate_csv(dst_csv)
    h_src, h_dst = sha256(src_csv), sha256(dst_csv)
    print(f"sha256 source {h_src}  {src_csv}")
    print(f"sha256 copy   {h_dst}  {dst_csv}")
    if h_src != h_dst:
        die("CSV copy hash mismatch")

    render_pdf(base / "docs" / "TECHNICAL_SHORT.md", out_dir / f"{stem}_TECHNICAL_SHORT.html",
               out_dir / f"{stem}.pdf", f"{team} - technical document")

    print("\nRemaining manual steps:")
    print("  1. Review the diff: git diff (TECHNICAL.md, TECHNICAL_SHORT.md, README.md, LICENSE)")
    print(f"  2. Commit (incl. submissions/{stem}.csv / .pdf if wanted) and push")
    print(f"  3. Confirm the repository is public: {args.repo}")
    print(f"  4. Upload submissions/{stem}.csv and submissions/{stem}.pdf in the form")
    if args.dry_run:
        print(f"\n(dry run: outputs are in {base}; repo untouched)")


if __name__ == "__main__":
    main()
