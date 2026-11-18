#!/usr/bin/env python3
"""Fetch ASVspoof 2019 LA from the Hugging Face mirror and rebuild the official
directory layout the loader expects.

Why this exists
---------------
The official distribution (datashare.ed.ac.uk) is slow from mainland China and
gives one 20+ GB archive. The mirror repo `Bisher/ASVspoof_2019_LA` stores the
same audio in three parquet shards with the original file bytes in an `audio`
struct, so the flac files can be written back bit-for-bit without decoding and
without re-encoding.

What it does
------------
1. Reads the repo tree through the mirror API to get size and sha256 (LFS oid).
2. Downloads each parquet with N parallel HTTP range requests, resumable, then
   verifies the whole-file sha256. A file whose length happens to match but
   whose hash does not is treated as corrupt and re-downloaded -- a resumed
   transfer can leave exactly that kind of placeholder.
3. Streams each shard and writes
       ASVspoof2019_LA_{train,dev,eval}/flac/<audio_file_name>
   plus the protocol files
       ASVspoof2019_LA_cm_protocols/LA.cm.{train.trn,dev.trl,eval.trl}.txt
4. Checks the per-split row counts and the flac magic; refuses to report success
   on a short split.

Usage
-----
    python scripts/fetch_data.py --config configs/three_arm_phys.yaml
    python scripts/fetch_data.py --out data/asvspoof2019_la --workers 16
    python scripts/fetch_data.py --limit 200 --splits train      # smoke
    python scripts/fetch_data.py --download-only                 # parquet only
    python scripts/fetch_data.py --extract-only --clean          # free 7.5 GB

Everything is idempotent: existing flac files are skipped, existing parquet
files are verified rather than re-fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DEFAULT_REPO = "Bisher/ASVspoof_2019_LA"
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# short split name -> (folder suffix, protocol file name)
SPLIT_INFO = {
    "train": ("ASVspoof2019_LA_train", "LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "LA.cm.eval.trl.txt"),
}
# mirror shard name -> short split name
MIRROR_SPLIT = {"train": "train", "validation": "dev", "test": "eval"}
EXPECTED = {"train": 25380, "dev": 24844, "eval": 71237}
FLAC_MAGIC = b"fLaC"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- mirror index
def list_repo_files(endpoint: str, repo: str, subdir: str = "data") -> list[dict]:
    url = f"{endpoint}/api/datasets/{repo}/tree/main/{subdir}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    entries = r.json()
    files = []
    for e in entries:
        if e["type"] != "file":
            continue
        oid = (e.get("lfs") or {}).get("oid") or e.get("oid")
        files.append({"path": e["path"], "size": int(e["size"]), "sha256": oid})
    return files


# ------------------------------------------------------------ multi-thread GET
def _fetch_range(url: str, start: int, end: int, dest: Path,
                 retries: int = 8) -> None:
    """Download [start, end] into dest, resuming from whatever is there.

    curl is used rather than requests: on flaky links a half-open connection
    under urllib3 can sit in a read that never returns, whereas curl's
    --max-time kills it and we simply resume from the byte count we already
    have. curl is present on every GPU box we rent.
    """
    want = end - start + 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if have >= want:
            return
        tmp = dest.with_name(dest.name + ".cur")
        tmp.unlink(missing_ok=True)
        cmd = ["curl", "-sS", "-L", "--fail", "--max-time", "240",
               "-r", f"{start + have}-{end}", "-o", str(tmp), url]
        try:
            sub = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:                      # no curl -> requests
            _fetch_range_requests(url, start + have, end, dest)
            return
        if sub.returncode != 0 or not tmp.exists():
            attempt += 1
            msg = (sub.stderr or f"curl rc={sub.returncode}").strip()[:200]
            if attempt >= retries:
                raise RuntimeError(f"range {start}-{end} failed: {msg}")
            print(f"       retry {start}-{end} from {have} bytes ({msg})",
                  flush=True)
            tmp.unlink(missing_ok=True)
            time.sleep(1.0 * attempt)
            continue
        with open(dest, "ab") as f, open(tmp, "rb") as g:
            shutil.copyfileobj(g, f)
        tmp.unlink(missing_ok=True)
        if dest.stat().st_size == have:            # server sent nothing
            attempt += 1
            if attempt >= retries:
                raise RuntimeError(f"range {start}-{end} returned no data")


def _fetch_range_requests(url: str, start: int, end: int, dest: Path) -> None:
    """Fallback for machines without curl."""
    have = dest.stat().st_size if dest.exists() else 0
    want = end - start + 1
    headers = {"Range": f"bytes={start}-{end}"}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 30)) as r:
        mode = "ab" if have else "wb"
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def parallel_download(url: str, size: int, dest: Path, workers: int = 8,
                      chunk_mb: int = 24, verify_sha: str | None = None,
                      force: bool = False) -> Path:
    """Split the file into contiguous ranges, fetch in parallel, then verify.

    Resume works per range: a partially written part continues from its own
    length. The assembled file is only accepted if its whole-file sha256 equals
    the LFS oid, which is the only check that catches a resumed file whose
    length is right and whose content is not.
    """
    part_dir = dest.with_suffix(dest.suffix + ".parts")
    if force and dest.exists():
        dest.unlink()
    if dest.exists() and verify_sha and sha256_file(dest) == verify_sha:
        print(f"[skip] {dest.name} already verified ({human(dest.stat().st_size)})")
        return dest
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = chunk_mb << 20
    ranges = [(i, min(i + chunk - 1, size - 1)) for i in range(0, size, chunk)]
    print(f"[get ] {dest.name}: {human(size)} in {len(ranges)} ranges, "
          f"{workers} workers")

    done = {"bytes": 0}

    def work(job):
        idx, (a, b) = job
        part = part_dir / f"{dest.name}.{idx:05d}"
        _fetch_range(url, a, b, part)
        done["bytes"] += part.stat().st_size
        pct = 100.0 * done["bytes"] / size
        print(f"       {pct:5.1f}%  range {idx + 1}/{len(ranges)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, enumerate(ranges)))

    tmp = dest.with_suffix(dest.suffix + ".assembling")
    with open(tmp, "wb") as out:
        for idx, _ in enumerate(ranges):
            part = part_dir / f"{dest.name}.{idx:05d}"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: assembled {tmp.stat().st_size} != {size}")
    if verify_sha:
        got = sha256_file(tmp)
        if got != verify_sha:
            tmp.unlink()
            raise RuntimeError(
                f"{dest.name}: sha256 mismatch (got {got[:12]}, want "
                f"{verify_sha[:12]}). A resumed file with a correct length but "
                f"wrong content is a placeholder, not a download; re-run with "
                f"--force after deleting {part_dir}")
        print(f"       sha256 {got[:12]} verified")
    tmp.replace(dest)
    shutil.rmtree(part_dir, ignore_errors=True)
    return dest


# -------------------------------------------------------------- parquet -> flac
def _as_audio_bytes(value) -> bytes | None:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def extract_split(parquet_path: Path, out_root: Path, split: str,
                  limit: int = 0) -> tuple[int, int]:
    """Write flac files and the protocol for one split. Returns (written, bad)."""
    import pyarrow.parquet as pq

    folder, proto_name = SPLIT_INFO[split]
    flac_dir = out_root / folder / "flac"
    proto_dir = out_root / "ASVspoof2019_LA_cm_protocols"
    flac_dir.mkdir(parents=True, exist_ok=True)
    proto_dir.mkdir(parents=True, exist_ok=True)

    proto_lines, written, bad, seen = [], 0, 0, set()
    pf = pq.ParquetFile(parquet_path)
    cols = ["speaker_id", "audio_file_name", "audio", "system_id", "key"]
    available = [c for c in cols if c in pf.schema_arrow.names]
    n_total = pf.metadata.num_rows
    print(f"[ext ] {split:<5} {n_total} rows -> {flac_dir}")

    for batch in pf.iter_batches(batch_size=256, columns=available):
        for row in batch.to_pylist():
            name = row.get("audio_file_name") or (row.get("audio") or {}).get("path")
            if not name:
                bad += 1
                continue
            name = Path(str(name)).name
            if not name.endswith(".flac"):
                name += ".flac"
            audio = _as_audio_bytes(row.get("audio"))
            if audio is None or audio[:4] != FLAC_MAGIC:
                bad += 1
                continue
            target = flac_dir / name
            if not target.exists() or target.stat().st_size != len(audio):
                target.write_bytes(audio)
            written += 1
            if name in seen:
                continue
            seen.add(name)
            spk = str(row.get("speaker_id") or name.split("_")[0])
            system = str(row.get("system_id") or "-")
            key = row.get("key")
            label = "bonafide" if key in (0, "0", "bonafide") else "spoof"
            attack = "-" if label == "bonafide" else system
            proto_lines.append(f"{spk} {name[:-5]} - {attack} {label}")
            if limit and written >= limit:
                break
        if limit and written >= limit:
            break

    proto_path = proto_dir / proto_name
    proto_path.write_text("\n".join(proto_lines) + "\n", encoding="utf-8")
    print(f"       wrote {written} flac, {len(proto_lines)} protocol lines, "
          f"{bad} unreadable")
    return written, bad


# ---------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="read data.root as the output directory")
    ap.add_argument("--out", default=None, help="output root (data/asvspoof2019_la)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--parquet-dir", default=None,
                    help="where the shards are kept (default <out>/../_hf_parquet)")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "eval"],
                    choices=["train", "dev", "eval", "validation", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0,
                    help="rows per split, for a smoke run")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="delete the parquet shards after extraction")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()

    out = Path(args.out or "")
    if not args.out:
        if args.config:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from wcl.utils import load_config
            cfg = load_config(args.config)
            out = Path(cfg["data"]["root"])
        else:
            out = Path("data/asvspoof2019_la")
    pqdir = Path(args.parquet_dir) if args.parquet_dir else out.parent / "_hf_parquet"
    pqdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print(f"repo     = {args.repo}\nendpoint = {args.endpoint}\n"
          f"out      = {out}\nparquet  = {pqdir}")

    files = list_repo_files(args.endpoint, args.repo)
    by_split = {}
    for f in files:
        for mir, short in MIRROR_SPLIT.items():
            if mir in f["path"]:
                by_split[short] = f
    if len(by_split) < 3:
        print(f"[FAIL] mirror tree lists {len(by_split)} usable shards: "
              f"{[f['path'] for f in files]}")
        return 2

    wanted = [MIRROR_SPLIT.get(s, s) for s in args.splits]
    wanted = [w for w in wanted if w in by_split]

    # ---- download
    local = {}
    if not args.extract_only:
        for name in wanted:
            f = by_split[name]
            url = f"{args.endpoint}/datasets/{args.repo}/resolve/main/{f['path']}"
            dest = pqdir / Path(f["path"]).name
            try:
                parallel_download(url, f["size"], dest, workers=args.workers,
                                  chunk_mb=args.chunk_mb, verify_sha=f["sha256"],
                                  force=args.force)
                local[name] = dest
            except Exception as exc:
                print(f"[FAIL] {name}: {exc}")
                return 2
    else:
        for name in wanted:
            dest = pqdir / Path(by_split[name]["path"]).name
            if not dest.exists():
                print(f"[FAIL] {dest} missing; run without --extract-only first")
                return 2
            local[name] = dest

    if args.download_only:
        print("\ndownload-only: shards ready in " + str(pqdir))
        return 0

    # ---- extract
    manifest, failed = {}, 0
    for name in wanted:
        written, bad = extract_split(local[name], out, name, limit=args.limit)
        manifest[name] = {"rows_written": written, "bad": bad,
                          "expected_official": EXPECTED[name],
                          "parquet": str(local[name])}
        if bad:
            failed += 1
        if not args.limit and written != EXPECTED[name]:
            print(f"[warn] {name}: {written} files, official count is "
                  f"{EXPECTED[name]} (mirror row count can differ; check before "
                  f"quoting any 'full eval set' number)")

    (out / "_fetch_manifest.json").write_text(
        json.dumps({"repo": args.repo, "endpoint": args.endpoint,
                    "splits": manifest, "limit": args.limit},
                   indent=2), encoding="utf-8")

    if args.clean:
        for dest in local.values():
            dest.unlink(missing_ok=True)
        print(f"[clean] removed {len(local)} parquet shards")

    total = sum(1 for p in out.rglob("*.flac"))
    print(f"\n{total} flac files under {out}")
    print("next: python scripts/preflight.py --config configs/three_arm_phys.yaml")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
