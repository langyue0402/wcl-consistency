"""Offline tests for the mirror fetcher's parquet -> official-layout step.

No network: the parquet shard is synthesised with byte payloads that start with
the flac magic, which is exactly what extract_split validates before writing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from wcl.dataset import parse_protocol  # noqa: E402


def _make_shard(path: Path, n: int = 4) -> Path:
    rows = []
    for i in range(n):
        rows.append({
            "speaker_id": "LA_0079",
            "audio_file_name": f"LA_T_{1000000 + i}",
            "audio": {"bytes": b"fLaC" + bytes(32 + i), "path": f"LA_T_{1000000+i}.flac"},
            "system_id": "-" if i % 2 == 0 else "A07",
            "key": 0 if i % 2 == 0 else 1,
        })
    table = pa.table({
        "speaker_id": [r["speaker_id"] for r in rows],
        "audio_file_name": [r["audio_file_name"] for r in rows],
        "audio": pa.array([r["audio"] for r in rows]),
        "system_id": [r["system_id"] for r in rows],
        "key": pa.array([r["key"] for r in rows], pa.int8()),
    })
    pq.write_table(table, path)
    return path


def test_extract_writes_official_layout(tmp_path):
    import fetch_data

    shard = _make_shard(tmp_path / "train-00000-of-00001.parquet")
    out = tmp_path / "out"
    written, bad = fetch_data.extract_split(shard, out, "train")
    assert (written, bad) == (4, 0)

    flac = sorted(p.name for p in (out / "ASVspoof2019_LA_train" / "flac").iterdir())
    assert flac == [f"LA_T_{1000000 + i}.flac" for i in range(4)]

    proto = out / "ASVspoof2019_LA_cm_protocols" / "LA.cm.train.trn.txt"
    entries = parse_protocol(proto, out / "ASVspoof2019_LA_train" / "flac")
    assert len(entries) == 4
    assert sum(e["label"] for e in entries) == 2          # bonafide count
    assert {e["attack"] for e in entries} >= {"bonafide", "A07"}


def test_extract_rejects_short_or_non_flac_payload(tmp_path):
    import fetch_data

    rows = [
        {"speaker_id": "LA_1", "audio_file_name": "LA_T_1",
         "audio": {"bytes": b"RIFF0000", "path": "LA_T_1.flac"},
         "system_id": "-", "key": 0},
        {"speaker_id": "LA_1", "audio_file_name": "LA_T_2",
         "audio": {"bytes": b"fLaC" + bytes(8), "path": "LA_T_2.flac"},
         "system_id": "A01", "key": 1},
    ]
    table = pa.table({
        "speaker_id": [r["speaker_id"] for r in rows],
        "audio_file_name": [r["audio_file_name"] for r in rows],
        "audio": pa.array([r["audio"] for r in rows]),
        "system_id": [r["system_id"] for r in rows],
        "key": pa.array([r["key"] for r in rows], pa.int8()),
    })
    shard = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(table, shard)
    out = tmp_path / "out"
    written, bad = fetch_data.extract_split(shard, out, "train")
    # the non-flac payload must not be written, and must be counted as bad
    assert written == 1 and bad == 1
    assert [p.name for p in (out / "ASVspoof2019_LA_train" / "flac").iterdir()] == ["LA_T_2.flac"]


def test_extract_is_idempotent(tmp_path):
    import fetch_data

    shard = _make_shard(tmp_path / "train-00000-of-00001.parquet")
    out = tmp_path / "out"
    fetch_data.extract_split(shard, out, "train")
    before = {p.name: p.stat().st_size for p in (out / "ASVspoof2019_LA_train" / "flac").iterdir()}
    written, bad = fetch_data.extract_split(shard, out, "train")
    after = {p.name: p.stat().st_size for p in (out / "ASVspoof2019_LA_train" / "flac").iterdir()}
    assert written == 4 and bad == 0 and before == after
