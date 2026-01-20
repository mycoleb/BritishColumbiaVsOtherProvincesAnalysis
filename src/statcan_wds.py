from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"
# WDS user guide documents the endpoint pattern for full-table downloads.  :contentReference[oaicite:5]{index=5}


@dataclass(frozen=True)
class StatCanTable:
    """A StatCan table, expressed in the numeric PID format used by WDS full-table downloads.
    Example: 14-10-0287-03 -> 14100287
    """
    pid: str
    lang: str = "en"


def download_full_table_zip(table: StatCanTable, out_dir: Path) -> Path:
    """Download the official full-table ZIP for a given StatCan table PID via WDS."""
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"{WDS_BASE}/getFullTableDownloadCSV/{table.pid}/{table.lang}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "SUCCESS":
        raise RuntimeError(f"WDS call failed: {payload}")

    zip_url = payload["object"]
    z = requests.get(zip_url, timeout=300)
    z.raise_for_status()

    zip_path = out_dir / f"{table.pid}-{table.lang}.zip"
    zip_path.write_bytes(z.content)
    return zip_path


def read_first_csv_from_zip(zip_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Most StatCan full-table ZIPs contain a main CSV + metadata files. We load the first .csv found."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in {zip_path.name}")
        name = csv_names[0]
        with zf.open(name) as f:
            # StatCan tables are typically comma-separated.
            return pd.read_csv(io.TextIOWrapper(f, encoding=encoding), low_memory=False)


def maybe_write(df: pd.DataFrame, path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
