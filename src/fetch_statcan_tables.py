from __future__ import annotations

from pathlib import Path

from statcan_wds import StatCanTable, download_full_table_zip, read_first_csv_from_zip, maybe_write


RAW_DIR = Path("data/raw")


# Table IDs shown on StatCan pages include dashes; WDS full-table download uses the numeric PID
# (e.g., 14-10-0287-03 -> 14100287). The WDS user guide shows this pattern. :contentReference[oaicite:6]{index=6}
TABLES = {
    "unemployment_monthly": StatCanTable(pid="14100287", lang="en"),  # 14-10-0287-03 :contentReference[oaicite:7]{index=7}
    "gdp_by_industry": StatCanTable(pid="36100402", lang="en"),        # 36-10-0402-01 :contentReference[oaicite:8]{index=8}
    "population": StatCanTable(pid="17100005", lang="en"),             # 17-10-0005-01 :contentReference[oaicite:9]{index=9}
    "income": StatCanTable(pid="11100091", lang="en"),                 # 11-10-0091-01 :contentReference[oaicite:10]{index=10}
}


def main() -> None:
    for key, table in TABLES.items():
        print(f"Downloading {key} (PID={table.pid}) ...")
        zip_path = download_full_table_zip(table, RAW_DIR)
        print(f"  -> {zip_path}")

        print("Loading CSV from zip ...")
        df = read_first_csv_from_zip(zip_path)

        out_csv = RAW_DIR / f"{key}.csv"
        maybe_write(df, out_csv)
        print(f"  wrote {out_csv} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
