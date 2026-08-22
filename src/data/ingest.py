"""CLI entrypoint: python -m src.data.ingest"""
from __future__ import annotations

import argparse
import logging

from src.data.etl import run_etl


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL: load synthetic CSVs into PostgreSQL.")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing CSVs")
    parser.add_argument("--database-url", type=str, default=None, help="Optional PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Validate and clean without writing to the database")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = None
    if not args.dry_run:
        from src.database.engine import make_engine
        engine = make_engine(args.database_url)

    run_etl(args.data_dir, engine=engine, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
