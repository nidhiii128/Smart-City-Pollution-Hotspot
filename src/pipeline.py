"""End-to-end pipeline runner.

    python -m src.pipeline                      # ingest -> etl -> hotspots -> forecast -> mongo
    python -m src.pipeline --skip-ingest        # reprocess existing raw data
    python -m src.pipeline --incremental        # near real-time refresh (new hours only)
    python -m src.pipeline --stages etl,hotspots
    python -m src.pipeline --source synthetic   # fully offline run
"""
from __future__ import annotations

import argparse
import json
import logging
import time

STAGES = ["ingest", "etl", "hotspots", "forecast", "mongo"]
log = logging.getLogger("pipeline")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s")
    logging.getLogger("py4j").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {STAGES}")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--source", choices=["auto", "openaq", "hybrid", "openmeteo", "synthetic"])
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages: {unknown}")
    if args.skip_ingest and "ingest" in stages:
        stages.remove("ingest")

    summary, t_all = {}, time.time()
    if "ingest" in stages:
        from src.ingestion import ingest
        t = time.time()
        summary["ingest"] = ingest.run(args.source, args.incremental)
        log.info("ingest done in %.0fs", time.time() - t)

    spark_stages = [s for s in stages if s != "ingest"]
    if spark_stages:
        from src.processing.spark_session import get_spark
        spark = get_spark()
        try:
            for stage in spark_stages:
                t = time.time()
                if stage == "etl":
                    from src.processing import etl
                    summary["etl"] = etl.run(spark)
                elif stage == "hotspots":
                    from src.processing import hotspots
                    summary["hotspots"] = hotspots.run(spark)
                elif stage == "forecast":
                    from src.processing import forecast
                    summary["forecast"] = forecast.run(spark)
                elif stage == "mongo":
                    from src.storage import mongo_store
                    if mongo_store.is_available():
                        summary["mongo"] = mongo_store.load_all(spark)
                    else:
                        log.warning("MongoDB is not reachable - skipping load. Start it with "
                                    "./scripts/start_mongo.sh; the dashboard will read Parquet "
                                    "files meanwhile.")
                        summary["mongo"] = "skipped (not reachable)"
                log.info("%s done in %.0fs", stage, time.time() - t)
        finally:
            spark.stop()

    log.info("Pipeline finished in %.0fs", time.time() - t_all)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
