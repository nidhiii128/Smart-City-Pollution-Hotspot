"""SparkSession factory."""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

from pyspark.sql import SparkSession

from src.config import ROOT, load_config


SUPPORTED_JAVA = (21, 17)


def _java_major(java_home: str | None) -> int | None:
    java = os.path.join(java_home, "bin", "java") if java_home else shutil.which("java")
    if not java or not os.path.exists(java):
        return None
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=20).stderr
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 else major  # "1.8" -> 8


def _ensure_java() -> None:
    """PySpark 4 needs Java 17 or 21. Newer JDKs (23+) break Hadoop's login code
    ("getSubject is not supported"), so switch JAVA_HOME to a supported JDK if needed."""
    if _java_major(os.environ.get("JAVA_HOME")) in SUPPORTED_JAVA:
        return
    candidates = []
    for root in ("/usr/lib/jvm", os.path.expanduser("~/.sdkman/candidates/java"),
                 "/opt/java", "/Library/Java/JavaVirtualMachines"):
        candidates += sorted(glob.glob(os.path.join(root, "*")))
    candidates += [os.path.join(c, "Contents", "Home") for c in candidates]
    found = {}
    for path in candidates:
        if os.path.islink(path) and os.path.basename(path) in ("current", "default-java"):
            continue
        major = _java_major(path)
        if major in SUPPORTED_JAVA and major not in found:
            found[major] = path
    for major in SUPPORTED_JAVA:
        if major in found:
            os.environ["JAVA_HOME"] = found[major]
            return
    raise RuntimeError("Spark 4 needs Java 17 or 21 but none was found. Install one, e.g. "
                       "`sudo apt install openjdk-21-jdk` or `sdk install java 21-tem`.")


def get_spark() -> SparkSession:
    _ensure_java()
    # Python workers must use this interpreter (venv) and be able to import `src`
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if str(ROOT) not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, paths)])
    cfg = load_config()["spark"]
    spark = (SparkSession.builder
             .appName(cfg["app_name"])
             .master(cfg["master"])
             .config("spark.driver.memory", cfg["driver_memory"])
             .config("spark.sql.shuffle.partitions", cfg["shuffle_partitions"])
             .config("spark.sql.session.timeZone", "Asia/Kolkata")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark
