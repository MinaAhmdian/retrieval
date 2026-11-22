"""
evidence_cache.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
from pathlib import Path


INPUTS = ("encoders_per_query", "model_per_query", "controls_per_query", "transfer_per_query")
EVIDENCE_COLUMNS = ("src", "retriever", "configuration", "backbone", "seed", "test_corpus", "split",
                    "archive", "rule", "k", "query_inputs", "level", "finding", "metric", "exam_id", "value")


def ensure_evidence(connection, inputs):
    from final_differences import build_db

    paths = {key: Path(inputs[key]) for key in INPUTS}
    signature = [("import_schema", "1")]
    stamps = {}
    for key, path in paths.items():
        stat = path.stat()
        stamps[key] = (stat.st_size, stat.st_mtime_ns)
        hasher = hashlib.sha256()
        with path.open("rb") as raw:
            for block in iter(lambda: raw.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        signature.append((key, hasher.hexdigest()))
    connection.execute("BEGIN TRANSACTION")
    try:
        objects = dict(connection.execute("SELECT table_name, table_type FROM information_schema.tables WHERE table_schema='main'").fetchall())
        if objects.get("ev") == "BASE TABLE" and objects.get("evidence_inputs") == "BASE TABLE":
            columns = tuple(r[1] for r in connection.execute("PRAGMA table_info('ev')").fetchall())
            metadata = tuple(r[1] for r in connection.execute("PRAGMA table_info('evidence_inputs')").fetchall())
            if columns == EVIDENCE_COLUMNS and metadata == ("input_name", "sha256"):
                cached = connection.execute("SELECT input_name, sha256 FROM evidence_inputs").fetchall()
            else:
                cached = []
            if sorted(cached) == sorted(signature):
                connection.execute("COMMIT")
                return
        if "ev" in objects:
            connection.execute("DROP VIEW ev" if objects["ev"] == "VIEW" else "DROP TABLE ev")
        build_db(connection, inputs)
        for key, path in paths.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != stamps[key]:
                raise RuntimeError(f"Input changed during import: {path}")
        if objects.get("evidence_inputs") == "VIEW":
            connection.execute("DROP VIEW evidence_inputs")
        connection.execute("CREATE OR REPLACE TABLE evidence_inputs(input_name VARCHAR PRIMARY KEY, sha256 VARCHAR)")
        connection.executemany("INSERT INTO evidence_inputs VALUES (?, ?)", signature)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
