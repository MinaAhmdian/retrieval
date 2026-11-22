"""
rule_differences.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import duckdb
import pandas as pd

from final_differences import DIFF_COLS, STAT_COLS, arm, run_spec
from config import Config


def main_rule_differences(config_path):
    settings = Config(config_path).get("statistics")
    os.makedirs(os.path.dirname(settings["database"]), exist_ok=True)
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    from evidence_cache import ensure_evidence
    con = duckdb.connect(settings["database"])
    ensure_evidence(con, settings["inputs"])
    con.execute("PRAGMA threads=14")
    sp = []
    frozen = [("biovilt_frozen", ["image_image", "image_report", "two_view_mean"]), ("raddino_frozen", ["image_image", "two_view_mean"]),
              ("biomedclip_frozen", ["image_image", "image_report", "two_view_mean"]), ("medsiglip_frozen", ["image_image", "image_report", "two_view_mean"]),
              ("cxrclip_frozen", ["image_image", "image_report", "two_view_mean"]), ("medimageinsight_frozen", ["image_image", "two_view_mean"])]
    clf = arm(retriever="finding_classifier", configuration="none", backbone="none", seeds=("none",), rule="profile")
    for fr, rules in frozen:
        for r in rules:
            sp.append({"name": f"finding_classifier_minus_{fr}", "a": clf, "b": arm(retriever=fr, configuration="none", backbone="none", seeds=("none",), rule=r)})
    for bb in ["biovilt", "raddino"]:
        sp.append({"name": "two_stage_minus_profile", "a": arm(backbone=bb, rule="two_stage", k=100), "b": arm(backbone=bb, rule="profile")})
        sp.append({"name": "two_stage_minus_alignment", "a": arm(backbone=bb, rule="two_stage", k=100), "b": arm(backbone=bb, rule="alignment")})
        sp.append({"name": "alignment_minus_profile", "a": arm(backbone=bb, rule="alignment"), "b": arm(backbone=bb, rule="profile")})
    diffs, stats = [], []
    for s in sp:
        run_spec(con, s, diffs, stats)
    pd.DataFrame(diffs, columns=DIFF_COLS).to_csv(settings["output_prefix"] + "_rule_diffs.csv", index=False)
    pd.DataFrame(stats, columns=STAT_COLS).to_csv(settings["output_prefix"] + "_rule_stats.csv", index=False)
