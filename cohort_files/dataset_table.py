"""
cohort_files/dataset_table.py
Created on November 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import pandas as pd

from config import Config

COLUMNS = ["corpus", "exams", "patients", "images", "test_queries", "test_archive", "share_with_lateral",
           "share_with_context_text", "share_with_age_and_sex"]


def one(df, **keys):
    m = pd.Series(True, index=df.index)
    for k, v in keys.items():
        m &= (df[k].isna()) if v is None else (df[k] == str(v))
    rows = df[m]
    if len(rows) != 1:
        raise LookupError(f"{len(rows)} rows for {keys}")
    return rows.iloc[0]


def dataset_rows(ch, cohort, corpora):
    def q(c, split, quantity):
        return float(one(ch, corpus=c, split=split, quantity=quantity)["value"])

    rows = []
    for c in corpora:
        n = q(c, "all", "n_exams")
        sub = cohort[cohort["corpus"] == c]
        agesex = ((sub["age"].notna()) & (sub["sex"].notna())).mean()
        rows.append({"corpus": c, "exams": int(n), "patients": pd.NA if c == "vindr" else int(q(c, "all", "n_patients")), "images": int(q(c, "all", "n_images")),
                     "test_queries": int(q(c, "test", "n_queries")), "test_archive": int(q(c, "test", "n_database")),
                     "share_with_lateral": q(c, "all", "n_with_lateral") / n,
                     "share_with_context_text": q(c, "all", "n_with_context_text") / n, "share_with_age_and_sex": float(agesex)})
    coh = cohort.astype({"n_frontal": int, "n_lateral": int})
    rows.append({"corpus": "all", "exams": len(coh), "patients": int(coh.loc[coh.corpus != "vindr"].groupby("corpus").patient_id.nunique().sum()),
                 "images": int((coh.n_frontal + coh.n_lateral).sum()), "share_with_lateral": float((coh.n_lateral > 0).mean())})
    counts = {c: "Int64" for c in ("exams", "patients", "images", "test_queries", "test_archive")}
    return pd.DataFrame(rows, columns=COLUMNS).astype(counts)


def finding_support(cohort, labels, minimum):
    rows=[]
    for corpus, data in cohort[cohort.query_or_database.isin(["query","database"])].groupby("corpus"):
        archive = data[data.query_or_database == "database"]
        query = data[data.query_or_database == "query"]
        for label in labels:
            if data[label].isna().all():
                continue
            yq,yd = pd.to_numeric(query[label]),pd.to_numeric(archive[label])
            positive = query[yq == 1]
            known = int((yd.fillna(-2) != -1).sum())
            r=int((yd==1).sum())
            rows.append(dict(corpus=corpus, finding=label, positive_queries=len(positive),
                             positive_query_units=positive.patient_id.nunique(), unit="image" if corpus == "vindr" else "patient",
                             archive_positive=r, archive_evaluated=known, archive_empty_value=int(yd.isna().sum()),
                             archive_uncertain=int((yd == -1).sum()), archive_positive_fraction=r/known if known else float("nan"),
                             eligible_ap=len(positive)>=minimum and r>0))
    return pd.DataFrame(rows)


def main_dataset_table(config_path):
    config = Config(config_path)
    settings = config.get("statistics")
    os.makedirs(os.path.dirname(settings["output_prefix"]), exist_ok=True)
    ch = pd.read_csv(settings["inputs"]["characteristics"], low_memory=False, dtype=str)
    cohort = pd.read_csv(settings["inputs"]["cohort"], low_memory=False, dtype=str)
    table = dataset_rows(ch, cohort, config.get("cohort.corpora"))
    table.to_csv(settings["output_prefix"] + "_dataset_table.csv", index=False)
    labels = list(config.get("cohort.chexpert_label_names").values())
    support = finding_support(cohort, labels, config.get("retrieval.evaluation.min_positive_queries"))
    unjudged = cohort[cohort.corpus == "padchest"][labels].isna().all(axis=1)
    pad = cohort.loc[unjudged.index]
    missing = int((unjudged & (pad.query_or_database == "database")).sum())
    mask = (support.corpus == "padchest") & support.eligible_ap
    support["missing_archive_ap_bound"] = float("nan")
    support.loc[mask, "missing_archive_ap_bound"] = support.loc[mask, "archive_positive"].map(
        lambda r: sum(missing / (j + missing) for j in range(1, r + 1)) / r)
    support.to_csv(settings["output_prefix"] + "_finding_support.csv", index=False)
